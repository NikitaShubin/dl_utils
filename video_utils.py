"""video_utils.py.

********************************************
*       Утилиты для работы с видео и       *
*              изображениями.              *
*                                          *
*                                          *
*                                          *
*                                          *
*                                          *
*   Модуль предоставляет конвейеры         *
* фильтров, чтение и запись видео          *
* (ViRead/ViSave), генерацию кадров        *
* (VideoGenerator), перепаковку битых      *
* записей (clean_remux) и работу с         *
* оптическим потоком (OptFlow).            *
*                                          *
********************************************
.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import av
import cv2
import numpy as np
import scipy.ndimage
from skimage import feature
from skimage.io import imread
from tqdm.auto import tqdm

from utils import (
    ImReadBuffer,
    cv2_img_exts,
    cv2_vid_exts,
    get_file_list,
    isfloat,
    isint,
    mkdirs,
    mpmap,
    overlap_with_alpha,
    rmpath,
    text2img,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Число каналов цветного изображения и измерений одноканального:
_NUM_CHANNELS = 3
_GRAY_NDIM = 2

# Режимы интерполяции scipy.ndimage.map_coordinates:
_MAP_MODE = Literal[
    'reflect',
    'grid-mirror',
    'constant',
    'grid-constant',
    'nearest',
    'mirror',
    'wrap',
    'grid-wrap',
]


def recomp2mp4(
    source_file: str | Path,
    target_file: str | Path,
    *,
    rm_soruce: bool = True,
    quiet: bool = True,
) -> None:
    """Пересжимает исходный файл в mp4.

    Полезно для достижения большей компактности за счёт использования межкадрового
    сжатия и лучшей аппаратной совместимости с различными устройствами воспроизведения.
    """
    source_file, target_file = os.fspath(source_file), os.fspath(target_file)

    # Формируем команду для пересжатия:
    cmd = [
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel',
        'quiet',
        '-i',
        source_file,
        '-c:v',
        'libx264',
        '-preset',
        'slow',
        '-crf',
        '20',
        '-tune',
        'animation',
        target_file,
    ]

    # Прячем stderr и stdout, если надо:
    stdio = subprocess.DEVNULL if quiet else None

    # Выполняем пересжатие:
    exit_code = subprocess.run(  # noqa: S603
        cmd,
        stdout=stdio,
        stderr=stdio,
        check=False,
    ).returncode

    # Если пересжатие прошло с ошибкой:
    if exit_code:
        # Удаляем неудачный файл:
        rmpath(target_file)

        msg = f'Ошибка пересжатия "{source_file}"!'
        raise OSError(msg)

    # Если целевой файл создан, а исходный надо удалить, то удаляем:
    if Path(target_file).is_file():
        if rm_soruce:
            rmpath(source_file)

    else:
        msg = 'Файл не создан, но ошибка не обнаружена!'
        raise NotImplementedError(msg)


def _find_broken_packets(source_file: str | Path) -> tuple[int, int, set[int]]:
    """Находит видео-пакеты, которые декодер не может превратить в кадр.

    Декодирование идёт пакет за пакетом, и только реальная ошибка декодера
    (InvalidDataError) помечает пакет битым - валидные кадры не теряются.
    Возвращает (packet_count, decoded_frames, broken): общее число видео-пакетов,
    число успешно раскодированных кадров и множество номеров битых пакетов.
    """
    broken = set()
    packet_count = 0
    decoded_frames = 0

    with av.open(os.fspath(source_file)) as container:
        stream = container.streams.video[0]

        for packet in container.demux(stream):
            # Служебный пакет конца потока (flush) пропускаем:
            if packet.pts is None:
                continue

            index = packet_count
            packet_count += 1

            try:
                decoded_frames += len(stream.decode(packet))
            except av.InvalidDataError:
                broken.add(index)

        # Дочитываем кадры, задержанные декодером (flush может сам сообщить
        # о битых данных, накопленных в декодере):
        with contextlib.suppress(av.FFmpegError):
            decoded_frames += len(stream.decode(None))

    return packet_count, decoded_frames, broken


def _remux_good_packets(
    source_file: str | Path,
    target_file: str | Path,
    broken: set[int],
) -> int:
    """Копирует все видео-пакеты, кроме помеченных битыми, без пересжатия.

    Возвращает число записанных пакетов. Пиксели не перекодируются: валидные
    пакеты копируются байт-в-байт.
    """
    written = 0

    with (
        av.open(os.fspath(source_file)) as source,
        av.open(os.fspath(target_file), 'w') as target,
    ):
        source_stream = source.streams.video[0]
        target_stream = target.add_stream_from_template(source_stream)

        index = 0
        for packet in source.demux(source_stream):
            # Служебный пакет конца потока (flush) пропускаем:
            if packet.pts is None:
                continue

            current = index
            index += 1

            if current in broken:
                continue

            packet.stream = target_stream
            target.mux(packet)
            written += 1

    return written


def _ensure(*, condition: bool, message: str) -> None:
    """Бросает RuntimeError с сообщением message, если условие не выполнено."""
    if not condition:
        raise RuntimeError(message)


def clean_remux(
    source_file: str | Path,
    target_file: str | Path,
    *,
    rm_source: bool = False,
    quiet: bool = True,
    full_check: bool = True,
) -> tuple[int, int]:
    """Перепаковывает видео, выбрасывая нераскодируемые (битые) кадры.

    В отличие от recomp2mp4, пиксели не пересжимаются: все валидные кадры
    копируются байт-в-байт, а удаляются только пакеты, которые декодер не
    может превратить в кадр (например, битые чанки записи устройства).
    Потеря или изменение валидных данных недопустимы: при любом несовпадении
    счётчиков результат удаляется, а функция бросает RuntimeError.

    Аудио и прочие потоки не переносятся - обрабатывается только видео.

    Возвращает (kept, dropped) - числа сохранённых и выброшенных пакетов.
    При rm_source=True исходный файл удаляется только после успешной записи.
    """
    source_file = os.fspath(source_file)
    target_file = os.fspath(target_file)

    if not Path(source_file).is_file():
        msg = f'Нет файла "{source_file}"!'
        raise FileNotFoundError(msg)

    if quiet:
        av.logging.set_level(av.logging.PANIC)

    # Определяем битые пакеты и сверяем два независимых счётчика:
    packet_count, decoded_frames, broken = _find_broken_packets(source_file)
    kept = packet_count - len(broken)

    if decoded_frames != kept:
        msg = (
            f'Число раскодированных кадров ({decoded_frames}) не совпало '
            f'с числом валидных пакетов ({kept}): нельзя гарантировать '
            'отсутствие потерь!'
        )
        raise RuntimeError(msg)

    # Перепаковываем только валидные пакеты:
    try:
        written = _remux_good_packets(source_file, target_file, broken)
        _ensure(
            condition=written == kept,
            message=f'Записано пакетов ({written}) меньше ожидаемого ({kept})!',
        )

        # Проверяем, что результат полностью раскодируется и не растерял кадры:
        if full_check:
            out_packets, out_frames, out_broken = _find_broken_packets(target_file)
            _ensure(
                condition=(
                    not out_broken and out_packets == kept and out_frames == kept
                ),
                message=(
                    f'Проверка "{target_file}" не пройдена: пакетов '
                    f'{out_packets}, кадров {out_frames}, битых '
                    f'{len(out_broken)} (ожидалось {kept} кадров)!'
                ),
            )

    except Exception:
        # Строгий режим: неполный/повреждённый результат не оставляем:
        rmpath(target_file)
        raise

    # Исходник удаляем только после успешной проверки:
    if rm_source:
        rmpath(source_file)

    return kept, len(broken)


##############################################
# Работа в конвейере (в т.ч. с видеопотоком) #
##############################################


def skip_none(func: Callable[[np.ndarray | None], np.ndarray | None]) -> Callable:
    """Декоратор для __call__-функций в функторах-фильтрах.

    Позволяет пропускать None дальше по конвейеру без применения самих
    функций. Такая возможность позволяет имитировать асинхронность даже
    при синхронной работе конвейера.
    """

    def func_(inp: np.ndarray | None) -> np.ndarray | None:
        return None if inp is None else func(inp)

    return func_


def _threshold_filter(image: np.ndarray, thresh: float) -> np.ndarray:
    """Возвращает бинарное изображение по заданному порогу яркости."""
    return cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]


def _named_filter(name: str) -> Callable[..., np.ndarray]:
    """Возвращает функцию-фильтр по её строковому синониму."""
    synonyms = {
        'rgb2gray': rgb2gray,
        'bgr2gray': bgr2gray,
        'rgb2yuv': rgb2yuv,
        'rgb2bgr': rgb2bgr,
        'bgr2rgb': bgr2rgb,
        'yuv2rgb': yuv2rgb,
        'yuv2bgr': yuv2bgr,
        'yuv2gray': yuv2gray,
        'gray2rgb': gray2rgb,
        'gray2bgr': gray2bgr,
        'im2double': im2double,
        'double2im': double2im,
    }
    if name.lower() not in synonyms:
        msg = f'"{name}" не входит в список синонимов.'
        raise ValueError(msg)
    return synonyms[name.lower()]


class Pipeline:
    """Конвейер фильтров обработки изображений."""

    # Конкатенация списка функций:
    @staticmethod
    def concat(
        image_filter_list: Iterable[object],
    ) -> tuple[list[Callable[..., np.ndarray]], list[str]]:
        """Разворачивает вложенные списки фильтров в плоский список функций."""
        functions, names = [], []

        for image_filter in image_filter_list:
            # Если строка-синоним:
            if isinstance(image_filter, str):
                functions.append(_named_filter(image_filter))
                names.append(image_filter)

            # Если функция/функтор:
            elif callable(image_filter):
                functions.append(image_filter)
                names.append(getattr(image_filter, 'name', 'NonameFunction'))

            # Если список/кортеж функций:
            elif isinstance(image_filter, (tuple, list)):
                functions_, names_ = Pipeline.concat(image_filter)
                functions.extend(functions_)
                names.extend(names_)

            # Если число-порог:
            elif isinstance(image_filter, int):
                functions.append(
                    partial(_threshold_filter, thresh=float(image_filter)),
                )
                names.append(f'threshold_{image_filter}')

            else:
                msg = f'{image_filter} не является подходящим элементом конвеера.'
                raise TypeError(msg)

        return functions, names

    def __init__(
        self,
        image_filter_list: Iterable[object],
        name: str = 'Pipeline',
    ) -> None:
        """Собирает конвейер из переданного списка фильтров."""
        self.functions, self.names = self.concat(image_filter_list)
        self.name = name

    def __call__(self, image: np.ndarray | None = None) -> np.ndarray:
        """Применяет все фильтры конвейера к переданному кадру."""
        image = self.functions[0]() if image is None else self.functions[0](image)

        for image_filter in self.functions[1:]:
            image = image_filter(image)

        return image

    # Применяет конвейер для конвертации файлов:
    def convert(  # noqa: PLR0913
        self,
        inp_file: str | Path,
        out_file: str | Path,
        step: int = 1,
        desc: str | None = 'auto',
        *,
        recompress: bool = True,
        skip_existed: bool = True,
    ) -> None:
        """Применяет конвейер ко всем кадрам видеофайла и записывает результат."""
        inp_path, out_path = Path(inp_file), Path(out_file)

        if skip_existed and out_path.is_file():
            return

        # Если тип конечного файла не поддерживается без пересжатия или
        # небоходимость пересжатия явно задана:
        need_recompress = recompress or out_path.suffix.lower() != '.avi'
        if need_recompress:
            # Определяем имя временного файла:
            tmp_path = out_path.with_name(f'{out_path.stem}_tmp.avi')
            if tmp_path.exists():
                msg = f'Временный файл уже существует: "{tmp_path}"!'
                raise FileExistsError(msg)

        # Если пересжатие не требуется, то временный файл и будет
        # окончательным:
        else:
            tmp_path = out_path

        # Описание прогресс-бара: для 'auto' берём имя исходного файла:
        if desc is not None and desc.lower() == 'auto':
            desc = str(inp_path)

        try:
            # Инициируем чтение и запись видео-файлов. Перебираем все кадры
            # источника, пока они читаются: на битых записях ViRead штатно
            # возвращает None, и цикл завершается без исключения:
            with (
                ViRead(inp_path) as vr,
                ViSave(tmp_path) as vs,
                tqdm(
                    total=vr.total_frames,
                    desc=desc,
                    disable=desc is None,
                ) as progress,
            ):
                frame_ind = 0
                while (frame := vr()) is not None:
                    # Если номер кадра соответствует текущему шагу, то
                    # обрабатываем и записываем его:
                    if frame_ind % step == 0:
                        vs(self(frame))

                    frame_ind += 1
                    progress.update()

        except Exception as e:
            msg = 'Обработка файла прервана из-за ошибки!'
            raise RuntimeError(msg) from e

        # Если файл требуется пересжать:
        if need_recompress:
            recomp2mp4(tmp_path, out_path)

    # Сброс всех использующихся фильтров:
    def reset(self, im_size: tuple[int, int] | None = None) -> None:  # noqa: ARG002
        """Сбрасывает состояние всех фильтров конвейера."""
        for image_filter in self.functions:
            if hasattr(image_filter, 'reset'):
                image_filter.reset()


class ViRead:
    """Возвращает последовательность кадров видео из файла.

    Работает и как функция, и как генератор.
    """

    # Сколько раз пытаться восстановить чтение после сбоя декодера:
    RECOVERY_ATTEMPTS = 8

    def __init__(
        self,
        path: str | Path | list[str] | tuple[str, ...] | set[str],
        start_frame: int | list[int] | tuple[int, int] = 0,
        colorspace: str = 'rgb',
        on_end: str = 'close',
    ) -> None:
        """Сохраняет источник и параметры чтения кадров."""
        self.path = path
        self.start_frame = start_frame
        self.colorspace = colorspace.lower()
        self.on_end = on_end.lower()
        self._rng = np.random.default_rng()
        self.reset()

    def _resolve_start_frame(self) -> int:
        """Вычисляет стартовый кадр с учётом отрицательных значений."""
        if isinstance(self.start_frame, int):
            start_frame = self.start_frame
            if start_frame < 0:
                start_frame += self.total_frames
        elif isinstance(self.start_frame, (list, tuple)):
            start, end = self.start_frame[:2]
            start = start if start >= 0 else self.total_frames + start
            end = end if end >= 0 else self.total_frames + end
            if start >= end:
                self.close()
                msg = f'В "{self.source_path}" всего {self.total_frames} кадров.'
                raise ValueError(
                    msg,
                )
            start_frame = int(self._rng.integers(start, end + 1))
        else:
            msg = 'Параметр "start_frame" должен быть числом или диапазоном.'
            raise TypeError(msg)
        return int(start_frame)

    def reset(self) -> None:
        """Открывает источник и переводит чтение в его начало."""
        if hasattr(self, 'cap'):
            self.close()

        # Определяем преобразователь из BGR в заданную цветовую схему:
        converters = {
            'yuv': bgr2yuv,
            'gray': bgr2gray,
            'rgb': bgr2rgb,
            'bgr': None,
        }
        if self.colorspace not in converters:
            msg = f'"{self.colorspace}" не входит в список доступных цветовых схем.'
            raise ValueError(
                msg,
            )
        self._converter = converters[self.colorspace]

        # Определяем файл-источник; из набора путей выбираем случайный:
        source = self.path
        if isinstance(source, (list, tuple, set)):
            source = self._rng.choice([os.fspath(path) for path in source])
        self.source_path = os.fspath(source)

        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            msg = f'Ошибка открытия файла "{self.source_path}"'
            raise ValueError(msg)

        # Число кадров из заголовка контейнера может отличаться от реально
        # декодируемого на битых записях:
        self.header_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = self.header_frames

        start_frame = self._resolve_start_frame()
        if start_frame < 0 or start_frame >= self.total_frames:
            self.close()
            msg = f'В "{self.source_path}" всего {self.total_frames} кадров.'
            raise ValueError(
                msg,
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.position = start_frame
        self.last_frame: np.ndarray | None = None

    def close(self) -> None:
        """Освобождает ресурсы видеозахвата."""
        self.cap.release()

    def count_frames(self) -> int:
        """Считает реально декодируемые кадры дешёвым проходом grab().

        В отличие от total_frames (заголовок контейнера), не декодирует
        пиксели и корректнее ведёт себя на битых записях.
        """
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            msg = f'Ошибка открытия файла "{self.source_path}"'
            raise ValueError(msg)
        total = 0
        while cap.grab():
            total += 1
        cap.release()
        return total

    def _read_frame(self) -> np.ndarray | None:
        """Читает очередной кадр; при сбое декодера пропускает битые пакеты.

        Пробует прочитать следующий кадр ограниченное число раз.
        """
        for _ in range(self.RECOVERY_ATTEMPTS):
            ret, frame = self.cap.read()
            if ret:
                self.position += 1
                converter = self._converter
                return frame if converter is None else converter(frame)

            # Декодер споткнулся: переходим к следующему кадру:
            self.position += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.position)
        return None

    def __call__(self) -> np.ndarray | None:
        """Читает следующий кадр с учётом поведения на конце потока."""
        frame = self._read_frame()
        if frame is not None:
            self.last_frame = frame
            return frame

        if self.on_end == 'reset':
            self.reset()
            return self()
        if self.on_end == 'close':
            self.close()
            return None
        if self.on_end == 'repeat_last_frame':
            return self.last_frame
        msg = f'Параметр "on_end" не может быть равен "{self.on_end}".'
        raise ValueError(msg)

    def __len__(self) -> int:
        """Число кадров по заголовку контейнера."""
        return self.total_frames

    def __iter__(self) -> Self:
        """Возвращает сам объект как итератор."""
        return self

    def __next__(self) -> np.ndarray:
        """Возвращает очередной кадр или завершает итерацию."""
        frame = self()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self) -> Self:
        """Возвращает сам объект для использования в with."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Освобождает ресурсы при выходе из with."""
        self.close()


class ViSave:
    """Покадрово записывает видеофайл."""

    def __init__(
        self,
        path: str | Path,
        colorspace: str = 'rgb',
        fps: float = 30.0,
    ) -> None:
        """Сохраняет путь, цветовую схему и частоту кадров."""
        self.path = os.fspath(path)
        self.fps = fps

        # Определяем преобразователь из заданной цветовой схемы в BGR:
        converters = {
            'yuv': yuv2bgr,
            'gray': gray2bgr,
            'rgb': rgb2bgr,
            'bgr': None,
        }
        colorspace = colorspace.lower()
        if colorspace not in converters:
            msg = f'"{colorspace}" не входит в список доступных цветовых схем.'
            raise ValueError(
                msg,
            )
        self._converter = converters[colorspace]

        # Создаём путь до файла, если надо:
        path_dir = Path(self.path).resolve().parent
        if not path_dir.is_dir():
            mkdirs(path_dir)

    def close(self) -> None:
        """Освобождает ресурсы видеозаписи."""
        if hasattr(self, 'wrt'):
            self.wrt.release()

    def __call__(self, frame: np.ndarray) -> None:
        """Записывает очередной кадр в файл."""
        if not hasattr(self, 'wrt'):
            self.wrt = cv2.VideoWriter(
                self.path,
                cv2.VideoWriter_fourcc(*'MJPG'),  # type: ignore[attr-defined]
                self.fps,
                (frame.shape[1], frame.shape[0]),
            )
        converter = self._converter
        self.wrt.write(frame if converter is None else converter(frame))

    def __enter__(self) -> Self:
        """Возвращает сам объект для использования в with."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Освобождает ресурсы при выходе из with."""
        self.close()


class AsType:
    """Меняет тип тензора."""

    def __init__(self, dtype: str | np.dtype) -> None:
        """Сохраняет целевой тип данных."""
        self.dtype = dtype

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Приводит изображение к целевому типу."""
        return img.astype(self.dtype)


class Resize:
    """Изменяет размер изображения."""

    def __init__(
        self,
        im_size: float | tuple[float, ...] | list | np.ndarray = 512,
    ) -> None:
        """Сохраняет целевой размер изображения."""
        self.reset_im_size(im_size)

    def reset_im_size(
        self,
        im_size: float | tuple[float, ...] | list | np.ndarray,
    ) -> None:
        """Изменяет целевой размер изображения."""
        # Если задан не кортеж/список/numpy-массив, дублируем его:
        if not isinstance(im_size, (tuple, list, np.ndarray)):
            im_size = (im_size, im_size)

        self.im_size = tuple(im_size)

    @staticmethod
    def adopt_axis_size(target_size: float, source_size: int) -> int:
        """Перерасчитывает размер оси изображения.

        Нужен, например, для перехода от относительного размера к
        абсолютному.
        """
        if isint(target_size):
            return int(target_size)

        if isfloat(target_size):
            return int(target_size * source_size)

        msg = (
            'Параметр "im_size" должен быть задан целыми '
            f'или вещественными числами. Получено {target_size}!'
        )
        raise ValueError(msg)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Изменяет размер изображения до целевого."""
        # Получаем размер входного изображения:
        im_size = img.shape[:2]

        # Перерассчитываем итоговый размер изображения:
        target_size = [
            self.adopt_axis_size(self.im_size[i], im_size[i]) for i in range(2)
        ]

        return cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_AREA)


class ResizeAndRestore:
    """Меняет размер изображения до обработки и возвращает его обратно.

    Перед применением заданного обработчика изображение приводится к
    заданному размеру, а результат возвращается к исходному размеру.
    """

    def __init__(
        self,
        im_size: float | tuple[float, ...],
        processor: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Сохраняет размер и обработчик изображения."""
        self.preprocessor = Resize(im_size)
        self.processor = processor
        self.postprocessor = Resize()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Применяет обработчик с временной сменой размера."""
        # Обновляем итоговый размер, если входной размер изменился
        im_size = img.shape[:2]
        if tuple(im_size) != tuple(self.postprocessor.im_size):
            self.postprocessor.reset_im_size(im_size)

        # Применяем всю цепь преобразований:
        out = self.preprocessor(img)
        out = self.processor(out)
        return self.postprocessor(out)

    def reset(self) -> None:
        """Сбрасывает состояние обработчика, если он сбрасываемый."""
        if hasattr(self.processor, 'reset'):
            self.processor.reset()


class AddCaption:
    """Наносит контрастный текст на изображение."""

    def __init__(self, text: str, scale: float = 0.6) -> None:
        """Сохраняет текст и масштаб шрифта."""
        self.text = text
        self.scale = scale
        self.mask: np.ndarray | None = None
        self.reset()

    @staticmethod
    def new_mask(text: str, im_size: tuple[int, int], scale: float) -> np.ndarray:
        """Создаёт RGBA-маску для нанесения контрастного текста."""
        # Растеризируем текст:
        mask = text2img(text, im_size[:2], scale)

        # Строим альфаканал как дилатированную версию текста:
        mask_alpha = mask.copy()

        # Размазываем по вертикали через сдвиги:
        mask_alpha_1 = np.roll(mask_alpha, -1, 0)
        mask_alpha_2 = np.roll(mask_alpha, 1, 0)
        mask_alpha = np.dstack([mask_alpha, mask_alpha_1, mask_alpha_2])
        mask_alpha = mask_alpha.max(-1)

        # Размазываем по горизонтали через сдвиги:
        mask_alpha_1 = np.roll(mask_alpha, -1, 1)
        mask_alpha_2 = np.roll(mask_alpha, 1, 1)
        mask_alpha = np.dstack([mask_alpha, mask_alpha_1, mask_alpha_2])
        mask_alpha = mask_alpha.max(-1)

        # В результате получится белый текст с чёрной обводкой.

        return np.dstack([mask, mask_alpha])

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Наносит текст на изображение."""
        # Если размер изображения изменился, или это первый запуст после сброса:
        if self.mask is None or self.mask.shape[:2] != img.shape[:2]:
            # Сбрасываем маску ещё раз:
            self.reset()

            # Векторизируем текст (создаём предварительную маску):
            mask = self.new_mask(self.text, img.shape[:2], self.scale)

            # Накладываем текст на изображение и фиксируем итоговую маску:
            img, self.mask = overlap_with_alpha(
                img,
                mask,
                return_with_watermark=True,
            )

        # Если маска уже есть, и она адекватна, то сразу используем её:
        else:
            img = overlap_with_alpha(img, self.mask)

        return img

    def reset(self) -> None:
        """Сбрасывает предварительно созданную маску."""
        self.mask = None


class PutTime:
    """Наносит на изображение время, либо (если fps не задан) номер кадра."""

    def __init__(
        self,
        fps: float | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Сохраняет fps и дополнительные параметры нанесения текста."""
        self.fps = fps
        self.args = args
        self.kwargs = kwargs
        self.reset()

    @staticmethod
    def seconds2sexagesimal(seconds: float) -> str:
        """Превращает секунды в строку в человеческом формате времени."""
        minutes = int(seconds // 60)
        seconds -= minutes * 60
        text = f'{seconds:06.3f}'
        if not minutes:
            return text

        hours = minutes // 60
        minutes -= hours * 60
        text = f'{minutes:02d}:' + text
        if not hours:
            return text

        days = hours // 24
        hours -= days * 24
        text = f'{hours:02d}:' + text
        if not days:
            return text
        return f'{days} {text}'

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Наносит на изображение время или номер кадра."""
        self.frame += 1  # Прирост счётчика кадров

        # Определяем содержимое строки вывода:
        if self.fps:
            text = self.seconds2sexagesimal(self.frame / self.fps)
        else:
            text = str(self.frame)

        # Наносим текст на изображение и возвращаем результат:
        return text2img(text, img, *self.args, **self.kwargs)

    def reset(self) -> None:
        """Сбрасывает счётчик кадров."""
        self.frame = 0


class Res:
    """Объединяет входное и выходное изображения для заданного конвейера."""

    def __init__(
        self,
        inner_pipeline: Callable[[np.ndarray], np.ndarray],
        mode: str = 'h',
    ) -> None:
        """Сохраняет конвейер и способ объединения."""
        self.pl = inner_pipeline
        self.mode = mode.lower()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Возвращает коллаж из входного и выходного изображений."""
        # Получаем входное и выходное изображения:
        inp = image
        out = self.pl(image)

        # Вход и выход должны быть одинакового типа:
        if inp.dtype != out.dtype:
            msg = f'{inp.dtype} != {out.dtype}'
            raise TypeError(msg)

        # Определяем способ объединения изображений в зависимости от параметра mode:
        concat = np.hstack if self.mode == 'h' else np.vstack

        # Возвращаем результат объединения:
        return concat([inp, out])


class Mix:
    """Накладывает выходное изображение на входное для вложенного конвейера."""

    def __init__(
        self,
        inner_pipeline: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.5,
    ) -> None:
        """Сохраняет конвейер и прозрачность наложения."""
        self.pl = inner_pipeline
        self.alpha = alpha

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Возвращает смесь входного и выходного изображений."""
        # Получаем входное и выходное изображения:
        inp = image
        out = self.pl(image)

        # Вход и выход должны быть одинакового типа и размера
        if inp.dtype != out.dtype:
            msg = f'{inp.dtype} != {out.dtype}'
            raise TypeError(msg)
        if inp.shape != out.shape:
            msg = f'{inp.shape} != {out.shape}'
            raise ValueError(msg)

        # Накладываем полупрозрачный выход на вход:
        mix = inp * (1.0 - self.alpha) + out * self.alpha

        # Возвращаем приведённый к нужному типу результат:
        return mix.astype(inp.dtype)


class Concat:
    """Объединяет изображения для списка вложенных конвейеров."""

    def __init__(
        self,
        pipelines: Iterable[Callable[[np.ndarray], np.ndarray] | None],
        mode: str = 'h',
    ) -> None:
        """Сохраняет конвейеры и способ объединения."""
        # pipelines должен быть списком или кортежем:
        if not isinstance(pipelines, (list, tuple)):
            msg = 'Параметр "pipelines" должен быть списком или кортежем.'
            raise TypeError(msg)

        self.pls = pipelines
        self.mode = mode.lower()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Возвращает коллаж изображений от всех конвейеров."""
        # Применяем все фильтры к исходному изображению:
        outs = [image if f is None else f(image) for f in self.pls]
        # Если вместо фильтра в списке стоит None, ...
        # ... то просто повторяем входное изображение.

        # Определяем способ объединения изображений в зависимости от параметра mode:
        concat = np.hstack if self.mode == 'h' else np.vstack

        # Возвращаем результат объединения:
        return concat(outs)


class CompareTwoFilters:
    """Выводит демо-коллаж из четырёх изображений.

    Раскладка в следующем виде::

        <Входное изображение             > <Выход из Filter1>
        <diff_func от выходов Filter1 и 2> <Выход из Filter2>

    Полезно для отладки и сравнения фильтров.
    """

    def __init__(
        self,
        filter1: Callable[[np.ndarray], np.ndarray],
        filter2: Callable[[np.ndarray], np.ndarray],
        diff_func: Callable[..., np.ndarray] = cv2.absdiff,
        name: str = 'Comparator',
    ) -> None:
        """Сохраняет фильтры и функцию сравнения."""
        self.f1 = filter1
        self.f2 = filter2
        self.diff = diff_func
        self.name = name

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Возвращает коллаж с результатами обоих фильтров."""
        out1 = self.f1(img)
        out2 = self.f2(img)
        diff = self.diff(out1, out2)

        return np.vstack([np.hstack([img, out1]), np.hstack([diff, out2])])

    def reset(self) -> None:
        """Сбрасывает внутренние состояния фильтров."""
        for f in [self.f1, self.f2]:
            if hasattr(f, 'reset'):
                f.reset()


class CompareFiltersWithTarget:
    """Выводит демо-коллаж сравнения фильтров с эталоном.

    Раскладка в следующем виде::

        <Входное изображение  > <Выход из Filter1> ... <Выход из FilterN>
        <Эталонное изображение> <diff1>            ... <diffN>

    На вход должен подаваться коллаж из входного и эталонного
    изображений, расположенных одно под другим и совпадающих по высоте.

    Полезно для отладки и сравнения фильтров.
    """

    def __init__(
        self,
        filters: Iterable[Callable[[np.ndarray], np.ndarray]],
        diff_func: Callable[..., np.ndarray] = cv2.absdiff,
        name: str = 'TargetComparator',
    ) -> None:
        """Сохраняет фильтры и функцию сравнения."""
        self.filters = filters
        self.diff = diff_func
        self.name = name

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Возвращает коллаж с результатами всех фильтров."""
        # Разделяем изображение на входное и эталонное:
        inp, out = np.vsplit(img, 2)

        # Выполняем фильтрацию:
        preds = [f(inp) for f in self.filters]

        # Вычисляем отличия от эталона:
        diffs = [self.diff(out, pred) for pred in preds]

        # Собираем в коллаж и возвращаем:
        return np.vstack([np.hstack([inp, *preds]), np.hstack([out, *diffs])])

    def reset(self) -> None:
        """Сбрасывает внутренние состояния фильтров."""
        for f in self.filters:
            if hasattr(f, 'reset'):
                f.reset()


class KerasModel:
    """Использование keras-модели как фильтра."""

    def __init__(self, model: object, name: str = 'KerasModel') -> None:
        """Сохраняет keras-модель."""
        self.model = model
        self.name = name
        self.reset()
        self.stateful = getattr(model, 'stateful', None)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Применяет модель к изображению."""
        predict = self.model.predict  # type: ignore[attr-defined]
        return predict(np.expand_dims(image, 0), verbose=0)[0, ...]

    def reset(self) -> None:
        """Сбрасывает состояние модели, если это возможно."""
        reset_states = getattr(self.model, 'reset_states', None)
        if reset_states is not None:
            reset_states()
        # В Keras3 нет reset_states для моделей
        # https://github.com/keras-team/keras/issues/18467#issuecomment-2096448735

    def close(self) -> None:
        """Восстанавливает исходное состояние модели."""
        if hasattr(self.model, 'stateful'):
            self.model.stateful = self.stateful
        self.reset()


class StoreLastFrames:
    """Возвращает не только текущий кадр, но и n-1 предыдущих."""

    def __init__(
        self,
        n: int = 3,
        fill_empty_frames: str | None = None,
        name: str | None = None,
    ) -> None:
        """Сохраняет число кадров и способ заполнения пропусков."""
        self.n = n

        if fill_empty_frames and fill_empty_frames.lower() != 'none':
            self.fill_empty_frames: str | None = fill_empty_frames.lower()
        else:
            self.fill_empty_frames = None

        self.name = name or f'StoreLast{n}Frames'
        self.frames: list[np.ndarray] = []
        self.reset()

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        """Добавляет кадр и возвращает последние n кадров."""
        self.frames.insert(0, image)
        if len(self.frames) == 1 and self.n > 1 and self.fill_empty_frames:
            if self.fill_empty_frames == 'zeros':
                self.frames += [np.zeros_like(image)] * (self.n - 1)
            elif self.fill_empty_frames == 'copy':
                self.frames += [image] * (self.n - 1)

        while len(self.frames) > self.n:
            self.frames.pop()

        return self.frames

    def reset(self) -> None:
        """Очищает буфер кадров."""
        self.frames = []


class DrawSemanticSegments:
    """Рисует цветные сегменты для семантической сегментации.

    Если сегментаторм задан, то исходное изображение раскрашивается цветами
    сегментов. Если сегментатор не указан, то входящее изображение принимается
    за результат сегментации и сегменты отрисовываются без фона.

    Если результат сегментации уже пропущен через argmax, то требуется задать
    число классов (num_classes). Если число классов не задано, то к результату
    сегментации будет применён argmax для последнего измерения.

    Если num_classes не задан, а число каналов тензора сегментации = 1, то вся
    постобработка будет исходить из задачи бинарной классификации с порогом =
    0.5.
    """

    # Размерности и число каналов входного изображения:
    COLOR_NDIM = 3
    GRAY_NDIM = 2
    COLOR_CHANNELS = 3
    GRAY_CHANNELS = 1

    # Параметры бинарной сегментации:
    BINARY_CLASSES = 2
    BINARY_THRESHOLD = 0.5

    def __init__(
        self,
        num_classes: int | None = None,
        segmentator: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Сохраняет число классов и сегментатор."""
        self.num_classes = num_classes
        self.segmentator = segmentator

    @staticmethod
    def _brightness(img: np.ndarray) -> tuple[np.dtype, float | np.ndarray]:
        """Возвращает тип выхода и яркостный канал исходного изображения."""
        cls = DrawSemanticSegments
        if img.ndim == cls.COLOR_NDIM:
            channels = img.shape[2]
            if channels == cls.COLOR_CHANNELS:
                return img.dtype, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if channels == cls.GRAY_CHANNELS:
                return img.dtype, img[..., 0]
        elif img.ndim == cls.GRAY_NDIM:
            return img.dtype, img

        msg = f'Неожиданный размер входного изображения: {img.shape}'
        raise ValueError(msg)

    @staticmethod
    def _prepare_segments(
        segments: np.ndarray,
        im_size: tuple[int, ...],
    ) -> tuple[np.ndarray, int]:
        """Приводит тензор сегментов к карте классов и их числу."""
        cls = DrawSemanticSegments
        # Если используется бинарная сегментация:
        if segments.ndim == cls.GRAY_NDIM or segments.shape[2] == cls.GRAY_CHANNELS:
            binary = segments > cls.BINARY_THRESHOLD
            return binary.reshape(im_size).astype(np.int8), cls.BINARY_CLASSES

        # Если классов больше двух:
        if segments.ndim == cls.COLOR_NDIM and segments.shape[2] > cls.BINARY_CLASSES:
            classes = segments.shape[2]
            return np.argmax(segments, axis=-1), classes

        msg = f'Неожиданный размер тензора сегментов: {segments.shape}'
        raise ValueError(msg)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Рисует цветные сегменты для семантической сегментации."""
        # Фиксируем размер изображения:
        im_size = img.shape[:2]

        # Инициируем выходное изображение:
        out: np.ndarray = np.zeros([*im_size, self.COLOR_CHANNELS], np.uint8)

        # Тип выходного изображения и яркость берём по входному:
        if self.num_classes:
            dtype, value = self._brightness(img)
        else:
            dtype, value = np.dtype(np.uint8), 255.0

        # Применяем сегментатор или берём готовый результат из входа:
        segments = img if self.segmentator is None else self.segmentator(img)

        # Применяем к тензору сегментов постобработку, если она нужна:
        if self.num_classes:
            num_classes = self.num_classes
        else:
            segments, num_classes = self._prepare_segments(segments, im_size)

        # Определяем оттенок по результатам сегментации:
        hue = segments / num_classes
        hue = (hue * 180).astype(np.uint8) if dtype == np.uint8 else hue * 360

        # Для выходного изображения используется цветовое пространство HSV:
        out[..., 0] = hue
        out[..., 2] = value
        out[..., 1] = 255 if dtype == np.uint8 else 1.0

        # Возвращаем результат, сконвертированный из HSV в RGB:
        return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние сегментатора."""
        reset = getattr(self.segmentator, 'reset', None)
        if reset is not None:
            reset()


class StreamRandomCrop:
    """Вырезает случайный фрагмент из целой видеопоследовательности.

    Рамка меняет своё положение только при вызове reset или изменении
    размера входного изображения.
    """

    # Относительный размер обрезки задаётся в интервале (0, 1]:
    MIN_RELATIVE_SIZE = 0.0
    MAX_RELATIVE_SIZE = 1.0

    def __init__(
        self,
        size: float | tuple[float, ...] | list | np.ndarray = 256,
    ) -> None:
        """Сохраняет размер вырезаемого фрагмента."""
        size_arr: np.ndarray = np.asarray(size)
        if size_arr.size == 1:
            size_arr = np.asarray([size_arr] * 2)
        self.size: np.ndarray = size_arr
        self.im_size: np.ndarray | None = None
        self.true_size: np.ndarray = np.zeros(2, dtype=int)
        self.di = 0
        self.dj = 0
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        """Забывает положение рамки."""
        self.im_size = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Возвращает случайно вырезанный фрагмент изображения."""
        im_size = np.array(image.shape[:2])
        if self.im_size is None or not np.all(np.equal(self.im_size, im_size)):
            self.im_size = im_size

            isint = np.issubdtype(self.size.dtype, np.uint64) or np.issubdtype(
                self.size.dtype,
                np.int64,
            )
            isfloat = np.issubdtype(self.size.dtype, np.float64)
            if isint and not isfloat:
                if np.any(self.im_size < self.size):
                    msg = (
                        'Размер изображения должен быть '
                        f'больше-равен {tuple(self.size)}.'
                    )
                    raise ValueError(msg)
                self.true_size = self.size.copy()
            elif not isint and isfloat:
                if np.any(self.size > self.MAX_RELATIVE_SIZE) or np.any(
                    self.size <= self.MIN_RELATIVE_SIZE,
                ):
                    msg = (
                        'Размер обрезки должен быть в интервале (0, 1] '
                        'от размера исходного изображения.'
                    )
                    raise ValueError(msg)
                self.true_size = (self.im_size * self.size).astype(
                    self.im_size.dtype,
                )
            else:
                msg = 'Параметр size должен быть вещественного или целочисленного типа'
                raise ValueError(msg)

            self.di = int(self._rng.integers(self.im_size[0] - self.true_size[0]))
            self.dj = int(self._rng.integers(self.im_size[1] - self.true_size[1]))

        return image[
            self.di : self.di + self.true_size[0],
            self.dj : self.dj + self.true_size[1],
            ...,
        ]


class StreamRandomFlip:
    """Отражает и/или поворачивает изображение.

    Применяет обратимую аугментацию; параметры преобразования меняются
    только при вызове reset или изменении размера входного изображения.
    """

    # Число возможных поворотов на 90 градусов:
    NINETY_DEGREES_STEPS = 4

    def __init__(
        self,
        *,
        ud: bool | None = None,
        lr: bool | None = None,
        rot: int | None = None,
    ) -> None:
        """Сохраняет настройки отражения и поворота."""
        self.ud = ud
        self.lr = lr
        self.rot = rot
        self.im_size: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        """Перевыбирает случайные параметры преобразования."""
        self.im_size = None
        self._ud = self._rng.choice([True, False]) if self.ud is None else self.ud
        self._lr = self._rng.choice([True, False]) if self.lr is None else self.lr
        self._rot = (
            int(self._rng.integers(self.NINETY_DEGREES_STEPS))
            if self.rot is None
            else self.rot
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Применяет случайную обратимую аугментацию."""
        im_size = np.array(image.shape[:2])
        if self.im_size is None or not np.all(np.equal(self.im_size, im_size)):
            self.reset()
            self.im_size = im_size

        if self._ud:
            image = np.flipud(image)
        if self._lr:
            image = np.fliplr(image)
        if self._rot:
            image = np.rot90(image, self._rot)

        return image


# RGB/BGR <-> YUV:
def rgb2yuv(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из RGB в YUV."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def yuv2rgb(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из YUV в RGB."""
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)


def bgr2yuv(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из BGR в YUV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def yuv2bgr(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из YUV в BGR."""
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)


def yuv2gray(img: np.ndarray) -> np.ndarray:
    """Берёт яркостный канал Y из изображения в YUV."""
    return img[:, :, 0]


def rgb2bgr(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из RGB в BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


bgr2rgb = rgb2bgr


# RGB/BGR <-> Gray:
def rgb2gray(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из RGB в оттенки серого."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bgr2gray(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из BGR в оттенки серого."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray2rgb(img: np.ndarray) -> np.ndarray:
    """Переводит одноканальное изображение в RGB."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def gray2bgr(img: np.ndarray) -> np.ndarray:
    """Переводит одноканальное изображение в BGR."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# uint8 <-> float32:
def im2double(img: np.ndarray) -> np.ndarray:
    """Переводит изображение из uint8 в float32 в диапазоне [0, 1]."""
    return img.astype(np.float32) / 255.0


def double2im(double: np.ndarray) -> np.ndarray:
    """Переводит изображение из float32 в uint8."""
    return (double * 255).astype(np.uint8)


def color_canny(image: np.ndarray) -> np.ndarray:
    """Считает трёхканальный детектор границ Кэнни."""
    frame0 = feature.canny(image[:, :, 0])
    frame1 = feature.canny(image[:, :, 1])
    frame2 = feature.canny(image[:, :, 2])
    return 1 - np.stack([frame0, frame1, frame2], -1)


def _resolve_filter_name(im_filter: object) -> str:
    """Возвращает имя фильтра или заглушку."""
    return getattr(im_filter, 'name', 'NoName_Filter')


def _to_3channels_rgb(frame: np.ndarray) -> np.ndarray:
    """Приводит кадр к трёхканальному RGB-формату."""
    if frame.ndim == _GRAY_NDIM:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[..., 0]
    elif frame.shape[2] == _NUM_CHANNELS:
        return frame
    else:
        msg = f'Формат кадра ({frame.shape}) не соответствует ожидаемому.'
        raise ValueError(msg)
    return np.stack([gray] * _NUM_CHANNELS, -1)


def _to_3channels_bgr(frame: np.ndarray) -> np.ndarray:
    """Приводит кадр к трёхканальному BGR-формату."""
    return _to_3channels_rgb(frame)[..., ::-1]


def _filter_with_oom_fallback(
    im_filter: Callable[[np.ndarray], np.ndarray],
    image: np.ndarray,
    *,
    rescale: bool | int | tuple[int, int] | list[int],
) -> np.ndarray:
    """Применяет фильтр, понижая разрешение при нехватке памяти."""
    if rescale:
        if isinstance(rescale, (list, tuple)):
            target_size = tuple(rescale)
        elif isinstance(rescale, int) and not isinstance(rescale, bool):
            target_size = (
                round(image.shape[1] / rescale) * rescale,
                round(image.shape[0] / rescale) * rescale,
            )
        else:
            target_size = (1280, 720)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    while True:
        if hasattr(im_filter, 'reset'):
            im_filter.reset()
        try:
            return im_filter(image)
        except RuntimeError:
            # OOM: понижаем разрешение изображения вдвое по каждой стороне:
            target_size = (round(image.shape[1] / 2), round(image.shape[0] / 2))
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def _rescale_target_size(
    cap: cv2.VideoCapture,
    *,
    rescale: bool | int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Определяет целевой размер кадра при масштабировании."""
    if rescale is True:
        return (1280, 720)
    if isinstance(rescale, int):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (round(width / rescale) * rescale, round(height / rescale) * rescale)
    if isinstance(rescale, (list, tuple)):
        return int(rescale[0]), int(rescale[1])
    msg = 'Параметр rescale задан неверно.'
    raise ValueError(msg)


def _resolve_video_out_file(
    inp_path: Path,
    out_file: str | Path | None,
    filter_name: str,
    default_out_dir: Path,
    *,
    save2subfolder: bool,
) -> Path | None:
    """Определяет путь выходного видеофайла."""
    basename = f'{filter_name}.avi'
    if save2subfolder:
        out_path = Path(out_file) if out_file else default_out_dir
        out_dir = (
            out_path.parent if out_path.suffix.lower() in cv2_vid_exts else out_path
        )
        return out_dir / basename
    if inp_path.suffix.lower() not in cv2_vid_exts:
        return None
    if not out_file:
        return default_out_dir / basename
    out_path = Path(out_file)
    if out_path.is_dir():
        return out_path / basename
    return out_path.with_suffix('.avi')


def apply2video(  # noqa: C901, PLR0913
    im_filter: Callable[[np.ndarray], np.ndarray],
    inp_file: str | Path | None = None,
    out_file: str | Path | None = None,
    *,
    save2subfolder: bool = False,
    verbose: bool = True,
    rescale: bool | int | tuple[int, int] | list[int] = True,
    step: int = 1,
    skip_existed: bool = True,
) -> None:
    """Применяет функцию обработки изображений к видео."""
    default_inp_dir = Path('/home/user/work/shubin/Test/Video')
    default_out_dir = Path('/home/user/work/shubin/Results/Video')

    # Если это папка - обрабатываем все файлы в ней:
    inp_path = Path(inp_file) if inp_file else default_inp_dir
    if inp_path.is_dir():
        out_dir = Path(out_file) if out_file else default_out_dir
        for file in sorted(inp_path.iterdir()):
            apply2video(
                im_filter,
                file,
                out_dir / file.name,
                save2subfolder=save2subfolder,
                verbose=verbose,
                rescale=rescale,
                step=step,
                skip_existed=skip_existed,
            )
        return

    # Определяем выходной файл:
    out_path = _resolve_video_out_file(
        inp_path,
        out_file,
        _resolve_filter_name(im_filter),
        default_out_dir,
        save2subfolder=save2subfolder,
    )
    if out_path is None:
        return

    # Пропуск существующих файлов, если надо:
    if skip_existed and out_path.exists():
        return

    # Создаём необходимые вложенные папки:
    if not out_path.parent.exists():
        mkdirs(out_path.parent)

    # Файлы чтения и записи:
    cap = cv2.VideoCapture(os.fspath(inp_path))
    target_size = (
        _rescale_target_size(cap, rescale=rescale)
        if rescale
        else (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    )
    out = cv2.VideoWriter(
        os.fspath(out_path),
        cv2.VideoWriter_fourcc(*'MJPG'),  # type: ignore[attr-defined]
        cap.get(cv2.CAP_PROP_FPS) / step,
        target_size,
    )

    # Сброс состояния фильтра:
    if hasattr(im_filter, 'reset'):
        im_filter.reset()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // step
    for frame_ind in tqdm(
        range(total_frames),
        os.fspath(inp_path),
        disable=not verbose,
    ):
        # Пропуск ряда кадров:
        if step != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind * step)

        ret, frame = cap.read()
        if not ret:
            continue

        if rescale:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        frame = im_filter(bgr2rgb(frame))

        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)

        # Запись кадра:
        out.write(_to_3channels_bgr(frame))

    # Закрытие файлов:
    cap.release()
    out.release()

    # Сброс состояния фильтра:
    if hasattr(im_filter, 'reset'):
        im_filter.reset()


class VideoGenerator:
    """Читает кадры из последовательности видео- и/или фотофайлов.

    Может собирать в одну последовательность разные кадры из разных
    файлов. Работает как генератор.
    """

    @staticmethod
    def get_file_total_frames(
        file: str | Path | list[str] | tuple[str, ...],
    ) -> int:
        """Определяет число кадров в видео или наборе изображений."""
        # Если имеется список изображений:
        if isinstance(file, (list, tuple)):
            for file_ in file:
                # Все изображения должны иметь поддерживаемый формат:
                file_ext = Path(file_).suffix.lower()
                if file_ext not in cv2_img_exts:
                    msg = (
                        'Вложенный список файлов должен '
                        'содержать только изображения. '
                        'Получен файл неподдерживаемого '
                        f'расширения: {file_ext}!'
                    )
                    raise ValueError(msg)

            # Число кадров для изображений равно числу файлов:
            return len(file)

        # Определяем тип файла:
        file_ext = Path(file).suffix.lower()

        # Если файл является изображением, то в нём может быть всего 1 кадр:
        if file_ext in cv2_img_exts:
            return 1

        # Если это видео:
        if file_ext in cv2_vid_exts:
            vcap = cv2.VideoCapture(os.fspath(file))
            if not vcap.isOpened():
                msg = f'Ошибка открытия файла "{file}"!'
                raise RuntimeError(msg)
            total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vcap.release()
            return total_frames

        msg = (
            'Ожидалось получить путь к видео, фото или '
            f'списку/кортежу фото. Получен: {file}!'
        )
        raise ValueError(msg)

    def __init__(
        self,
        files: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        frame_ranges: list | tuple | None = None,
    ) -> None:
        """Сохраняет список файлов и соответствующих диапазонов кадров."""
        self.files: list = []
        self.frame_ranges: list = []
        self.im_read_buffers: list[ImReadBuffer] = []

        # Абстракция для удобного чтения очередного кадра:
        self.im_read_buffer = ImReadBuffer()

        # Наполняем списки переданными значениями:
        self.extend(files or [], frame_ranges or [])

    def extend(
        self,
        files: str | Path | list[str | Path] | tuple[str | Path, ...],
        frame_ranges: list | tuple | None = None,
    ) -> None:
        """Добавляет несколько файлов и их диапазоны в общий список."""
        # Если передан список/кортеж, то кортеж превращаем в список, а
        # список копируем:
        if isinstance(files, (list, tuple)):
            files = list(files)
        # Если передан не список/кортеж, то считаем, что это один файл:
        else:
            files = [files]
            frame_ranges = frame_ranges or [None]

        # Заполняем frame_ranges пустышками, если не задан:
        frame_ranges = frame_ranges or [None] * len(files)

        # frame_ranges и files должны иметь равную длину:
        if len(frame_ranges) != len(files):
            msg = 'Число файлов и диапазонов кадров должно совпадать.'
            raise ValueError(msg)

        # Поочерёдно вносим каждый файл с его диапазоном в общий список:
        for file, frame_range in zip(files, frame_ranges, strict=False):
            self.append(file, frame_range)

    def append(
        self,
        file: str | Path | list[str] | tuple[str, ...],
        frame_range: range | int | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Добавляет очередной файл и его диапазон в общий список.

        frame_ranges для изображений не используется. Для видео и списка
        кадров frame_ranges может быть задан несколькими способами:
            1) Объектом класса range. Тогда он сохраняется без изменений.
            2) Целое число. Оно принимается как аргумент объекта range.
                Если оно отрицательное, производится предварительный
                перерасчёт относительно длины последовательности.
            3) Список или кортеж. Его элементы принимаются как аргументы
                объекта range, если их 2 или 3, и второй из них является
                отрицательным (указывает на номер с конца, как в случае
                с п.2). Если первый элемент неотрицательный, или элементов
                больше 3х, то сам список/кортеж принимается как
                перечисление номеров кадров (порядок может быть любым).
            4) None или пустой список/кортеж. В этом случае берутся все
                кадры последовательности.
        """
        # Если передан путь к папке, то подменяем его списком вложенных в неё
        # изображений:
        if isinstance(file, (str, Path)) and Path(file).is_dir():
            file = sorted(
                str(path)
                for path in Path(file).iterdir()
                if path.suffix.lower() in cv2_img_exts
            )

        # Вносим имя очередного файла в список:
        self.files.append(file)

        file_ext = ''
        # Если в качестве файла передан список изображений:
        if isinstance(file, (list, tuple)):
            # Проверяем корректность списка:
            self.get_file_total_frames(file)
            is_im_seq = True  # Флаг последовательности изображений

        # Если указан всего один файл:
        else:
            # Определяем тип файла:
            file_ext = Path(file).suffix.lower()

            # Если файл является изображением, то в нём лишь один кадр:
            if file_ext in cv2_img_exts:
                self.frame_ranges.append(range(1))
                return

            is_im_seq = False  # Флаг последовательности изображений

        # Если файл является набором изображений или видеофайлом:
        if is_im_seq or (file_ext in cv2_vid_exts):
            # Если frame_range действительно является диапазоном, то вносим
            # без изменений:
            if isinstance(frame_range, range):
                self.frame_ranges.append(frame_range)

            # Если frame_range не указан, берём все кадры исходной
            # последовательности:
            elif frame_range is None:
                total_frames = self.get_file_total_frames(file)
                self.frame_ranges.append(range(total_frames))

            # Если frame_range является списокм или кортежем:
            elif isinstance(frame_range, (list, tuple)):
                # Если frame_range состоит из 2х или 3х элементов, и второй
                # является отрицательным:
                if len(frame_range) in {2, 3} and frame_range[1] < 0:
                    # Собираем диапазон с отсчётом от конца файла:
                    total_frames = self.get_file_total_frames(file)
                    frame_range = range(
                        frame_range[0],
                        total_frames + frame_range[1],
                        *frame_range[2:],
                    )
                    self.frame_ranges.append(frame_range)

                # Если frame_range не состоит из 2х или 3х элементов, или
                # второй из них не является отрицательным:
                else:
                    # Фоспринимаем список/кортеж как набор номеров кадров:
                    self.frame_ranges.append(frame_range)
            else:
                raise ValueError(
                    'Неверное значение параметра ' + f'"frame_range": {frame_range}!',
                )
        else:
            msg = f'Неподдерживаемый тип файла: {file_ext}!'
            raise TypeError(msg)
        return

    def __len__(self) -> int:
        """Возвращает общее число кадров последовательности."""
        return sum(map(len, self.frame_ranges))

    def __iter__(self) -> Self:
        """Начинает итерацию по кадрам последовательности."""
        self.files_ind = 0
        self.frames_iter = iter(self.frame_ranges[self.files_ind])
        return self

    def __next__(self) -> np.ndarray:
        """Возвращает очередной кадр последовательности."""
        # Определяем номер файла и номер следующего кадра в нём:
        while True:
            try:
                frame = next(self.frames_iter)
                break
            except StopIteration:
                self.files_ind += 1

                # Выходим, если дошли до конца:
                if self.files_ind == len(self.files):
                    self.im_read_buffer.close()
                    raise StopIteration from None

                self.frames_iter = iter(self.frame_ranges[self.files_ind])

        # Читаем и возвращаем следующий кадр:
        cur_file = self.files[self.files_ind]
        return self.im_read_buffer(cur_file, frame)

    def __getitem__(self, index: int) -> np.ndarray:
        """Возвращает кадр с произвольным номером."""
        # Создаём список буферов кадров, если это первый вызов функции:
        if not self.im_read_buffers:
            self.im_read_buffers = [ImReadBuffer() for _ in self.files]

        # Определяем номер файла и номер его кадра:
        frame_range_ind = int(index)
        for file_ind, frame_range in enumerate(self.frame_ranges):  # noqa: B007
            if frame_range_ind < len(frame_range):
                break
            frame_range_ind -= len(frame_range)
        else:
            msg = f'{index} >= {len(self)}'
            raise IndexError(msg)

        # Читаем сам кадр:
        return self.im_read_buffers[file_ind](
            self.files[file_ind],
            frame_range_ind,
        )

    def __del__(self) -> None:
        """Закрывает все открытые буферы чтения."""
        self.im_read_buffer.close()
        for im_read_buffer in self.im_read_buffers:
            im_read_buffer.close()


def _resolve_image_out_file(
    out_file: str | Path | None,
    filter_name: str,
    default_out_dir: Path,
    *,
    save2subfolder: bool,
) -> Path:
    """Определяет путь выходного изображения."""
    if save2subfolder:
        basename = f'{filter_name}.png'
        out_path = Path(out_file) if out_file else default_out_dir
        if out_path.is_dir():
            return out_path / basename
        return out_path.with_suffix('.png')
    if not out_file:
        return default_out_dir / f'{filter_name}.png'
    out_path = Path(out_file)
    if out_path.is_dir():
        return out_path / f'{filter_name}.png'
    return out_path.with_suffix('.png')


def apply2image(  # noqa: PLR0913
    im_filter: Callable[[np.ndarray], np.ndarray],
    inp_file: str | Path | None = None,
    out_file: str | Path | None = None,
    *,
    save2subfolder: bool = False,
    rescale: bool | int | tuple[int, int] | list[int] = True,
    skip_existed: bool = True,
) -> None:
    """Применяет функцию обработки изображений к изображениям."""
    default_inp_dir = Path('/home/user/work/shubin/Test/Image')
    default_out_dir = Path('/home/user/work/shubin/Results/Image')

    # Если это папка - обрабатываем все файлы в ней:
    inp_path = Path(inp_file) if inp_file else default_inp_dir
    if inp_path.is_dir():
        out_dir = Path(out_file) if out_file else default_out_dir
        for file in sorted(inp_path.iterdir()):
            apply2image(
                im_filter,
                file,
                out_dir / file.name,
                save2subfolder=save2subfolder,
                rescale=rescale,
                skip_existed=skip_existed,
            )
        return

    # Определяем выходной файл:
    out_path = _resolve_image_out_file(
        out_file,
        _resolve_filter_name(im_filter),
        default_out_dir,
        save2subfolder=save2subfolder,
    )

    # Пропуск существующих файлов, если надо:
    if skip_existed and out_path.exists():
        return

    inp = imread(os.fspath(inp_path))

    # Приводим входное изображение к трёхканальному RGB:
    inp = _to_3channels_rgb(inp)

    # Применяем фильтр, понижая разрешение при нехватке памяти:
    out = _filter_with_oom_fallback(im_filter, inp, rescale=rescale)

    if out.dtype != np.uint8:
        out = (out * 255).astype(np.uint8)

    # Приводим результат к трёхканальному BGR для OpenCV:
    bgr = _to_3channels_bgr(out)

    # Создаём необходимые вложенные папки:
    if not out_path.parent.exists():
        mkdirs(out_path.parent)

    cv2.imwrite(os.fspath(out_path), bgr)


def convert_videos(
    inp_path: str | Path,
    out_path: str | Path,
    *,
    skip_existed: bool = True,
) -> None:
    """Пересжимает видеофайлы из одной директории в другую.

    Используется для подготовки видеорезультатов к демонстрации.
    """
    inp_root = Path(inp_path)
    out_root = Path(out_path)
    file_list = get_file_list(os.fspath(inp_root))
    for inp_file in tqdm(file_list):
        source = Path(inp_file)
        if source.suffix.lower() not in cv2_vid_exts:
            continue

        rel_path = Path(os.path.relpath(source, inp_root)).with_suffix('.mp4')
        out_file = out_root / rel_path
        if skip_existed and out_file.exists():
            continue
        if not out_file.parent.exists():
            out_file.parent.mkdir(parents=True)

        recomp2mp4(source, out_file)


def _resolve_cvals(
    cval: float | list[float] | tuple[float, ...],
    channels: int,
) -> list[float]:
    """Приводит cval к списку значений по числу каналов."""
    if isinstance(cval, (list, tuple)):
        if len(cval) != channels:
            msg = (
                'Параметр "cval" должен быть скаляром, либо '
                'списком/кортежем длиной, соответствующей числу '
                f'каналов изображения. Получено: {cval}!'
            )
            raise ValueError(msg)
        return list(cval)
    return [cval] * channels


class OptFlow:
    """Обвязка вокруг cv2.calcOpticalFlowFarneback.

    Позволяет выполнять различные операции на базе оптического потока.
    """

    # Число измерений и каналов изображения:
    COLOR_NDIM = 3
    GRAY_NDIM = 2
    COLOR_CHANNELS = 3

    def __init__(  # noqa: PLR0913
        self,
        *,
        pyr_scale: float = 0.5,
        levels: int = 1,
        winsize: int = 40,
        poly_n: int = 5,
        poly_sigma: float = 1.1,
        iterations: int = 10,
        flags: int = cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ) -> None:
        """Сохраняет параметры оптического потока."""
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.iterations = iterations
        self.flags = flags

    def __call__(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        flow: np.ndarray | None = None,
        flags: int | None = None,
    ) -> np.ndarray:
        """Вычисляет оптический поток для двух изображений."""
        # Переводим цветные изображения в оттенки серого, если надо:
        if img1.ndim > self.GRAY_NDIM and img1.shape[2] == self.COLOR_CHANNELS:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim > self.GRAY_NDIM and img2.shape[2] == self.COLOR_CHANNELS:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Берём флаги из входного параметра, если он задан:
        flags = self.flags if flags is None else flags

        # Если на вход передана начальная аппроксимация потока, то
        # указываем во флагах необходимость её использовать:
        if flow is not None:
            flags = flags | cv2.OPTFLOW_USE_INITIAL_FLOW

            # Чтобы не писать результат во входную переменную:
            flow = flow.copy()

        # Вычисляем сам оптический поток:
        return cv2.calcOpticalFlowFarneback(
            img1,
            img2,
            flow,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            iterations=self.iterations,
            flags=flags,
        )

    @staticmethod
    def create_meshgrid(shape: tuple[int, int]) -> np.ndarray:
        """Строит координатную сетку для деформации изображения."""
        y: np.ndarray = np.arange(shape[0], dtype=float)
        x: np.ndarray = np.arange(shape[1], dtype=float)
        xv: np.ndarray
        yv: np.ndarray
        xv, yv = np.meshgrid(x, y)
        return np.dstack([xv, yv])

    @classmethod
    def apply_flow2meshgrid(
        cls,
        flow: np.ndarray,
        meshgrid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Деформирует или создаёт координатную сетку по потоку."""
        # Копируем или создаём исходную сетку:
        if meshgrid is None:
            meshgrid = cls.create_meshgrid(flow.shape)
        if flow.shape != meshgrid.shape:
            msg = (
                f'Формы потока и сетки не совпадают: {flow.shape} != {meshgrid.shape}.'
            )
            raise ValueError(msg)

        # Деформируем сетку:
        return meshgrid - flow

    @staticmethod
    def apply_meshgrid2img(
        meshgrid: np.ndarray,
        img: np.ndarray,
        *,
        nearest_interpolation: bool = False,
        mode: _MAP_MODE = 'mirror',
        cval: float | list[float] | tuple[float, ...] = 0.0,
    ) -> np.ndarray:
        """Интерполирует каналы изображения по координатной сетке."""
        # Готовим сетку для использования:
        if nearest_interpolation:  # Округляем координаты, если требуется
            meshgrid = np.round(meshgrid)
        coords = [meshgrid[..., 1], meshgrid[..., 0]]

        # Если изображение одноканально:
        if img.ndim == OptFlow.GRAY_NDIM:
            return scipy.ndimage.map_coordinates(img, coords, mode=mode, cval=cval)

        # Если изображение многоканально:
        if img.ndim == OptFlow.COLOR_NDIM:
            cvals = _resolve_cvals(cval, img.shape[2])
            return np.dstack(
                [
                    scipy.ndimage.map_coordinates(
                        img[..., ch],
                        coords,
                        mode=mode,
                        cval=channel_cval,
                    )
                    for ch, channel_cval in enumerate(cvals)
                ],
            )

        msg = 'Изображение должно иметь 2 или 3 измерения!'
        raise ValueError(msg)

    @classmethod
    def apply_flow2img(
        cls,
        flow: np.ndarray,
        img: np.ndarray,
        *,
        nearest_interpolation: bool = False,
        mode: _MAP_MODE = 'mirror',
        cval: float | list[float] | tuple[float, ...] = 0.0,
    ) -> np.ndarray:
        """Восстанавливает второе изображение по первому и потоку."""
        # Строим деформированную координатную сетку:
        meshgrid = cls.apply_flow2meshgrid(flow)

        # Деформируем изображение по координатной сетке:
        return cls.apply_meshgrid2img(
            meshgrid,
            img,
            nearest_interpolation=nearest_interpolation,
            mode=mode,
            cval=cval,
        )

    def seq_flows(
        self,
        imgs: list[np.ndarray],
        *,
        cum_sum: bool = True,
        **mpmap_kwargs: object,
    ) -> list[np.ndarray]:
        """Вычисляет потоки между соседними кадрами последовательности."""
        # Расчёт потока между каждой парой соседних кадров:
        flows = mpmap(self.__call__, imgs[:-1], imgs[1:], **mpmap_kwargs)

        # Если поток отстраивается от первого изображения:
        if cum_sum:
            # Инициируем нулями поток для первого кадра с самим собой:
            flow = np.zeros_like(flows[0])
            cum_flows = [flow]

            # Накапливаем сдвиги для следующих кадров:
            for dflow in flows:
                flow = flow + dflow
                cum_flows.append(flow)

            # Заменяем исходные потоки потоками с накоплением:
            flows = cum_flows

        return flows

    def apply_flows2img(
        self,
        flows: list[np.ndarray],
        first_img: np.ndarray,
        desc: str | None = None,
    ) -> list[np.ndarray]:
        """Восстанавливает последовательность кадров по первому и потокам."""
        imgs = [first_img.copy()]
        for flow in tqdm(flows, desc=desc, disable=not desc):
            imgs.append(self.apply_flow2img(flow, imgs[-1]))
        return imgs
