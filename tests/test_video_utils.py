"""test_video_utils.py - Тесты для модуля video_utils.py."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

import video_utils
from video_utils import (
    AddCaption,
    AsType,
    CompareFiltersWithTarget,
    CompareTwoFilters,
    Concat,
    DrawSemanticSegments,
    KerasModel,
    Mix,
    OptFlow,
    Pipeline,
    PutTime,
    Res,
    Resize,
    ResizeAndRestore,
    StoreLastFrames,
    StreamRandomCrop,
    StreamRandomFlip,
    VideoGenerator,
    ViRead,
    ViSave,
    _filter_with_oom_fallback,
    _named_filter,
    _rescale_target_size,
    _resolve_cvals,
    _resolve_filter_name,
    _resolve_image_out_file,
    _resolve_video_out_file,
    _threshold_filter,
    _to_3channels_bgr,
    _to_3channels_rgb,
    apply2image,
    apply2video,
    bgr2gray,
    bgr2rgb,
    bgr2yuv,
    clean_remux,
    color_canny,
    convert_videos,
    double2im,
    gray2bgr,
    gray2rgb,
    im2double,
    recomp2mp4,
    rgb2bgr,
    rgb2gray,
    rgb2yuv,
    skip_none,
    yuv2bgr,
    yuv2gray,
    yuv2rgb,
)

# Размер кадра синтетических видео (ширина, высота) и частота кадров:
VIDEO_SIZE = (32, 24)
VIDEO_FPS = 10.0
VIDEO_FRAMES = 10

# Размер кадра синтетических картинок (высота, ширина):
IMG_SIZE = (24, 32)

# Реальная битая запись устройства (для интеграционной проверки, если есть):
PREVIEW_AVI = (
    Path(__file__).resolve().parents[2]
    / 'VideoDataOps/services/videocutter/workspaces/6/preview.avi'
)


def _frame(index: int, size: tuple[int, int] = VIDEO_SIZE) -> np.ndarray:
    """Возвращает однотонный BGR-кадр с уникальной яркостью по индексу."""
    width, height = size
    level = (index * 20) % 256
    return np.full((height, width, 3), level, dtype=np.uint8)


def _make_video(
    path: str | Path,
    count: int = VIDEO_FRAMES,
    size: tuple[int, int] = VIDEO_SIZE,
    fps: float = VIDEO_FPS,
) -> str:
    """Записывает MJPG-avi с заданным числом однотонных кадров."""
    path = os.fspath(path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    assert writer.isOpened()
    for index in range(count):
        writer.write(_frame(index, size))
    writer.release()
    return path


def _read_frames(path: str | Path) -> list[np.ndarray]:
    """Считывает все кадры видеофайла и возвращает их списком."""
    capture = cv2.VideoCapture(os.fspath(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class TestViRead(unittest.TestCase):
    """Тесты чтения видео через ViRead."""

    def test_len_matches_header(self) -> None:
        """Длина ViRead совпадает с числом кадров в контейнере."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with ViRead(path) as reader:
                assert len(reader) == VIDEO_FRAMES

    def test_iterates_all_frames(self) -> None:
        """Итерация ViRead отдаёт все кадры и завершается."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            assert len(list(ViRead(path))) == VIDEO_FRAMES

    def test_close_returns_none_at_end(self) -> None:
        """По умолчанию на конце потока возвращается None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reader = ViRead(path)
            for _ in range(VIDEO_FRAMES):
                assert reader() is not None
            assert reader() is None
            reader.close()

    def test_repeat_last_frame(self) -> None:
        """Режим repeat_last_frame повторяет последний кадр."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reader = ViRead(path, on_end='repeat_last_frame')
            last = None
            for _ in range(VIDEO_FRAMES):
                last = reader()
            assert last is not None
            assert np.array_equal(reader(), last)
            reader.close()

    def test_reset_mode_restarts(self) -> None:
        """Режим reset перезапускает поток с начала."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reader = ViRead(path, on_end='reset')
            first = reader()
            for _ in range(VIDEO_FRAMES - 1):
                reader()
            assert np.array_equal(reader(), first)
            reader.close()

    def test_colorspace_gray(self) -> None:
        """Цветовая схема gray даёт двумерный кадр."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            frame = ViRead(path, colorspace='gray')()
            assert frame is not None
            assert frame.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0])

    def test_colorspace_bgr_matches_source(self) -> None:
        """Цветовая схема bgr не меняет кадр источника."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reference = _read_frames(path)[0]
            frame = ViRead(path, colorspace='bgr')()
            assert frame is not None
            assert np.array_equal(frame, reference)

    def test_start_frame_positive(self) -> None:
        """Положительный start_frame начинает чтение с нужного кадра."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reference = _read_frames(path)[4]
            frame = ViRead(path, start_frame=4, colorspace='bgr')()
            assert frame is not None
            assert np.array_equal(frame, reference)

    def test_start_frame_negative(self) -> None:
        """Отрицательный start_frame отсчитывается от конца."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            reference = _read_frames(path)[-1]
            frame = ViRead(path, start_frame=-1, colorspace='bgr')()
            assert frame is not None
            assert np.array_equal(frame, reference)

    def test_start_frame_range(self) -> None:
        """Диапазон start_frame выбирает случайный кадр внутри него."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            assert ViRead(path, start_frame=[2, 5])() is not None

    def test_invalid_start_frame_raises(self) -> None:
        """start_frame за пределами видео вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with pytest.raises(ValueError, match='всего'):
                ViRead(path, start_frame=100)

    def test_invalid_colorspace_raises(self) -> None:
        """Неизвестная цветовая схема вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with pytest.raises(ValueError, match='цветовых схем'):
                ViRead(path, colorspace='cmyk')


class TestViSave(unittest.TestCase):
    """Тесты записи видео через ViSave."""

    def test_writes_frames(self) -> None:
        """ViSave записывает все переданные кадры."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'out.avi'
            with ViSave(str(path), colorspace='bgr') as writer:
                for index in range(VIDEO_FRAMES):
                    writer(_frame(index))
            assert path.is_file()
            assert len(_read_frames(path)) == VIDEO_FRAMES

    def test_invalid_colorspace_raises(self) -> None:
        """Неизвестная цветовая схема вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'o.avi'
            with pytest.raises(ValueError, match='цветовых схем'):
                ViSave(str(path), colorspace='cmyk')


class TestPipeline(unittest.TestCase):
    """Тесты конвейера фильтров Pipeline."""

    def test_concat_named_filters(self) -> None:
        """Строковые синонимы фильтров попадают в Pipeline с именами."""
        pipe = Pipeline(['bgr2rgb', 'rgb2gray'])
        assert pipe.names == ['bgr2rgb', 'rgb2gray']

    def test_concat_nested_list(self) -> None:
        """Вложенный список фильтров разворачивается."""
        pipe = Pipeline([['bgr2rgb'], 'rgb2gray'])
        assert len(pipe.functions) == 2

    def test_call_applies_filter(self) -> None:
        """Pipeline применяет фильтры к переданному кадру."""
        result = Pipeline(['rgb2gray'])(_frame(3))
        assert result.shape == (VIDEO_SIZE[1], VIDEO_SIZE[0])

    def test_convert_avi(self) -> None:
        """Метод convert без пересжатия пишет все кадры в avi."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = Path(temp_dir) / 'dst.avi'
            Pipeline(['bgr2rgb']).convert(
                source,
                str(target),
                desc=None,
                recompress=False,
            )
            assert target.is_file()
            assert len(_read_frames(target)) == VIDEO_FRAMES

    def test_convert_step(self) -> None:
        """Метод convert с шагом 2 пишет каждый второй кадр."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = Path(temp_dir) / 'dst.avi'
            Pipeline(['bgr2rgb']).convert(
                source,
                str(target),
                step=2,
                desc=None,
                recompress=False,
            )
            assert len(_read_frames(target)) == VIDEO_FRAMES // 2

    def test_convert_recompress_mp4(self) -> None:
        """Метод convert с пересжатием создаёт mp4-файл."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = Path(temp_dir) / 'dst.mp4'
            Pipeline(['bgr2rgb']).convert(source, str(target), desc=None)
            assert target.is_file()
            assert target.stat().st_size > 0


class TestVideoGenerator(unittest.TestCase):
    """Тесты генератора кадров VideoGenerator."""

    def test_total_frames_video(self) -> None:
        """Для видео число кадров берётся из контейнера."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            assert VideoGenerator.get_file_total_frames(path) == VIDEO_FRAMES

    def test_total_frames_image(self) -> None:
        """Для одиночного изображения число кадров равно единице."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'i.png'
            cv2.imwrite(str(path), _frame(0))
            assert VideoGenerator.get_file_total_frames(str(path)) == 1

    def test_total_frames_image_list(self) -> None:
        """Для списка изображений число кадров равно их числу."""
        with tempfile.TemporaryDirectory() as temp_dir:
            paths: list[str] = []
            for index in range(2):
                path = Path(temp_dir) / f'i{index}.png'
                cv2.imwrite(str(path), _frame(index))
                paths.append(str(path))
            assert VideoGenerator.get_file_total_frames(paths) == 2

    def test_total_frames_invalid(self) -> None:
        """Неподдерживаемое расширение вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'i.txt'
            with pytest.raises(ValueError, match='Ожидалось получить путь'):
                VideoGenerator.get_file_total_frames(str(path))

    def test_len_iter_getitem(self) -> None:
        """len/итерация/доступ по индексу согласованы по числу кадров."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            generator = VideoGenerator(path)
            assert len(generator) == VIDEO_FRAMES
            assert len(list(generator)) == VIDEO_FRAMES
            assert generator[3].shape == (VIDEO_SIZE[1], VIDEO_SIZE[0], 3)

    def test_append_range(self) -> None:
        """Метод append с диапазоном кадров ограничивает длину."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            generator = VideoGenerator()
            generator.append(path, range(2, 5))
            assert len(generator) == 3
            assert len(list(generator)) == 3

    def test_append_negative_range(self) -> None:
        """Метод append с отрицательным концом диапазона считает от конца."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            generator = VideoGenerator()
            generator.append(path, [2, -1])
            assert len(generator) == VIDEO_FRAMES - 1 - 2


class TestRecomp2Mp4(unittest.TestCase):
    """Тесты пересжатия видео в mp4."""

    def test_recompress_keeps_source(self) -> None:
        """recomp2mp4 не удаляет исходник при rm_soruce=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = Path(temp_dir) / 'dst.mp4'
            recomp2mp4(source, str(target), rm_soruce=False)
            assert target.is_file()
            assert Path(source).is_file()
            assert target.stat().st_size > 0

    def test_recompress_removes_source(self) -> None:
        """recomp2mp4 удаляет исходник при rm_soruce=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = Path(temp_dir) / 'dst.mp4'
            recomp2mp4(source, str(target), rm_soruce=True)
            assert target.is_file()
            assert not Path(source).exists()


class TestCleanRemux(unittest.TestCase):
    """Тесты перепаковки видео без пересжатия (clean_remux)."""

    def test_clean_video_kept_fully(self) -> None:
        """На чистом видео не выбрасывается ни один кадр."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'src.avi'
            _make_video(source)
            target = Path(temp_dir) / 'dst.mkv'
            kept, dropped = clean_remux(source, target)
            assert (kept, dropped) == (VIDEO_FRAMES, 0)
            assert len(_read_frames(target)) == VIDEO_FRAMES

    def test_frames_are_not_reencoded(self) -> None:
        """Содержимое кадров не меняется при перепаковке."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'src.avi'
            _make_video(source)
            target = Path(temp_dir) / 'dst.mkv'
            clean_remux(source, target)
            assert np.array_equal(_read_frames(source)[0], _read_frames(target)[0])

    def test_source_kept_by_default(self) -> None:
        """По умолчанию исходный файл сохраняется."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'src.avi'
            _make_video(source)
            target = Path(temp_dir) / 'dst.mkv'
            clean_remux(source, target)
            assert source.is_file()
            assert target.is_file()

    def test_source_removed(self) -> None:
        """При rm_source=True исходник удаляется после записи."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'src.avi'
            _make_video(source)
            target = Path(temp_dir) / 'dst.mkv'
            clean_remux(source, target, rm_source=True)
            assert not source.exists()
            assert target.is_file()

    def test_accepts_path_objects(self) -> None:
        """Функция принимает пути и в виде str, и в виде Path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / 'src.avi'
            _make_video(source)
            target = Path(temp_dir) / 'dst.mkv'
            kept, dropped = clean_remux(source, target)
            assert (kept, dropped) == (VIDEO_FRAMES, 0)
            assert target.is_file()

    def test_accepts_str_paths(self) -> None:
        """Функция одинаково работает со строками путей."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = str(Path(temp_dir) / 'dst.mkv')
            kept, dropped = clean_remux(source, target)
            assert (kept, dropped) == (VIDEO_FRAMES, 0)
            assert Path(target).is_file()

    def test_fast_check_counts(self) -> None:
        """full_check=False пропускает проверку, но счётчики верны."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source = _make_video(Path(temp_dir) / 'src.avi')
            target = str(Path(temp_dir) / 'dst.mkv')
            kept, dropped = clean_remux(source, target, full_check=False)
            assert (kept, dropped) == (VIDEO_FRAMES, 0)

    def test_missing_source_raises(self) -> None:
        """Отсутствие исходного файла приводит к FileNotFoundError."""
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(FileNotFoundError, match='Нет файла'),
        ):
            clean_remux(Path(temp_dir) / 'no.avi', Path(temp_dir) / 'out.mkv')


class TestCleanRemuxCorrupt(unittest.TestCase):
    """Интеграционная проверка clean_remux на реальной битой записи."""

    @unittest.skipUnless(PREVIEW_AVI.exists(), 'нет битого preview.avi')
    def test_drops_only_broken_frames(self) -> None:
        """Битые кадры выбрасываются, остальные сохраняются без пересжатия."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'clean.mkv'
            kept, dropped = clean_remux(PREVIEW_AVI, target)
            assert kept == 770
            assert dropped == 36
            assert target.is_file()
            assert target.stat().st_size > 0


# ------------- Фильтры, аугментации, поток и работа с файлами -------------


def _rgb_frame(index: int = 0, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Возвращает RGB-кадр с уникальными значениями каналов по индексу."""
    height, width = size
    frame: np.ndarray = np.zeros((height, width, 3), np.uint8)
    frame[..., 0] = (index * 20) % 256
    frame[..., 1] = (index * 20 + 40) % 256
    frame[..., 2] = (index * 20 + 80) % 256
    return frame


def _gray_frame(index: int = 0, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Возвращает двумерный кадр заданной яркости."""
    return np.full(size, (index * 20) % 256, np.uint8)


def _to_float(frame: np.ndarray) -> np.ndarray:
    """Переводит кадр в вещественные значения диапазона [0, 1]."""
    return frame.astype(np.float64) / 255


def _make_image(path: str | Path) -> str:
    """Записывает RGB-картинку стандартного размера."""
    path = os.fspath(path)
    assert cv2.imwrite(path, bgr2rgb(_rgb_frame()))
    return path


# ------------------------------------------- Конвертация цветов и пороги ---


class TestColorHelpers:
    """Проверки конвертаций цветов и бинаризации."""

    def test_rgb2bgr_is_involutive(self) -> None:
        """Перевод RGB->BGR дважды возвращает исходный кадр."""
        img = _rgb_frame()
        assert np.array_equal(rgb2bgr(bgr2rgb(img)), img)

    def test_gray2rgb_repeats_channels(self) -> None:
        """Серый кадр разворачивается в три одинаковых канала."""
        out = gray2rgb(_gray_frame(3))
        assert out.shape == (*IMG_SIZE, 3)
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_rgb2yuv_roundtrip(self) -> None:
        """Переход RGB->YUV->RGB возвращает исходный кадр."""
        img = _rgb_frame()
        assert np.allclose(yuv2rgb(rgb2yuv(img)), img, atol=1)

    def test_bgr2yuv_roundtrip(self) -> None:
        """Переход BGR->YUV->BGR возвращает исходный кадр."""
        img = bgr2rgb(_rgb_frame())
        assert np.allclose(yuv2bgr(bgr2yuv(img)), img, atol=1)

    def test_yuv2gray_takes_luma(self) -> None:
        """Из YUV берётся первый канал яркости."""
        assert np.array_equal(yuv2gray(rgb2yuv(_rgb_frame())), rgb2gray(_rgb_frame()))

    def test_bgr2gray_matches_rgb2gray(self) -> None:
        """Яркость не зависит от порядка цветовых каналов."""
        assert np.array_equal(bgr2gray(bgr2rgb(_rgb_frame())), rgb2gray(_rgb_frame()))

    def test_gray2bgr_repeats_channels(self) -> None:
        """Серый кадр разворачивается в BGR."""
        out = gray2bgr(_gray_frame(3))
        assert out.shape == (*IMG_SIZE, 3)
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_im2double_roundtrip(self) -> None:
        """Обратное преобразование double2im(im2double) не меняет uint8."""
        img = _rgb_frame()
        assert np.array_equal(double2im(im2double(img)), img)

    def test_color_canny_returns_three_channels(self) -> None:
        """Детектор Кэнни отдаёт трёхканальный инвертированный результат."""
        out = color_canny(_rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)
        assert np.all(out <= 1)

    def test_threshold_filter(self) -> None:
        """Пороговый фильтр оставляет только значения выше порога."""
        img = _gray_frame()
        out = _threshold_filter(img, 128)
        assert set(np.unique(out)).issubset({0, 255})

    def test_named_filter_known_synonym(self) -> None:
        """Строковый синоним разрешается в функцию-фильтр."""
        out = _named_filter('rgb2gray')(_rgb_frame())
        assert out.ndim == 2

    def test_named_filter_unknown_raises(self) -> None:
        """Неизвестный синоним вызывает ValueError."""
        with pytest.raises(ValueError, match='не входит в список синонимов'):
            _named_filter('no_such_filter')

    def test_skip_none_passes_none_through(self) -> None:
        """Декоратор skip_none пропускает None и применяет функцию иначе."""
        decorated = skip_none(lambda img: img + 1)
        assert decorated(None) is None
        assert decorated(np.zeros((2, 2), np.uint8))[0, 0] == 1


# ------------------------------------------------------- Сборка Pipeline ---


class TestPipelineConcat:
    """Проверки разворачивания списка фильтров в конвейер."""

    def test_int_threshold_creates_filter(self) -> None:
        """Целое число в списке превращается в пороговый фильтр."""
        functions, names = Pipeline.concat([128])
        assert callable(functions[0])
        assert names == ['threshold_128']

    def test_nested_tuple_is_flattened(self) -> None:
        """Вложенные кортежи разворачиваются в плоский список."""
        _, names = Pipeline.concat([(rgb2gray, 'rgb2bgr'), (gray2rgb,)])
        assert names == ['NonameFunction', 'rgb2bgr', 'NonameFunction']

    def test_callable_name_taken_from_attribute(self) -> None:
        """Имя функции берётся из атрибута name, если он есть."""
        _, names = Pipeline.concat([PutTime()])
        assert names == ['NonameFunction']

    def test_unsupported_element_raises(self) -> None:
        """Неподходящий элемент списка вызывает TypeError."""
        with pytest.raises(TypeError, match='не является подходящим элементом'):
            Pipeline.concat([object()])

    def test_pipeline_reset_calls_filters(self) -> None:
        """Сброс конвейера долетает до сбрасываемых фильтров."""
        calls = []

        class Resettable:
            name = 'Resettable'

            def __call__(self, img: np.ndarray) -> np.ndarray:
                return img

            def reset(self) -> None:
                calls.append(1)

        Pipeline([Resettable()]).reset()
        assert calls == [1]

    def test_pipeline_applies_chain_in_order(self) -> None:
        """Фильтры конвейера применяются в порядке перечисления."""
        out = Pipeline([rgb2gray, gray2rgb])(_rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)


# --------------------------------------------------------------- Resize ---


class TestResize:
    """Проверки изменения размера кадра."""

    def test_scalar_size_is_squared(self) -> None:
        """Скалярный размер дублируется по обеим осям."""
        assert Resize(64).im_size == (64, 64)

    def test_tuple_size_kept(self) -> None:
        """Кортеж размера сохраняется как есть."""
        assert Resize((16, 32)).im_size == (16, 32)

    def test_call_resizes(self) -> None:
        """Вызов меняет размер кадра до заданного."""
        out = Resize((12, 16))(_rgb_frame())
        assert out.shape[:2] == (12, 16)

    def test_adopt_axis_size_relative(self) -> None:
        """Относительный размер пересчитывается от размера источника."""
        assert Resize.adopt_axis_size(0.5, 100) == 50

    def test_adopt_axis_size_absolute(self) -> None:
        """Целый размер возвращается без изменений."""
        assert Resize.adopt_axis_size(37, 100) == 37

    def test_adopt_axis_size_invalid_raises(self) -> None:
        """Некорректный размер оси вызывает ValueError."""
        with pytest.raises(ValueError, match='должен быть задан целыми'):
            Resize.adopt_axis_size('bad', 100)

    def test_resize_and_restores_size(self) -> None:
        """ResizeAndRestore возвращает кадр к исходному размеру."""
        filter_ = ResizeAndRestore((8, 12), lambda img: img)
        out = filter_(_rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)

    def test_resize_and_restore_resets_processor(self) -> None:
        """Сброс ResizeAndRestore долетает до процессора."""
        calls = []

        class Processor:
            def __call__(self, img: np.ndarray) -> np.ndarray:
                return img

            def reset(self) -> None:
                calls.append(1)

        ResizeAndRestore((8, 12), Processor()).reset()
        assert calls == [1]


# ------------------------------------------------------- Наложение текста ---


class TestPutTime:
    """Проверки нанесения времени и номера кадра."""

    @pytest.mark.parametrize(
        ('seconds', 'expected'),
        [
            (0.0, '00.000'),
            (1.5, '01.500'),
            (61.25, '01:01.250'),
            (3661.5, '01:01:01.500'),
            (90061.0, '1 01:01:01.000'),
        ],
    )
    def test_seconds2sexagesimal(self, seconds: float, expected: str) -> None:
        """Секунды разворачиваются в человеческий формат времени."""
        assert PutTime.seconds2sexagesimal(seconds) == expected

    def test_frame_counter_increments(self) -> None:
        """Счётчик кадров растёт на каждом вызове."""
        put_time = PutTime()
        put_time(_rgb_frame())
        put_time(_rgb_frame())
        assert put_time.frame == 2

    def test_reset_zeroes_counter(self) -> None:
        """Сброс обнуляет счётчик кадров."""
        put_time = PutTime()
        put_time(_rgb_frame())
        put_time.reset()
        assert put_time.frame == 0

    def test_with_fps_draws_time(self) -> None:
        """При заданном fps наносится текст времени."""
        out = PutTime(fps=VIDEO_FPS)(_rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)


class TestAddCaption:
    """Проверки нанесения подписи."""

    def test_text_is_drawn(self) -> None:
        """Подпись наносится поверх кадра."""
        caption = AddCaption('Привет')
        first = caption(_rgb_frame())
        assert first.shape == (*IMG_SIZE, 3)
        assert not np.array_equal(first, _rgb_frame())

    def test_mask_is_cached(self) -> None:
        """После первого вызова маска переиспользуется."""
        caption = AddCaption('Привет')
        caption(_rgb_frame())
        mask = caption.mask
        assert mask is not None
        caption(_rgb_frame())
        assert caption.mask is mask

    def test_reset_drops_mask(self) -> None:
        """Сброс удаляет закэшированную маску."""
        caption = AddCaption('Привет')
        caption(_rgb_frame())
        caption.reset()
        assert caption.mask is None

    def test_mask_rebuilt_on_new_size(self) -> None:
        """При смене размера кадра маска строится заново."""
        caption = AddCaption('Привет')
        caption(_rgb_frame())
        caption(_rgb_frame(size=(12, 16)))
        assert caption.mask is not None
        assert caption.mask.shape[:2] == (12, 16)


# ------------------------------------------------------------- Аугментации ---


class TestStreamRandomCrop:
    """Проверки случайного вырезания фрагмента кадра."""

    def test_absolute_size(self) -> None:
        """Целый размер задаёт абсолютные размеры фрагмента."""
        assert StreamRandomCrop(8)(_rgb_frame()).shape[:2] == (8, 8)

    def test_relative_size(self) -> None:
        """Вещественный размер задаёт долю от размера кадра."""
        out = StreamRandomCrop(0.5)(_rgb_frame())
        assert out.shape[:2] == tuple(s // 2 for s in IMG_SIZE)

    def test_shape_is_stable_between_calls(self) -> None:
        """Положение рамки не меняется до сброса."""
        crop = StreamRandomCrop(8)
        first = crop(_rgb_frame())
        assert np.array_equal(first, crop(_rgb_frame()))

    def test_reset_changes_position(self) -> None:
        """Сброс позволяет выбрать другое положение рамки."""
        crop = StreamRandomCrop(8)
        crop(_rgb_frame())
        assert crop.im_size is not None
        crop.reset()
        assert crop.im_size is None
        crop(_rgb_frame())
        assert crop.im_size is not None

    def test_too_big_size_raises(self) -> None:
        """Фрагмент больше кадра вызывает ValueError."""
        with pytest.raises(ValueError, match='больше-равен'):
            StreamRandomCrop(100)(_rgb_frame())

    @pytest.mark.parametrize('size', [0.0, -0.5, 1.5])
    def test_relative_size_out_of_range_raises(self, size: float) -> None:
        """Относительный размер вне (0, 1] вызывает ValueError."""
        with pytest.raises(ValueError, match='интервале'):
            StreamRandomCrop(size)(_rgb_frame())


class TestStreamRandomFlip:
    """Проверки отражения и поворота кадра."""

    def test_vertical_flip(self) -> None:
        """Отражение по вертикали меняет строки кадра."""
        out = StreamRandomFlip(ud=True, lr=False, rot=0)(_rgb_frame())
        assert np.array_equal(out, np.flipud(_rgb_frame()))

    def test_horizontal_flip(self) -> None:
        """Отражение по горизонтали меняет столбцы кадра."""
        out = StreamRandomFlip(ud=False, lr=True, rot=0)(_rgb_frame())
        assert np.array_equal(out, np.fliplr(_rgb_frame()))

    def test_rotation(self) -> None:
        """Поворот меняет местами размеры кадра."""
        out = StreamRandomFlip(ud=False, lr=False, rot=1)(_rgb_frame())
        assert out.shape[:2] == IMG_SIZE[::-1]

    def test_transform_is_stable_between_calls(self) -> None:
        """Параметры не перевыбираются до сброса."""
        flip = StreamRandomFlip()
        first = flip(_rgb_frame()).copy()
        assert np.array_equal(first, flip(_rgb_frame()))

    def test_reset_keeps_explicit_params(self) -> None:
        """Явные параметры переживают сброс."""
        flip = StreamRandomFlip(ud=True, lr=False, rot=0)
        flip.reset()
        assert np.array_equal(flip(_rgb_frame()), np.flipud(_rgb_frame()))


class TestStoreLastFrames:
    """Проверки буфера последних кадров."""

    def test_keeps_n_frames(self) -> None:
        """Буфер не превышает заданного числа кадров."""
        store = StoreLastFrames(n=3)
        for index in range(5):
            store(_rgb_frame(index))
        assert len(store.frames) == 3

    def test_fill_with_zeros(self) -> None:
        """Пропуски заполняются нулями."""
        store = StoreLastFrames(n=3, fill_empty_frames='Zeros')
        frames = store(_rgb_frame(1))
        assert len(frames) == 3
        assert not frames[-1].any()

    def test_fill_with_copy(self) -> None:
        """Пропуски заполняются копией первого кадра."""
        store = StoreLastFrames(n=2, fill_empty_frames='copy')
        frames = store(_rgb_frame(1))
        assert np.array_equal(frames[0], frames[1])

    def test_no_fill_by_default(self) -> None:
        """Без параметра заполнения буфер растёт постепенно."""
        store = StoreLastFrames(n=3)
        assert len(store(_rgb_frame())) == 1

    def test_default_name(self) -> None:
        """Имя формируется из числа кадров."""
        assert StoreLastFrames(n=5).name == 'StoreLast5Frames'

    def test_reset_clears_buffer(self) -> None:
        """Сброс очищает буфер кадров."""
        store = StoreLastFrames(n=2)
        store(_rgb_frame())
        store.reset()
        assert store.frames == []


# ------------------------------------------------------- Смешивание и склейка ---


class TestMix:
    """Проверки наложения результата конвейера на исходный кадр."""

    def test_alpha_zero_returns_input(self) -> None:
        """При нулевой прозрачности возвращается исходный кадр."""
        out = Mix(np.zeros_like, alpha=0.0)(_rgb_frame())
        assert np.array_equal(out, _rgb_frame())

    def test_alpha_one_returns_inner(self) -> None:
        """При единичной прозрачности возвращается результат конвейера."""
        out = Mix(lambda img: np.full_like(img, 7), alpha=1.0)(_rgb_frame())
        assert np.array_equal(out, np.full_like(_rgb_frame(), 7))

    def test_dtype_mismatch_raises(self) -> None:
        """Несовпадение типов входа и выхода вызывает TypeError."""
        with pytest.raises(TypeError, match='!='):
            Mix(lambda img: img.astype(np.float64))(_rgb_frame())

    def test_shape_mismatch_raises(self) -> None:
        """Несовпадение форм входа и выхода вызывает ValueError."""
        with pytest.raises(ValueError, match='!='):
            Mix(
                lambda img: np.zeros(
                    (img.shape[0] // 2, img.shape[1] // 2, 3),
                    np.uint8,
                ),
            )(_rgb_frame())


class TestConcat:
    """Проверки склейки результатов нескольких конвейеров."""

    def test_horizontal(self) -> None:
        """Горизонтальная склейка расширяет ширину вдвое."""
        out = Concat([None, None])(_rgb_frame())
        assert out.shape == (IMG_SIZE[0], IMG_SIZE[1] * 2, 3)

    def test_vertical(self) -> None:
        """Вертикальная склейка удваивает высоту."""
        out = Concat([None, None], mode='v')(_rgb_frame())
        assert out.shape == (IMG_SIZE[0] * 2, IMG_SIZE[1], 3)

    def test_non_list_raises(self) -> None:
        """Передача не списка вызывает TypeError."""
        with pytest.raises(TypeError, match='списком или кортежем'):
            Concat(bgr2rgb)


class TestComparators:
    """Проверки демонстрационных коллажей."""

    def test_compare_two_filters_layout(self) -> None:
        """Коллаж двух фильтров имеет размер вдвое больше по обеим осям."""
        out = CompareTwoFilters(rgb2bgr, rgb2bgr)(_rgb_frame())
        assert out.shape == (IMG_SIZE[0] * 2, IMG_SIZE[1] * 2, 3)

    def test_compare_two_filters_resets(self) -> None:
        """Сброс сравнителя долетает до вложенных фильтров."""
        calls = []

        class Resettable:
            def __call__(self, img: np.ndarray) -> np.ndarray:
                return img

            def reset(self) -> None:
                calls.append(1)

        CompareTwoFilters(Resettable(), Resettable()).reset()
        assert calls == [1, 1]

    def test_compare_filters_with_target_layout(self) -> None:
        """Коллаж сравнения с эталоном собирает вход, эталон, предсказания и diff."""
        pair = np.vstack([_rgb_frame(), _rgb_frame()])
        out = CompareFiltersWithTarget([rgb2bgr])(pair)
        assert out.shape == (IMG_SIZE[0] * 2, IMG_SIZE[1] * 2, 3)

    def test_compare_filters_with_target_resets(self) -> None:
        """Сброс сравнителя долетает до всех фильтров."""
        calls = []

        class Resettable:
            def __call__(self, img: np.ndarray) -> np.ndarray:
                return img

            def reset(self) -> None:
                calls.append(1)

        CompareFiltersWithTarget([Resettable(), Resettable()]).reset()
        assert calls == [1, 1]


# ------------------------------------------------------------ KerasModel ---


class TestKerasModel:
    """Проверки обёртки keras-модели."""

    def test_predict_is_called(self) -> None:
        """Обёртка вызывает predict и убирает первое измерение."""

        class FakeModel:
            stateful = True

            def predict(self, batch: np.ndarray, verbose: int) -> np.ndarray:
                assert batch.shape[0] == 1
                assert verbose == 0
                return np.zeros((1, *IMG_SIZE, 3), np.float32)

        out = KerasModel(FakeModel())(_rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)

    def test_reset_states_called(self) -> None:
        """Сброс вызывает reset_states, если он есть у модели."""

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            def predict(self, batch: np.ndarray, verbose: int) -> np.ndarray:  # noqa: ARG002
                return np.zeros((1, *IMG_SIZE, 3), np.float32)

            def reset_states(self) -> None:
                self.calls += 1

        model = FakeModel()
        filter_ = KerasModel(model)
        assert model.calls == 1
        filter_.reset()
        assert model.calls == 2

    def test_close_restores_stateful(self) -> None:
        """Закрытие восстанавливает исходное значение stateful."""

        class FakeModel:
            stateful = False

            def predict(self, batch: np.ndarray, verbose: int) -> np.ndarray:  # noqa: ARG002
                return np.zeros((1, *IMG_SIZE, 3), np.float32)

        model = FakeModel()
        model.stateful = True  # noqa: FBT003
        filter_ = KerasModel(model)
        model.stateful = False  # noqa: FBT003
        filter_.close()
        assert model.stateful is True


# --------------------------------------------------- Семантическая сегментация ---


class TestDrawSemanticSegments:
    """Проверки отрисовки семантической сегментации."""

    def test_argmax_classes(self) -> None:
        """Без num_classes применяется argmax по последнему измерению."""
        segments: np.ndarray = np.zeros((*IMG_SIZE, 3), np.float32)
        segments[..., 0] = 1.0
        out = DrawSemanticSegments()(segments)
        assert out.shape == (*IMG_SIZE, 3)
        assert out.dtype == np.uint8

    def test_binary_segmentation(self) -> None:
        """Одноканальный тензор трактуется как бинарная сегментация."""
        segments: np.ndarray = _gray_frame(5).astype(np.float32) / 255
        out = DrawSemanticSegments()(segments)
        assert out.shape == (*IMG_SIZE, 3)

    def test_num_classes_with_segmentator(self) -> None:
        """При заданном num_classes применяется внешний сегментатор."""
        segments: np.ndarray = np.ones(IMG_SIZE, np.float32)

        def segmentator(img: np.ndarray) -> np.ndarray:
            """Возвращает карту классов, проверяя размерность входа."""
            assert img.ndim == 2
            return segments

        out = DrawSemanticSegments(num_classes=3, segmentator=segmentator)(
            _gray_frame(),
        )
        assert out.shape == (*IMG_SIZE, 3)

    def test_color_input_brightness(self) -> None:
        """Цветной вход даёт яркость по серому каналу."""
        dtype, value = DrawSemanticSegments._brightness(_rgb_frame())  # noqa: SLF001
        assert dtype == np.uint8
        assert value.ndim == 2

    def test_unexpected_shape_raises(self) -> None:
        """Неожиданная размерность входа вызывает ValueError."""
        with pytest.raises(ValueError, match='Неожиданный размер'):
            DrawSemanticSegments._brightness(  # noqa: SLF001
                np.zeros((2, 2, 2, 2), np.uint8),
            )

    def test_reset_calls_segmentator(self) -> None:
        """Сброс долетает до сегментатора."""
        calls = []

        def segmentator(img: np.ndarray) -> np.ndarray:  # noqa: ARG001
            return np.zeros((*IMG_SIZE, 3), np.float32)

        segmentator.reset = lambda: calls.append(1)  # type: ignore[attr-defined]
        DrawSemanticSegments(segmentator=segmentator).reset()
        assert calls == [1]


# ----------------------------------------------------------------- OptFlow ---


class TestOptFlow:
    """Проверки оптического потока и деформации изображений."""

    def test_flow_shape(self) -> None:
        """Поток имеет размер кадра и два канала смещений."""
        flow = OptFlow(winsize=5)(_gray_frame(), _gray_frame(1))
        assert flow.shape == (*IMG_SIZE, 2)

    def test_flow_of_identical_frames_is_zero(self) -> None:
        """Поток между одинаковыми кадрами равен нулю."""
        flow = OptFlow(winsize=5)(_gray_frame(), _gray_frame())
        assert np.allclose(flow, 0, atol=1e-3)

    def test_color_input_is_converted(self) -> None:
        """Цветные кадры переводятся в оттенки серого."""
        flow = OptFlow(winsize=5)(_rgb_frame(), _rgb_frame(1))
        assert flow.shape == (*IMG_SIZE, 2)

    def test_initial_flow_is_used(self) -> None:
        """Начальная аппроксимация не переписывается на месте."""
        initial: np.ndarray = np.zeros((*IMG_SIZE, 2), np.float32)
        initial[...] = 0.5
        flow = OptFlow(winsize=5)(_gray_frame(), _gray_frame(1), flow=initial)
        assert np.all(initial == 0.5)
        assert flow.shape == (*IMG_SIZE, 2)

    def test_create_meshgrid(self) -> None:
        """Координатная сетка совпадает с размером кадра."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE)
        assert mesh.shape == (*IMG_SIZE, 2)
        assert mesh[0, 0].tolist() == [0.0, 0.0]

    def test_apply_flow2meshgrid_zero_flow(self) -> None:
        """При нулевом потоке сетка не меняется."""
        zero_flow: np.ndarray = np.zeros((*IMG_SIZE, 2), np.float32)
        mesh = OptFlow.create_meshgrid(IMG_SIZE)
        assert np.array_equal(OptFlow.apply_flow2meshgrid(zero_flow, mesh), mesh)

    def test_apply_flow2meshgrid_shape_mismatch(self) -> None:
        """Несовпадение форм потока и сетки вызывает ValueError."""
        bad_flow: np.ndarray = np.zeros((*IMG_SIZE, 2), np.float32)
        with pytest.raises(ValueError, match='не совпадают'):
            OptFlow.apply_flow2meshgrid(bad_flow, np.zeros((4, 4, 2), np.float32))

    def test_apply_meshgrid2img_gray(self) -> None:
        """Деформация серого кадра сохраняет его форму."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE)
        out = OptFlow.apply_meshgrid2img(mesh, _gray_frame())
        assert out.shape == IMG_SIZE

    def test_apply_meshgrid2img_color(self) -> None:
        """Деформация цветного кадра сохраняет число каналов."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE)
        out = OptFlow.apply_meshgrid2img(mesh, _rgb_frame())
        assert out.shape == (*IMG_SIZE, 3)

    def test_apply_meshgrid2img_nearest(self) -> None:
        """Округление координат даёт ближайший сосед."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE) + 0.6
        out = OptFlow.apply_meshgrid2img(
            mesh,
            _gray_frame(),
            nearest_interpolation=True,
        )
        assert out.shape == IMG_SIZE

    def test_apply_meshgrid2img_cval_list(self) -> None:
        """Список cval применяется по каналам."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE) + 100
        out = OptFlow.apply_meshgrid2img(mesh, _rgb_frame(), cval=[0.0, 0.0, 0.0])
        assert out.shape == (*IMG_SIZE, 3)

    def test_apply_meshgrid2img_bad_ndim(self) -> None:
        """Четырёхмерное изображение вызывает ValueError."""
        mesh = OptFlow.create_meshgrid(IMG_SIZE)
        with pytest.raises(ValueError, match='2 или 3 измерения'):
            OptFlow.apply_meshgrid2img(mesh, np.zeros((*IMG_SIZE, 2, 2), np.uint8))

    def test_apply_flow2img_recovers_shift(self) -> None:
        """Сдвиг восстанавливается потоком и обратной деформацией."""
        img = _gray_frame()
        moved = np.roll(img, 2, axis=1)
        flow = OptFlow(winsize=5)(img, moved)
        restored = OptFlow.apply_flow2img(flow, img)
        assert np.abs(restored.astype(int) - moved.astype(int)).mean() < 5

    def test_seq_flows_length(self) -> None:
        """Последовательность потоков совпадает с числом кадров."""
        imgs = [_gray_frame(i) for i in range(4)]
        flows = OptFlow(winsize=5).seq_flows(imgs)
        assert len(flows) == 4

    def test_seq_flows_without_cum_sum(self) -> None:
        """Без накопления возвращается на один поток меньше."""
        imgs = [_gray_frame(i) for i in range(4)]
        flows = OptFlow(winsize=5).seq_flows(imgs, cum_sum=False)
        assert len(flows) == 3

    def test_apply_flows2img_length(self) -> None:
        """Восстановленная последовательность содержит все кадры."""
        imgs = [_gray_frame(i) for i in range(3)]
        opt_flow = OptFlow(winsize=5)
        flows = opt_flow.seq_flows(imgs)
        restored = opt_flow.apply_flows2img(flows, imgs[0], desc=None)
        assert len(restored) == len(imgs) + 1


# ------------------------------------------------------- Вспомогательные функции ---


class TestResolveHelpers:
    """Проверки вспомогательных функций разрешения путей и параметров."""

    def test_to_3channels_rgb_from_gray(self) -> None:
        """Серый кадр разворачивается в три канала."""
        assert _to_3channels_rgb(_gray_frame()).shape == (*IMG_SIZE, 3)

    def test_to_3channels_rgb_from_single_channel(self) -> None:
        """Кадр с одним каналом разворачивается в три."""
        assert _to_3channels_rgb(_gray_frame()[..., None]).shape == (*IMG_SIZE, 3)

    def test_to_3channels_rgb_keeps_color(self) -> None:
        """Трёхканальный кадр остаётся без изменений."""
        assert _to_3channels_rgb(_rgb_frame()).shape == (*IMG_SIZE, 3)

    def test_to_3channels_rgb_bad_channels(self) -> None:
        """Неожиданное число каналов вызывает ValueError."""
        with pytest.raises(ValueError, match='не соответствует ожидаемому'):
            _to_3channels_rgb(np.zeros((*IMG_SIZE, 5), np.uint8))

    def test_to_3channels_bgr_reverses(self) -> None:
        """BGR получается переворотом каналов RGB."""
        rgb = _rgb_frame()
        assert np.array_equal(_to_3channels_bgr(rgb), _to_3channels_rgb(rgb)[..., ::-1])

    def test_resolve_cvals_scalar(self) -> None:
        """Скалярное значение cval тиражируется по каналам."""
        assert _resolve_cvals(1.5, 3) == [1.5, 1.5, 1.5]

    def test_resolve_cvals_sequence(self) -> None:
        """Список cval нужной длины принимается как есть."""
        assert _resolve_cvals((0.0, 1.0, 2.0), 3) == [0.0, 1.0, 2.0]

    def test_resolve_cvals_length_mismatch(self) -> None:
        """Список неверной длины вызывает ValueError."""
        with pytest.raises(ValueError, match='длиной'):
            _resolve_cvals((0.0, 1.0), 3)

    def test_resolve_filter_name(self) -> None:
        """Имя фильтра берётся из атрибута name."""
        assert _resolve_filter_name(PutTime()) == 'NoName_Filter'

    def test_rescale_target_size_true(self) -> None:
        """rescale=True даёт фиксированный размер 1280x720."""
        cap = cv2.VideoCapture()
        assert _rescale_target_size(cap, rescale=True) == (1280, 720)

    def test_rescale_target_size_int(self) -> None:
        """Целочисленный rescale делит размер нацело по кратности."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cap = cv2.VideoCapture(_make_video(Path(temp_dir) / 'v.avi'))
            assert _rescale_target_size(cap, rescale=2) == (32, 24)
            cap.release()

    def test_rescale_target_size_sequence(self) -> None:
        """Последовательность rescale используется как есть."""
        cap = cv2.VideoCapture()
        assert _rescale_target_size(cap, rescale=(10, 20)) == (10, 20)

    def test_rescale_target_size_invalid(self) -> None:
        """Некорректный rescale вызывает ValueError."""
        cap = cv2.VideoCapture()
        with pytest.raises(ValueError, match='задан неверно'):
            _rescale_target_size(cap, rescale='bad')


class TestFilterWithOomFallback:
    """Проверки понижения разрешения при нехватке памяти."""

    def test_no_rescale(self) -> None:
        """Без масштабирования размер кадра не меняется."""
        out = _filter_with_oom_fallback(rgb2gray, _rgb_frame(), rescale=False)
        assert out.shape == IMG_SIZE

    def test_rescale_true(self) -> None:
        """rescale=True приводит кадр к 1280x720."""
        out = _filter_with_oom_fallback(rgb2gray, _rgb_frame(), rescale=True)
        assert out.shape[:2] == (720, 1280)

    def test_rescale_int(self) -> None:
        """Целочисленный rescale округляет размер по кратности."""
        out = _filter_with_oom_fallback(rgb2gray, _rgb_frame(), rescale=8)
        assert out.shape[:2] == (24, 32)

    def test_rescale_sequence(self) -> None:
        """Последовательный rescale используется как есть."""
        out = _filter_with_oom_fallback(rgb2gray, _rgb_frame(), rescale=(10, 20))
        assert out.shape[:2] == (20, 10)

    def test_oom_halves_size(self) -> None:
        """При RuntimeError разрешение уменьшается вдвое."""

        class Flaky:
            calls = 0

            def __call__(self, img: np.ndarray) -> np.ndarray:
                Flaky.calls += 1
                if Flaky.calls == 1:
                    msg = 'out of memory'
                    raise RuntimeError(msg)
                return img

        out = _filter_with_oom_fallback(Flaky(), _rgb_frame(), rescale=False)
        assert out.shape[:2] == tuple(s // 2 for s in IMG_SIZE)


class TestResolveOutFiles:
    """Проверки определения имён выходных файлов."""

    def test_video_out_file_in_dir(self) -> None:
        """Путь к папке дополняется именем фильтра."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = _resolve_video_out_file(
                Path('in.avi'),
                temp_dir,
                'MyFilter',
                Path('/default'),
                save2subfolder=False,
            )
            assert out == Path(temp_dir) / 'MyFilter.avi'

    def test_video_out_file_explicit_name(self) -> None:
        """Явное имя файла приводится к расширению avi."""
        out = _resolve_video_out_file(
            Path('in.avi'),
            'res.mkv',
            'MyFilter',
            Path('/default'),
            save2subfolder=False,
        )
        assert out == Path('res.avi')

    def test_video_out_file_unsupported_source(self) -> None:
        """Неподдерживаемый формат источника даёт None."""
        assert (
            _resolve_video_out_file(
                Path('in.txt'),
                None,
                'MyFilter',
                Path('/default'),
                save2subfolder=False,
            )
            is None
        )

    def test_video_out_file_subfolder(self) -> None:
        """При save2subfolder результат кладётся в подпапку."""
        out = _resolve_video_out_file(
            Path('in.avi'),
            'res',
            'MyFilter',
            Path('/default'),
            save2subfolder=True,
        )
        assert out == Path('res') / 'MyFilter.avi'

    def test_image_out_file_in_dir(self) -> None:
        """Путь к папке дополняется именем фильтра с расширением png."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = _resolve_image_out_file(
                temp_dir,
                'MyFilter',
                Path('/default'),
                save2subfolder=False,
            )
            assert out == Path(temp_dir) / 'MyFilter.png'

    def test_image_out_file_default_dir(self) -> None:
        """Без выходного пути используется папка по умолчанию."""
        out = _resolve_image_out_file(
            None,
            'MyFilter',
            Path('/default'),
            save2subfolder=False,
        )
        assert out == Path('/default/MyFilter.png')

    def test_image_out_file_explicit_name(self) -> None:
        """Явное имя файла приводится к расширению png."""
        out = _resolve_image_out_file(
            'res.jpg',
            'MyFilter',
            Path('/default'),
            save2subfolder=False,
        )
        assert out == Path('res.png')


# ------------------------------------------------------- Применение к файлам ---


class TestApply2Video:
    """Проверки применения фильтра к видеофайлам."""

    def test_writes_output(self) -> None:
        """После обработки создаётся видеофайл с заданным числом кадров."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            apply2video(rgb2gray, src, dst, verbose=False, rescale=False)
            assert dst.is_file()
            assert VideoGenerator.get_file_total_frames(dst) > 0

    def test_step_skips_frames(self) -> None:
        """Шаг больше единицы уменьшает число кадров результата."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            apply2video(rgb2gray, src, dst, verbose=False, rescale=False, step=2)
            assert VideoGenerator.get_file_total_frames(dst) < VIDEO_FRAMES

    def test_skip_existed(self) -> None:
        """Существующий выходной файл не перезаписывается."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            dst.write_bytes(b'stale')
            apply2video(
                rgb2gray,
                src,
                dst,
                verbose=False,
                rescale=False,
                skip_existed=True,
            )
            assert dst.read_bytes() == b'stale'

    def test_rescale_changes_size(self) -> None:
        """Масштабирование меняет размер кадров результата."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            apply2video(rgb2gray, src, dst, verbose=False, rescale=(16, 12))
            cap = cv2.VideoCapture(os.fspath(dst))
            assert (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                ),
            ) == (16, 12)
            cap.release()

    def test_processes_directory(self) -> None:
        """Обработка папки проходит по всем файлам внутри."""

        def named_filter(img: np.ndarray) -> np.ndarray:
            return rgb2gray(img)

        named_filter.name = 'NamedFilter'  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / 'in'
            dst_dir = Path(temp_dir) / 'out'
            src_dir.mkdir()
            _make_video(src_dir / 'a.avi')
            _make_video(src_dir / 'b.avi')
            apply2video(
                named_filter,
                src_dir,
                dst_dir,
                verbose=False,
                rescale=False,
                save2subfolder=True,
            )
            # При save2subfolder выходное имя задаётся именем фильтра:
            assert (dst_dir / 'NamedFilter.avi').is_file()

    def test_filter_receives_rgb(self) -> None:
        """Фильтр получает кадр в RGB, а не в BGR."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            seen = []

            def spy(img: np.ndarray) -> np.ndarray:
                seen.append(img.copy())
                return rgb2gray(img)

            apply2video(spy, src, dst, verbose=False, rescale=False)
            assert seen


class TestApply2Image:
    """Проверки применения фильтра к картинкам."""

    def test_writes_output(self) -> None:
        """После обработки создаётся картинка."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_image(Path(temp_dir) / 'in.png')
            dst = Path(temp_dir) / 'out.png'
            apply2image(rgb2gray, src, dst, rescale=False)
            assert dst.is_file()

    def test_skip_existed(self) -> None:
        """Существующий выходной файл не перезаписывается."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_image(Path(temp_dir) / 'in.png')
            dst = Path(temp_dir) / 'out.png'
            dst.write_bytes(b'stale')
            apply2image(rgb2gray, src, dst, rescale=False)
            assert dst.read_bytes() == b'stale'

    def test_float_output_is_converted(self) -> None:
        """Вещественный результат фильтра приводится к uint8."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_image(Path(temp_dir) / 'in.png')
            dst = Path(temp_dir) / 'out.png'
            apply2image(_to_float, src, dst, rescale=False)
            assert dst.is_file()

    def test_processes_directory(self) -> None:
        """Обработка папки проходит по всем картинкам внутри."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / 'in'
            src_dir.mkdir()
            _make_image(src_dir / 'a.png')
            _make_image(src_dir / 'b.png')
            dst_dir = Path(temp_dir) / 'out'
            apply2image(
                rgb2gray,
                src_dir,
                dst_dir,
                rescale=False,
                save2subfolder=True,
            )
            assert (dst_dir / 'a.png').is_file()
            assert (dst_dir / 'b.png').is_file()

    def test_creates_nested_dirs(self) -> None:
        """Отсутствующие вложенные папки создаются."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_image(Path(temp_dir) / 'in.png')
            dst = Path(temp_dir) / 'a' / 'b' / 'out.png'
            apply2image(rgb2gray, src, dst, rescale=False)
            assert dst.is_file()


class TestConvertVideos:
    """Проверки пакетного пересжатия видео в mp4."""

    def test_converts_all_videos(self) -> None:
        """Все видео из папки пережимаются в mp4."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / 'in'
            dst_dir = Path(temp_dir) / 'out'
            src_dir.mkdir()
            _make_video(src_dir / 'a.avi')
            _make_video(src_dir / 'b.avi')
            convert_videos(src_dir, dst_dir)
            assert (dst_dir / 'a.mp4').is_file()
            assert (dst_dir / 'b.mp4').is_file()

    def test_skip_existed(self) -> None:
        """Существующие mp4 не пересжимаются заново."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / 'in'
            dst_dir = Path(temp_dir) / 'out'
            src_dir.mkdir()
            dst_dir.mkdir()
            _make_video(src_dir / 'a.avi')
            (dst_dir / 'a.mp4').write_bytes(b'stale')
            convert_videos(src_dir, dst_dir)
            assert (dst_dir / 'a.mp4').read_bytes() == b'stale'


class TestVideoGeneratorErrors:
    """Проверки ошибок определения числа кадров."""

    def test_invalid_extension_raises(self) -> None:
        """Неподдерживаемое расширение файла вызывает ValueError."""
        with pytest.raises(ValueError, match='Ожидалось получить'):
            VideoGenerator.get_file_total_frames('in.txt')

    def test_unreadable_video_raises(self) -> None:
        """Невозможность открыть видеофайл вызывает RuntimeError."""
        with pytest.raises(RuntimeError, match='Ошибка открытия файла'):
            VideoGenerator.get_file_total_frames('missing.avi')

    def test_list_with_bad_extension_raises(self) -> None:
        """Список с неподдерживаемым расширением вызывает ValueError."""
        with pytest.raises(ValueError, match='только изображения'):
            VideoGenerator.get_file_total_frames(['a.png', 'b.txt'])


# ------------------ Догоняем оставшиеся ветки и проверки ------------------


class TestRes:
    """Проверки объединения входного и выходного изображения."""

    def test_horizontal(self) -> None:
        """Горизонтальное объединение расширяет ширину вдвое."""
        out = Res(rgb2bgr)(_rgb_frame())
        assert out.shape == (IMG_SIZE[0], IMG_SIZE[1] * 2, 3)

    def test_vertical(self) -> None:
        """Вертикальное объединение удваивает высоту."""
        out = Res(rgb2bgr, mode='V')(_rgb_frame())
        assert out.shape == (IMG_SIZE[0] * 2, IMG_SIZE[1], 3)

    def test_dtype_mismatch_raises(self) -> None:
        """Несовпадение типов входа и выхода вызывает TypeError."""
        with pytest.raises(TypeError, match='!='):
            Res(lambda img: img.astype(np.float64))(_rgb_frame())


class TestAsType:
    """Проверки приведения типа тензора."""

    def test_converts_dtype(self) -> None:
        """Кадр приводится к заданному типу."""
        assert AsType(np.float32)(_rgb_frame()).dtype == np.float32


class TestViReadGuards:
    """Проверки защитных проверок ViRead."""

    def test_count_frames(self) -> None:
        """Дешёвый подсчёт кадров совпадает с числом кадров в файле."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with ViRead(path) as reader:
                assert reader.count_frames() == VIDEO_FRAMES

    def test_list_of_sources(self) -> None:
        """Список путей выбирается случайным образом."""
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _make_video(Path(temp_dir) / 'a.avi')
            second = _make_video(Path(temp_dir) / 'b.avi')
            with ViRead([first, second]) as reader:
                assert reader() is not None

    def test_missing_file_raises(self) -> None:
        """Несуществующий файл вызывает ValueError."""
        with pytest.raises(ValueError, match='Ошибка открытия файла'):
            ViRead('no_such_video.avi')

    def test_reversed_range_raises(self) -> None:
        """Диапазон с началом больше конца вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with pytest.raises(ValueError, match='всего'):
                ViRead(path, start_frame=[5, 2])

    def test_bad_start_frame_type_raises(self) -> None:
        """Неверный тип start_frame вызывает TypeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with pytest.raises(TypeError, match='числом или диапазоном'):
                ViRead(path, start_frame='bad')

    def test_on_end_close(self) -> None:
        """Режим close завершает поток принудительно."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with ViRead(path, on_end='close') as reader:
                for _ in range(VIDEO_FRAMES):
                    assert reader() is not None
                assert reader() is None

    def test_bad_on_end_raises(self) -> None:
        """Неизвестный режим on_end вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with ViRead(path, on_end='bad') as reader:
                for _ in range(VIDEO_FRAMES):
                    reader()
                with pytest.raises(ValueError, match='on_end'):
                    reader()


class TestVideoGeneratorSequence:
    """Проверки работы VideoGenerator с несколькими источниками."""

    @staticmethod
    def _image_dir(root: Path, count: int = 3) -> Path:
        """Создаёт папку с заданным числом картинок."""
        root.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            _make_image(root / f'{index}.png')
        return root

    def test_directory_expanded_to_images(self) -> None:
        """Путь к папке разворачивается в список картинок."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = self._image_dir(Path(temp_dir) / 'imgs')
            generator = VideoGenerator(src)
            assert len(generator) == 3

    def test_iterates_over_several_files(self) -> None:
        """Итерация переходит с одного файла на следующий."""
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _make_video(Path(temp_dir) / 'a.avi', count=3)
            second = _make_video(Path(temp_dir) / 'b.avi', count=2)
            assert len(list(VideoGenerator([first, second]))) == 5

    def test_frame_range_with_negative_end(self) -> None:
        """Отрицательный конец отсчитывается от конца файла."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi', count=5)
            assert VideoGenerator(path, frame_ranges=[(0, -2)]).frame_ranges == [
                range(3),
            ]

    def test_frame_range_list_is_enumeration(self) -> None:
        """Список с неотрицательным вторым элементом трактуется как перечисление."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi', count=5)
            generator = VideoGenerator(path, frame_ranges=[[1, 3]])
            assert generator.frame_ranges == [[1, 3]]

    def test_bad_frame_range_raises(self) -> None:
        """Неверный тип frame_range вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi')
            with pytest.raises(ValueError, match='frame_range'):
                VideoGenerator(path, frame_ranges=['bad'])

    def test_unsupported_file_raises(self) -> None:
        """Неподдерживаемое расширение файла вызывает TypeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'notes.txt'
            path.write_text('x')
            with pytest.raises(TypeError, match='Неподдерживаемый тип файла'):
                VideoGenerator(path)

    def test_length_mismatch_raises(self) -> None:
        """Расхождение числа файлов и диапазонов вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            first = _make_video(Path(temp_dir) / 'a.avi')
            second = _make_video(Path(temp_dir) / 'b.avi')
            with pytest.raises(ValueError, match='должно совпадать'):
                VideoGenerator([first, second], frame_ranges=[None])

    def test_getitem_out_of_range_raises(self) -> None:
        """Номер кадра за пределами последовательности вызывает IndexError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi', count=3)
            generator = VideoGenerator(path)
            with pytest.raises(IndexError, match='>='):
                generator[999]


class TestPipelineConvertGuards:
    """Проверки защитных проверок Pipeline.convert."""

    def test_skip_existed_returns(self) -> None:
        """Существующий выходной файл не перезаписывается."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            dst.write_bytes(b'stale')
            Pipeline([rgb2gray]).convert(src, dst)
            assert dst.read_bytes() == b'stale'

    def test_existing_tmp_file_raises(self) -> None:
        """Занятый временный файл вызывает FileExistsError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            (Path(temp_dir) / 'out_tmp.avi').write_bytes(b'busy')
            with pytest.raises(FileExistsError, match='Временный файл'):
                Pipeline([rgb2gray]).convert(src, Path(temp_dir) / 'out.mp4')

    def test_desc_auto_uses_source_name(self) -> None:
        """Описание auto подменяется именем исходного файла."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            Pipeline([rgb2gray]).convert(
                src,
                Path(temp_dir) / 'out.avi',
                recompress=False,
                desc='auto',
            )
            assert (Path(temp_dir) / 'out.avi').is_file()

    def test_filter_error_is_wrapped(self) -> None:
        """Ошибка фильтра оборачивается в RuntimeError."""

        def failing(img: np.ndarray) -> np.ndarray:
            msg = f'сломалось на кадре {img.shape}'
            raise ValueError(msg)

        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            with pytest.raises(RuntimeError, match='прервана'):
                Pipeline([failing]).convert(
                    src,
                    Path(temp_dir) / 'out.avi',
                    recompress=False,
                )


class TestViSaveGuards:
    """Проверки создания папок в ViSave."""

    def test_creates_parent_dir(self) -> None:
        """Отсутствующая папка под результат создаётся."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / 'nested' / 'dir' / 'out.avi'
            with ViSave(dst) as saver:
                saver(_rgb_frame())
            assert dst.is_file()


class TestRecompErrors:
    """Проверки ошибочных веток пересжатия."""

    def test_bad_source_raises_oserror(self) -> None:
        """Нечитаемый источник вызывает OSError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / 'in.avi'
            src.write_bytes(b'not a video')
            dst = Path(temp_dir) / 'out.mp4'
            with pytest.raises(OSError, match='Ошибка пересжатия'):
                recomp2mp4(src, dst)
            assert not dst.exists()


class TestCleanRemuxGuards:
    """Проверки ошибочных веток clean_remux без битого файла."""

    def test_counter_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Расхождение счётчиков вызывает RuntimeError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            monkeypatch.setattr(
                video_utils,
                '_find_broken_packets',
                lambda _: (5, 4, set()),
            )
            with pytest.raises(RuntimeError, match='не совпало'):
                clean_remux(src, Path(temp_dir) / 'out.avi')

    def test_written_mismatch_removes_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Неполная перепаковка удаляет результат и пробрасывает ошибку."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            monkeypatch.setattr(
                video_utils,
                '_find_broken_packets',
                lambda _: (5, 5, set()),
            )
            monkeypatch.setattr(
                video_utils,
                '_remux_good_packets',
                lambda *_: 2,
            )
            with pytest.raises(RuntimeError, match='меньше ожидаемого'):
                clean_remux(src, dst)
            assert not dst.exists()


class TestDrawSemanticSegmentsGuards:
    """Проверки защитных проверок разметки сегментов."""

    def test_single_channel_color_input(self) -> None:
        """Трёхмерный кадр с одним каналом трактуется как яркость."""
        dtype, value = DrawSemanticSegments._brightness(  # noqa: SLF001
            _rgb_frame()[..., :1],
        )
        assert dtype == np.uint8
        assert value.ndim == 2

    def test_unexpected_segments_shape_raises(self) -> None:
        """Тензор сегментов с двумя каналами вызывает ValueError."""
        with pytest.raises(ValueError, match='Неожиданный размер'):
            DrawSemanticSegments._prepare_segments(  # noqa: SLF001
                np.zeros((*IMG_SIZE, 2), np.float32),
                IMG_SIZE,
            )


class TestStreamRandomCropTypeGuard:
    """Проверка недопустимого типа размера обрезки."""

    def test_bool_size_raises(self) -> None:
        """Логический тип размера вызывает ValueError."""
        with pytest.raises(ValueError, match='целочисленного типа'):
            StreamRandomCrop(True)(_rgb_frame())  # noqa: FBT003


class TestApply2VideoGuards:
    """Проверки защитных веток apply2video."""

    def test_unsupported_source_returns(self) -> None:
        """Неподдерживаемый формат источника просто завершает работу."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = Path(temp_dir) / 'notes.txt'
            src.write_text('x')
            dst = Path(temp_dir) / 'out.avi'
            apply2video(rgb2gray, src, dst, verbose=False, rescale=False)
            assert not dst.exists()

    def test_filter_reset_around_processing(self) -> None:
        """Фильтр сбрасывается до и после обработки файла."""
        calls = []

        class Resettable:
            name = 'Resettable'

            def __call__(self, img: np.ndarray) -> np.ndarray:
                return rgb2gray(img)

            def reset(self) -> None:
                calls.append(1)

        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            apply2video(
                Resettable(),
                src,
                Path(temp_dir) / 'out.avi',
                verbose=False,
                rescale=False,
            )
        assert len(calls) == 2

    def test_float_frame_converted_to_uint8(self) -> None:
        """Вещественный кадр приводится к uint8 перед записью."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            dst = Path(temp_dir) / 'out.avi'
            apply2video(_to_float, src, dst, verbose=False, rescale=False)
            assert dst.is_file()


class TestMiscBranches:
    """Проверки оставшихся единичных веток."""

    def test_oom_fallback_resets_filter(self) -> None:
        """Фильтр сбрасывается перед каждой попыткой применения."""

        class Flaky:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, img: np.ndarray) -> np.ndarray:
                self.calls += 1
                if self.calls == 1:
                    msg = 'out of memory'
                    raise RuntimeError(msg)
                return img

            def reset(self) -> None:
                return

        flaky = Flaky()
        _filter_with_oom_fallback(flaky, _rgb_frame(), rescale=False)
        assert flaky.calls == 2

    def test_convert_videos_skips_non_video(self) -> None:
        """Невидеофайлы в папке пропускаются."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / 'in'
            dst_dir = Path(temp_dir) / 'out'
            src_dir.mkdir()
            _make_video(src_dir / 'a.avi')
            (src_dir) / 'notes.txt'
            (src_dir / 'notes.txt').write_text('x')
            convert_videos(src_dir, dst_dir)
            assert (dst_dir / 'a.mp4').is_file()
            assert not (dst_dir / 'notes.mp4').exists()

    def test_video_out_file_default_dir(self) -> None:
        """Без выходного пути используется папка по умолчанию."""
        out = _resolve_video_out_file(
            Path('in.avi'),
            None,
            'MyFilter',
            Path('/default'),
            save2subfolder=False,
        )
        assert out == Path('/default/MyFilter.avi')

    def test_video_out_file_subfolder_is_dir(self) -> None:
        """Указание на папку при save2subfolder даёт файл внутри неё."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = _resolve_video_out_file(
                Path('in.avi'),
                temp_dir,
                'MyFilter',
                Path('/default'),
                save2subfolder=True,
            )
            assert out == Path(temp_dir) / 'MyFilter.avi'

    def test_image_out_file_subfolder_is_dir(self) -> None:
        """Указание на папку при save2subfolder даёт png внутри неё."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out = _resolve_image_out_file(
                temp_dir,
                'MyFilter',
                Path('/default'),
                save2subfolder=True,
            )
            assert out == Path(temp_dir) / 'MyFilter.png'

    def test_compare_two_filters_reset_without_reset_method(self) -> None:
        """Сброс сравнителя не падает, если у фильтров нет reset."""
        CompareTwoFilters(rgb2gray, rgb2bgr).reset()

    def test_compare_with_target_reset_without_reset_method(self) -> None:
        """Сброс сравнителя с эталоном не падает без reset у фильтров."""
        CompareFiltersWithTarget([rgb2gray, rgb2bgr]).reset()

    def test_keras_model_reset_without_reset_states(self) -> None:
        """Сброс keras-обёртки не падает без метода reset_states."""

        class FakeModel:
            def predict(
                self,
                batch: np.ndarray,  # noqa: ARG002
                verbose: int,  # noqa: ARG002
            ) -> np.ndarray:
                return np.zeros((1, *IMG_SIZE, 3), np.float32)

        KerasModel(FakeModel()).reset()

    def test_store_last_frames_unknown_fill(self) -> None:
        """Неизвестный способ заполнения игнорируется."""
        store = StoreLastFrames(n=2, fill_empty_frames='none')
        assert store.fill_empty_frames is None
        assert len(store(_rgb_frame())) == 1

    def test_resize_and_restore_reset_without_reset_method(self) -> None:
        """Сброс ResizeAndRestore не падает без reset у процессора."""
        ResizeAndRestore((8, 12), rgb2gray).reset()


class TestFinalBranches:
    """Проверки последних единичных веток модуля."""

    def test_recomp_without_output_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Успешный код возврата без результата вызывает NotImplementedError."""

        class FakeResult:
            returncode = 0

        def fake_run(*args: object, **kwargs: object) -> FakeResult:  # noqa: ARG001
            return FakeResult()

        monkeypatch.setattr(subprocess, 'run', fake_run)
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(NotImplementedError, match='Файл не создан'),
        ):
            recomp2mp4(Path(temp_dir) / 'in.avi', Path(temp_dir) / 'out.mp4')

    def test_remux_skips_broken_packets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Пакеты из множества broken не попадают в результат."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi', count=4)
            dst = Path(temp_dir) / 'out.avi'

            def fake_packets(source: str | Path) -> tuple[int, int, set[int]]:
                assert source == src
                return 4, 3, {1}

            def fake_ensure(*, condition: bool, message: str) -> None:  # noqa: ARG001
                return

            monkeypatch.setattr(video_utils, '_find_broken_packets', fake_packets)
            monkeypatch.setattr(video_utils, '_ensure', fake_ensure)
            kept, dropped = clean_remux(src, dst, full_check=False)
            # Пакет с индексом 1 помечен как битый и не переносится:
            assert (kept, dropped) == (3, 1)
            assert dst.is_file()

    def test_count_frames_after_file_removed(self) -> None:
        """Подсчёт кадров удалённого файла вызывает ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'v.avi'
            _make_video(path)
            reader = ViRead(path)
            path.unlink()
            with pytest.raises(ValueError, match='Ошибка открытия файла'):
                reader.count_frames()
            reader.close()

    def test_pipeline_reset_without_reset_method(self) -> None:
        """Сброс конвейера не падает у фильтров без метода reset."""
        Pipeline([rgb2gray]).reset()

    def test_resize_and_restore_same_size_twice(self) -> None:
        """Повторный вызов с тем же размером не пересчитывает постпроцессор."""
        filter_ = ResizeAndRestore((8, 12), rgb2bgr)
        frame = _rgb_frame()
        assert filter_(frame).shape == filter_(frame).shape

    def test_keras_model_close_without_stateful(self) -> None:
        """Закрытие keras-обёртки не падает без атрибута stateful."""

        class FakeModel:
            def predict(
                self,
                batch: np.ndarray,  # noqa: ARG002
                verbose: int,  # noqa: ARG002
            ) -> np.ndarray:
                return np.zeros((1, *IMG_SIZE, 3), np.float32)

        model = FakeModel()
        KerasModel(model).close()
        assert not hasattr(model, 'stateful')

    def test_store_last_frames_unknown_fill_value(self) -> None:
        """Неизвестный способ заполнения не добавляет кадров."""
        store = StoreLastFrames(n=3, fill_empty_frames='repeat')
        assert len(store(_rgb_frame())) == 1

    def test_brightness_two_channels_raises(self) -> None:
        """Кадр с двумя каналами вызывает ValueError."""
        with pytest.raises(ValueError, match='Неожиданный размер'):
            DrawSemanticSegments._brightness(  # noqa: SLF001
                np.zeros((*IMG_SIZE, 2), np.uint8),
            )

    def test_segments_reset_without_reset_method(self) -> None:
        """Сброс разметки не падает у сегментатора без метода reset."""

        def segmentator(img: np.ndarray) -> np.ndarray:  # noqa: ARG001
            return np.zeros((*IMG_SIZE, 3), np.float32)

        DrawSemanticSegments(segmentator=segmentator).reset()

    def test_stream_random_crop_with_size_pair(self) -> None:
        """Размер-пара не дублируется, а используется как есть."""
        crop = StreamRandomCrop((4, 6))
        assert crop(_rgb_frame()).shape[:2] == (4, 6)

    def test_video_generator_getitem_twice(self) -> None:
        """Повторный доступ по индексу переиспользует уже созданные буферы."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_video(Path(temp_dir) / 'v.avi', count=3)
            generator = VideoGenerator(path)
            first = generator[0]
            second = generator[0]
            assert np.array_equal(first, second)

    def test_video_generator_single_image(self) -> None:
        """Одиночная картинка даёт диапазон из одного кадра."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _make_image(Path(temp_dir) / 'one.png')
            assert VideoGenerator(path).frame_ranges == [range(1)]

    def test_clean_remux_not_quiet(self) -> None:
        """Режим без тишины тоже отрабатывает штатно."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src = _make_video(Path(temp_dir) / 'in.avi')
            kept, dropped = clean_remux(
                src,
                Path(temp_dir) / 'out.avi',
                quiet=False,
            )
            assert kept > 0
            assert dropped == 0
