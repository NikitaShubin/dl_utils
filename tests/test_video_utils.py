"""test_video_utils.py - Тесты для модуля video_utils.py."""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_utils import (
    Pipeline,
    VideoGenerator,
    ViRead,
    ViSave,
    clean_remux,
    recomp2mp4,
)

# Размер кадра синтетических видео (ширина, высота) и частота кадров:
VIDEO_SIZE = (32, 24)
VIDEO_FPS = 10.0
VIDEO_FRAMES = 10

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

    @pytest.mark.xfail(
        reason='Пре-существующий баг: concat рекурсивно зовёт concat, '
        'а не Pipeline.concat -> NameError. Не чиним сейчас.',
        strict=True,
    )
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
