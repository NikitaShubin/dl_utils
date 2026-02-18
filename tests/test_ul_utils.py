"""test_ul_utils.py - Тесты для модуля ul_utils.py."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cv_utils import BBox, Mask
from ul_utils import ULModel, UltralyticsModel, _result2objs, _unpad_ul_masks


# Мок для YOLO класса (доработан для поддержки новых требований)
class MockYOLO:
    """Мок-класс для имитации YOLO с необходимыми атрибутами."""

    def __init__(self) -> None:
        """Инициализация мок-класса YOLO."""
        self.predictor = MagicMock()
        self.predictor.trackers = []  # По умолчанию пустой список трекеров
        self.predict = MagicMock()
        self.track = MagicMock()

    def __call__(self, *args: object, **kwargs: object) -> MagicMock:
        """Вызов мок-класса как функции (для имитации model(...))."""
        return self.predict(*args, **kwargs)


class TestUnpadUlMasks(unittest.TestCase):
    """Тесты для функции _unpad_ul_masks."""

    def test_unpad_square_mask(self) -> None:
        """Тест удаления рамки у квадратной маски."""
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 10:90, 10:90] = 255
        orig_shape = (80, 80)

        result = _unpad_ul_masks(masks, orig_shape)
        assert result.shape == (1, 100, 100)
        assert np.all(result[0, 10:90, 10:90] == 255)

    def test_unpad_rectangular_mask(self) -> None:
        """Тест удаления рамки у прямоугольной маски."""
        masks = np.zeros((1, 120, 100), dtype=np.uint8)
        masks[0, 10:110, 10:90] = 255
        orig_shape = (100, 80)

        result = _unpad_ul_masks(masks, orig_shape)
        assert result.shape[0] == 1
        assert result.shape[1] in [120, 119]
        assert result.shape[2] in [96, 97]

    def test_unpad_multiple_masks(self) -> None:
        """Тест обработки нескольких масок."""
        masks = np.zeros((3, 100, 100), dtype=np.uint8)
        for i in range(3):
            masks[i, 10:90, 10:90] = (i + 1) * 50
        orig_shape = (80, 80)

        result = _unpad_ul_masks(masks, orig_shape)
        assert result.shape == (3, 100, 100)

    def test_unpad_identical_sizes(self) -> None:
        """Тест, когда размеры маски и оригинала совпадают."""
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 20:80, 20:80] = 255
        orig_shape = (100, 100)

        result = _unpad_ul_masks(masks, orig_shape)
        assert result.shape == (1, 100, 100)

    def test_unpad_empty_masks(self) -> None:
        """Тест с пустыми масками."""
        masks = np.zeros((0, 100, 100), dtype=np.uint8)
        orig_shape = (100, 100)

        result = _unpad_ul_masks(masks, orig_shape)
        assert result.shape == (0, 100, 100)


class TestResult2Objs(unittest.TestCase):
    """Тесты для функции _result2objs."""

    def create_mock_result_with_detections(self) -> MagicMock:
        """Создает мок для результата с детекциями."""
        mock_result = MagicMock()
        boxes_mock = MagicMock()
        boxes_mock.xyxy = MagicMock()
        boxes_mock.xyxy.numpy.return_value = np.array(
            [[10, 10, 50, 50], [60, 60, 100, 100]],
        )
        boxes_mock.cls = MagicMock()
        boxes_mock.cls.numpy.return_value = np.array([0, 1])
        boxes_mock.conf = MagicMock()
        boxes_mock.conf.numpy.return_value = np.array([0.9, 0.8])
        boxes_mock.is_track = False
        boxes_mock.id = None

        mock_result.boxes = boxes_mock
        mock_result.names = {0: 'class1', 1: 'class2'}
        mock_result.orig_shape = (200, 200)
        mock_result.masks = None
        mock_result.obb = None
        mock_result.keypoints = None
        return mock_result

    def create_mock_result_with_masks(self) -> MagicMock:
        """Создает мок для результата с масками."""
        mock_result = MagicMock()
        boxes_mock = MagicMock()
        boxes_mock.xyxy = MagicMock()
        boxes_mock.xyxy.numpy.return_value = np.array([[10, 10, 50, 50]])
        boxes_mock.cls = MagicMock()
        boxes_mock.cls.numpy.return_value = np.array([0])
        boxes_mock.conf = MagicMock()
        boxes_mock.conf.numpy.return_value = np.array([0.9])
        boxes_mock.is_track = False
        boxes_mock.id = None

        mock_result.boxes = boxes_mock
        mock_result.names = {0: 'class1'}
        mock_result.orig_shape = (200, 200)

        masks_mock = MagicMock()
        masks_mock.data = MagicMock()
        masks_mock.data.numpy.return_value = (
            np.ones((1, 100, 100), dtype=np.float32) * 0.5
        )
        mock_result.masks = masks_mock
        mock_result.obb = None
        mock_result.keypoints = None
        return mock_result

    def create_mock_empty_result(self) -> MagicMock:
        """Создает мок для пустого результата."""
        mock_result = MagicMock()
        empty_boxes_mock = MagicMock()
        empty_boxes_mock.xyxy = MagicMock()
        empty_boxes_mock.xyxy.numpy.return_value = np.array([])
        empty_boxes_mock.cls = MagicMock()
        empty_boxes_mock.cls.numpy.return_value = np.array([])
        empty_boxes_mock.conf = MagicMock()
        empty_boxes_mock.conf.numpy.return_value = np.array([])
        empty_boxes_mock.is_track = False
        empty_boxes_mock.id = None

        mock_result.boxes = empty_boxes_mock
        mock_result.names = {}
        mock_result.orig_shape = (100, 100)
        mock_result.masks = None
        mock_result.obb = None
        mock_result.keypoints = None
        return mock_result

    def test_result2objs_with_detections(self) -> None:
        """Тест конвертации результата с детекциями."""
        mock_result = self.create_mock_result_with_detections()
        attribs = {'source': 'test'}
        objs = _result2objs(mock_result, attribs)

        assert len(objs) == 2
        assert isinstance(objs[0], BBox)
        assert isinstance(objs[1], BBox)
        assert objs[0].attribs['label'] == 'class1'
        assert objs[0].attribs['confidence'] == 0.9
        assert objs[0].attribs['source'] == 'test'
        assert objs[0].attribs['track_id'] is None

    def test_result2objs_with_masks(self) -> None:
        """Тест конвертации результата с масками."""
        mock_result = self.create_mock_result_with_masks()
        with patch(
            'ul_utils._unpad_ul_masks',
            return_value=np.ones((1, 100, 100), dtype=np.uint8) * 255,
        ):
            objs = _result2objs(mock_result, {})
            assert len(objs) == 1
            assert isinstance(objs[0], Mask)
            assert objs[0].attribs['label'] == 'class1'
            assert objs[0].attribs['confidence'] == 0.9

    def test_result2objs_with_tracking(self) -> None:
        """Тест конвертации с трекингом."""
        mock_result = self.create_mock_result_with_detections()
        mock_result.boxes.is_track = True
        mock_result.boxes.id = MagicMock()
        mock_result.boxes.id.numpy.return_value = np.array([1.0, 2.0])

        objs = _result2objs(mock_result, {})
        assert objs[0].attribs['track_id'] == 1
        assert objs[1].attribs['track_id'] == 2

    def test_result2objs_empty_result(self) -> None:
        """Тест обработки пустого результата (без детекций)."""
        mock_result = self.create_mock_empty_result()
        objs = _result2objs(mock_result, {})
        assert len(objs) == 0
        assert isinstance(objs, list)

    def test_result2objs_not_implemented_obb(self) -> None:
        """Тест вызова исключения для OBB."""
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = MagicMock()
        mock_result.boxes.xyxy.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.cls = MagicMock()
        mock_result.boxes.cls.numpy.return_value = np.array([0])
        mock_result.boxes.conf = MagicMock()
        mock_result.boxes.conf.numpy.return_value = np.array([0.9])
        mock_result.boxes.is_track = False
        mock_result.boxes.id = None
        mock_result.names = {0: 'class1'}
        mock_result.orig_shape = (200, 200)
        mock_result.masks = None
        mock_result.obb = MagicMock()
        mock_result.keypoints = None

        with pytest.raises(
            NotImplementedError,
            match='Повёрнутые прямоугольники не реализованы!',
        ):
            _result2objs(mock_result, {})

    def test_result2objs_not_implemented_keypoints(self) -> None:
        """Тест вызова исключения для keypoints."""
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = MagicMock()
        mock_result.boxes.xyxy.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.cls = MagicMock()
        mock_result.boxes.cls.numpy.return_value = np.array([0])
        mock_result.boxes.conf = MagicMock()
        mock_result.boxes.conf.numpy.return_value = np.array([0.9])
        mock_result.boxes.is_track = False
        mock_result.boxes.id = None
        mock_result.names = {0: 'class1'}
        mock_result.orig_shape = (200, 200)
        mock_result.masks = None
        mock_result.obb = None
        mock_result.keypoints = MagicMock()

        with pytest.raises(NotImplementedError, match='Скелеты не реализованы!'):
            _result2objs(mock_result, {})

    def test_result2objs_custom_attribs(self) -> None:
        """Тест с пользовательскими атрибутами."""
        mock_result = self.create_mock_result_with_detections()
        custom_attribs = {'custom_key': 'custom_value', 'another_key': 123}
        objs = _result2objs(mock_result, custom_attribs)

        assert objs[0].attribs['custom_key'] == 'custom_value'
        assert objs[0].attribs['another_key'] == 123

    def test_result2objs_attribs_priority(self) -> None:
        """Тест приоритета атрибутов."""
        mock_result = self.create_mock_result_with_detections()
        custom_attribs = {'label': 'custom_label', 'confidence': 0.5}
        objs = _result2objs(mock_result, custom_attribs)

        assert objs[0].attribs['label'] == 'class1'
        assert objs[0].attribs['confidence'] == 0.9


class TestUltralyticsModel(unittest.TestCase):
    """Тесты для класса UltralyticsModel."""

    def setUp(self) -> None:
        """Настройка тестовых данных."""
        self.mock_yolo = MockYOLO()

    @patch('ul_utils.YOLO')
    def test_init_with_string_path(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с путем к модели."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            model = UltralyticsModel(str(tmp_path))

            assert model.model == self.mock_yolo
            assert model.tracker is None
            assert model.mode == 'preannotation'
            assert model.frame_ind == 0
            assert model.kwargs == {}

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    @patch('pathlib.Path.home')
    @patch('pathlib.Path.is_file')
    def test_init_with_model_name_only(
        self,
        mock_is_file: MagicMock,
        mock_home: MagicMock,
        mock_yolo_class: MagicMock,
    ) -> None:
        """Тест инициализации только с именем модели."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / 'models'
            models_dir.mkdir()
            model_file = models_dir / 'test.pt'
            model_file.write_text('dummy model data')

            mock_home.return_value = Path(tmpdir)
            mock_is_file.return_value = True

            model = UltralyticsModel('test.pt')
            assert isinstance(model, UltralyticsModel)

    @patch('ul_utils.YOLO')
    def test_init_with_string_and_kwargs(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с путем к модели и дополнительными kwargs."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            model = UltralyticsModel(str(tmp_path), task='detect', verbose=True)

            mock_yolo_class.assert_called_once_with(
                str(tmp_path),
                task='detect',
                verbose=True,
            )
            assert model.kwargs == {}
            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_init_with_yolo_object(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с объектом YOLO."""
        with patch('ul_utils.isinstance') as mock_isinstance:

            def isinstance_side_effect(obj: object, cls: object) -> bool:
                if cls is ULModel and obj is self.mock_yolo:
                    return True
                if cls == (str, Path):
                    return False
                if isinstance(cls, (type, tuple)):
                    return isinstance(obj, cls)
                return False

            mock_isinstance.side_effect = isinstance_side_effect
            mock_yolo_class.return_value = self.mock_yolo

            model = UltralyticsModel(self.mock_yolo)

            assert model.model == self.mock_yolo
            assert model.frame_ind == 0
            assert model.kwargs == {}

    @patch('ul_utils.YOLO')
    def test_init_with_yolo_object_and_kwargs(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с объектом YOLO и kwargs для инференса."""
        with patch('ul_utils.isinstance') as mock_isinstance:

            def isinstance_side_effect(obj: object, cls: object) -> bool:
                if cls is ULModel and obj is self.mock_yolo:
                    return True
                if cls == (str, Path):
                    return False
                if isinstance(cls, (type, tuple)):
                    return isinstance(obj, cls)
                return False

            mock_isinstance.side_effect = isinstance_side_effect
            mock_yolo_class.return_value = self.mock_yolo

            model = UltralyticsModel(self.mock_yolo, conf=0.5, iou=0.6)

            assert model.model == self.mock_yolo
            assert model.kwargs == {'conf': 0.5, 'iou': 0.6}

    @patch('ul_utils.YOLO')
    def test_init_invalid_model_type(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с неверным типом модели."""
        mock_yolo_class.return_value = self.mock_yolo

        with patch('ul_utils.isinstance') as mock_isinstance:
            mock_isinstance.return_value = False

            with pytest.raises(TypeError) as exc_info:
                UltralyticsModel(123)

            error_msg = str(exc_info.value)
            assert 'Объект model должен быть строкой или' in error_msg
            assert (
                'экземпляром класса' in error_msg or 'зкземпляром класса' in error_msg
            )
            assert 'Получен объект типа:' in error_msg

    @patch('ul_utils.YOLO')
    def test_init_with_tracker(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с трекером."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            model = UltralyticsModel(str(tmp_path), tracker='bytetrack.yaml')

            assert model.tracker == 'bytetrack.yaml'
            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_init_with_different_modes(self, mock_yolo_class: MagicMock) -> None:
        """Тест инициализации с разными режимами."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            for mode in ['preannotation', 'preview']:
                model = UltralyticsModel(str(tmp_path), mode=mode)
                assert model.mode == mode

            model = UltralyticsModel(str(tmp_path), mode='PreAnnotation')
            assert model.mode == 'preannotation'

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_reset(self, mock_yolo_class: MagicMock) -> None:
        """Тест сброса состояния с фильтрами."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            mock_filter1 = MagicMock()
            mock_filter1.reset = MagicMock()
            mock_filter2 = MagicMock()
            mock_filter2.reset = MagicMock()

            model = UltralyticsModel(
                str(tmp_path),
                postprocess_filters=[mock_filter1, mock_filter2],
            )

            mock_filter1.reset.reset_mock()
            mock_filter2.reset.reset_mock()
            model.frame_ind = 10
            model.reset()

            assert model.frame_ind == 0
            mock_filter1.reset.assert_called_once()
            mock_filter2.reset.assert_called_once()

            tmp_path.unlink()

    def test_reset_with_nested_predictor(self) -> None:
        """Тест reset, когда трекеры находятся в model.predictor."""
        with patch('ul_utils.isinstance') as mock_isinstance:
            mock_model = MagicMock(spec=ULModel)

            def isinstance_side_effect(obj: object, cls: object) -> bool:
                if cls is ULModel and obj is mock_model:
                    return True
                # Проверяем, что cls — тип или кортеж типов, затем вызываем isinstance
                if isinstance(cls, (type, tuple)):
                    return isinstance(obj, cls)
                return False

            mock_isinstance.side_effect = isinstance_side_effect

            mock_predictor = MagicMock()
            mock_tracker1 = MagicMock()
            mock_tracker2 = MagicMock()
            mock_predictor.trackers = [mock_tracker1, mock_tracker2]
            mock_model.predictor = mock_predictor

            model = UltralyticsModel(mock_model)
            # Сбрасываем счётчики вызовов после автоматического reset в конструкторе
            mock_tracker1.reset.reset_mock()
            mock_tracker2.reset.reset_mock()

            model.reset()

            mock_tracker1.reset.assert_called_once()
            mock_tracker2.reset.assert_called_once()

    @patch('ul_utils.YOLO')
    @patch('ul_utils._result2objs')
    def test_img2df(
        self,
        mock_result2objs: MagicMock,  # noqa: ARG002
        mock_yolo_class: MagicMock,
    ) -> None:
        """Тест метода img2df."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path))
            mock_result = MagicMock()
            model._img2result = MagicMock(return_value=mock_result)  # noqa: SLF001
            mock_df = pd.DataFrame({'label': ['test']})
            model.result2df = MagicMock(return_value=mock_df)

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            df = model.img2df(test_img)

            model._img2result.assert_called_once_with(test_img)  # noqa: SLF001
            model.result2df.assert_called_once_with(mock_result)
            assert model.frame_ind == 1
            assert df is mock_df

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_img2df_with_tracking(self, mock_yolo_class: MagicMock) -> None:
        """Тест img2df с трекингом."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path), tracker='bytetrack.yaml')
            mock_result = MagicMock()
            self.mock_yolo.track.return_value = [mock_result]
            model.result2df = MagicMock(return_value=pd.DataFrame({'label': ['test']}))

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            model._img2result(test_img)  # noqa: SLF001

            self.mock_yolo.track.assert_called_once()
            call_kwargs = self.mock_yolo.track.call_args[1]
            assert call_kwargs['tracker'] == 'bytetrack.yaml'
            assert call_kwargs['persist'] is True
            assert call_kwargs['verbose'] is False

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_draw(self, mock_yolo_class: MagicMock) -> None:
        """Тест метода draw."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path))
            mock_result = MagicMock()
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            model._img2result = MagicMock(return_value=mock_result)  # noqa: SLF001

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = model.draw(test_img)

            model._img2result.assert_called_once_with(test_img)  # noqa: SLF001
            mock_result.plot.assert_called_once()
            assert isinstance(result, np.ndarray)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_call_preannotation_mode(self, mock_yolo_class: MagicMock) -> None:
        """Тест вызова в режиме preannotation."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path), mode='preannotation')
            model.img2df = MagicMock(return_value=pd.DataFrame({'label': ['test']}))

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = model(test_img)

            model.img2df.assert_called_once_with(test_img)
            assert isinstance(result, pd.DataFrame)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_call_preview_mode(self, mock_yolo_class: MagicMock) -> None:
        """Тест вызова в режиме preview."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path), mode='preview')
            model.draw = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = model(test_img)

            model.draw.assert_called_once_with(test_img)
            assert isinstance(result, np.ndarray)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_call_invalid_mode(self, mock_yolo_class: MagicMock) -> None:
        """Тест вызова с неверным режимом."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path), mode='invalid_mode')
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)

            with pytest.raises(
                ValueError,
                match='Неподдерживаемый режим: invalid_mode!',
            ):
                model(test_img)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_result2df_with_bbox(self, mock_yolo_class: MagicMock) -> None:
        """Тест result2df с bounding boxes."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path))
            model.frame_ind = 1

            mock_result = MagicMock()
            mock_result.names = {0: 'class1'}
            mock_result.orig_shape = (200, 200)
            mock_result.orig_img = np.zeros((200, 200, 3), dtype=np.uint8)

            boxes_mock = MagicMock()
            boxes_mock.xyxy = MagicMock()
            boxes_mock.xyxy.numpy.return_value = np.array([[10, 10, 50, 50]])
            boxes_mock.cls = MagicMock()
            boxes_mock.cls.numpy.return_value = np.array([0])
            boxes_mock.conf = MagicMock()
            boxes_mock.conf.numpy.return_value = np.array([0.9])
            boxes_mock.is_track = False
            boxes_mock.id = None

            mock_result.boxes = boxes_mock
            mock_result.masks = None
            mock_result.obb = None
            mock_result.keypoints = None

            mock_bbox = MagicMock(spec=BBox)
            mock_bbox.attribs = {'label': 'class1', 'frame': 1, 'true_frame': 1}

            with (
                patch('ul_utils._result2objs', return_value=[mock_bbox]),
                patch('ul_utils.CVATPoints.from_bbox') as mock_from_bbox,
                patch('ul_utils.concat_dfs') as mock_concat_dfs,
            ):
                mock_points = MagicMock()
                mock_from_bbox.return_value = mock_points
                mock_df = pd.DataFrame({'label': ['class1']})
                mock_points.to_dfrow.return_value = mock_df
                mock_concat_dfs.return_value = mock_df

                df = model.result2df(mock_result)

                mock_from_bbox.assert_called_once_with(mock_bbox)
                assert isinstance(df, pd.DataFrame)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_result2df_with_mask(self, mock_yolo_class: MagicMock) -> None:
        """Тест result2df с маской."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path))
            model.frame_ind = 1

            mock_result = MagicMock()
            mock_result.names = {0: 'class1'}
            mock_result.orig_shape = (200, 200)
            mock_result.orig_img = np.zeros((200, 200, 3), dtype=np.uint8)

            boxes_mock = MagicMock()
            boxes_mock.xyxy = MagicMock()
            boxes_mock.xyxy.numpy.return_value = np.array([[10, 10, 50, 50]])
            boxes_mock.cls = MagicMock()
            boxes_mock.cls.numpy.return_value = np.array([0])
            boxes_mock.conf = MagicMock()
            boxes_mock.conf.numpy.return_value = np.array([0.9])
            boxes_mock.is_track = False
            boxes_mock.id = None

            mock_result.boxes = boxes_mock
            mock_result.masks = MagicMock()
            mock_result.obb = None
            mock_result.keypoints = None

            mock_mask = MagicMock(spec=Mask)
            mock_mask.array = np.zeros((100, 100), dtype=np.uint8)
            mock_mask.attribs = {'label': 'class1', 'frame': 1, 'true_frame': 1}

            with (
                patch('ul_utils._result2objs', return_value=[mock_mask]),
                patch('ul_utils.CVATPoints.from_mask') as mock_from_mask,
                patch('ul_utils.concat_dfs') as mock_concat_dfs,
            ):
                mock_points = MagicMock()
                mock_from_mask.return_value = mock_points
                mock_points.scale = MagicMock(return_value=mock_points)
                mock_df = pd.DataFrame({'label': ['class1']})
                mock_points.to_dfrow.return_value = mock_df
                mock_concat_dfs.return_value = mock_df

                df = model.result2df(mock_result)

                mock_from_mask.assert_called_once_with(mock_mask)
                mock_points.scale.assert_called_once()
                assert isinstance(df, pd.DataFrame)

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_result2df_no_objects(self, mock_yolo_class: MagicMock) -> None:
        """Тест result2df без объектов (пустой результат)."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            model = UltralyticsModel(str(tmp_path))

            with patch('ul_utils._result2objs', return_value=[]):
                df = model.result2df(MagicMock())
                assert df is None

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_result2df_with_postprocess_filters(
        self,
        mock_yolo_class: MagicMock,
    ) -> None:
        """Тест result2df с фильтрами постобработки."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            call_count = 0

            def test_filter(objs: list[object]) -> list[object]:
                nonlocal call_count
                call_count += 1
                return objs

            model = UltralyticsModel(str(tmp_path), postprocess_filters=[test_filter])

            with patch('ul_utils._result2objs', return_value=[]):
                df = model.result2df(MagicMock())
                assert call_count == 1
                assert df is None

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    def test_result2df_with_invalid_filter(self, mock_yolo_class: MagicMock) -> None:
        """Тест result2df с невалидным фильтром."""
        mock_yolo_class.return_value = self.mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            def invalid_filter() -> list:
                return []

            model = UltralyticsModel(
                str(tmp_path),
                postprocess_filters=[invalid_filter],
            )

            mock_bbox = MagicMock(spec=BBox)
            with (
                patch('ul_utils._result2objs', return_value=[mock_bbox]),
                pytest.raises(
                    ValueError,
                    match='Фильтр постобработки должен принимать аргументы',
                ),
            ):
                model.result2df(MagicMock())

            tmp_path.unlink()

    @patch('ul_utils.VideoGenerator')
    @patch('ul_utils.tqdm')
    def test_video2subtask(
        self,
        mock_tqdm: MagicMock,
        mock_video_gen: MagicMock,
    ) -> None:
        """Тест метода video2subtask."""
        with patch('ul_utils.isinstance') as mock_isinstance:
            mock_model = MagicMock(spec=ULModel)

            def isinstance_side_effect(obj: object, cls: object) -> bool:
                if cls is ULModel and obj is mock_model:
                    return True
                # Проверяем, что cls — тип или кортеж типов, затем вызываем isinstance
                if isinstance(cls, (type, tuple)):
                    return isinstance(obj, cls)
                return False

            mock_isinstance.side_effect = isinstance_side_effect

            mock_result1 = MagicMock()
            mock_result2 = MagicMock()
            # Настраиваем, чтобы result.cpu() возвращал сам результат (или что-то)
            mock_result1.cpu.return_value = mock_result1
            mock_result2.cpu.return_value = mock_result2
            mock_model.return_value = [mock_result1, mock_result2]

            model_wrapper = UltralyticsModel(mock_model)
            model_wrapper.result2df = MagicMock(
                side_effect=[
                    pd.DataFrame({'a': [1]}),
                    pd.DataFrame({'b': [2]}),
                ],
            )
            model_wrapper.frame_ind = 0

            mock_video_gen.return_value.__len__.return_value = 2

            # Вызываем без desc, чтобы избежать оборачивания в tqdm
            subtask = model_wrapper.video2subtask('dummy.mp4', desc=None)

            mock_model.assert_called_once_with(source='dummy.mp4', stream=True)
            assert model_wrapper.result2df.call_count == 2
            assert model_wrapper.frame_ind == 2
            assert isinstance(subtask, tuple)
            assert len(subtask) == 3
            _df, path, mapping = subtask
            assert path == 'dummy.mp4'
            assert mapping == {0: 0, 1: 1}
            # tqdm не должен вызываться, так как desc=None
            mock_tqdm.assert_not_called()


class TestUltralyticsModelIntegration(unittest.TestCase):
    """Интеграционные тесты для UltralyticsModel."""

    @patch('ul_utils.YOLO')
    @patch('ul_utils._result2objs')
    def test_full_pipeline_no_detections(
        self,
        mock_result2objs: MagicMock,
        mock_yolo_class: MagicMock,
    ) -> None:
        """Тест полного пайплайна без детекций."""
        mock_yolo = MockYOLO()
        mock_yolo_class.return_value = mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            mock_result = MagicMock()
            boxes_mock = MagicMock()
            boxes_mock.xyxy = MagicMock()
            boxes_mock.xyxy.numpy.return_value = np.array([])
            boxes_mock.cls = MagicMock()
            boxes_mock.cls.numpy.return_value = np.array([])
            boxes_mock.conf = MagicMock()
            boxes_mock.conf.numpy.return_value = np.array([])
            boxes_mock.is_track = False
            boxes_mock.id = None

            mock_result.boxes = boxes_mock
            mock_result.names = {}
            mock_result.orig_shape = (100, 100)
            mock_result.masks = None
            mock_result.obb = None
            mock_result.keypoints = None

            mock_yolo.predict.return_value = [mock_result]
            mock_result2objs.return_value = []

            model = UltralyticsModel('yolov8n.pt', mode='preannotation')
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)

            df = model(test_img)

            assert df is None

            tmp_path.unlink()

    @patch('ul_utils.YOLO')
    @patch('ul_utils._result2objs')
    def test_frame_counter_increment(
        self,
        mock_result2objs: MagicMock,
        mock_yolo_class: MagicMock,
    ) -> None:
        """Тест увеличения счетчика кадров."""
        mock_yolo = MockYOLO()
        mock_yolo_class.return_value = mock_yolo

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = Path(tmp.name)

            mock_result = MagicMock()
            boxes_mock = MagicMock()
            boxes_mock.xyxy = MagicMock()
            boxes_mock.xyxy.numpy.return_value = np.array([])
            boxes_mock.cls = MagicMock()
            boxes_mock.cls.numpy.return_value = np.array([])
            boxes_mock.conf = MagicMock()
            boxes_mock.conf.numpy.return_value = np.array([])
            boxes_mock.is_track = False
            boxes_mock.id = None

            mock_result.boxes = boxes_mock
            mock_result.names = {}
            mock_result.orig_shape = (100, 100)
            mock_result.masks = None
            mock_result.obb = None
            mock_result.keypoints = None

            mock_yolo.predict.return_value = [mock_result]
            mock_result2objs.return_value = []

            model = UltralyticsModel(str(tmp_path))

            test_img = np.zeros((100, 100, 3), dtype=np.uint8)

            assert model.frame_ind == 0
            model.img2df(test_img)
            assert model.frame_ind == 1
            model.img2df(test_img)
            assert model.frame_ind == 2

            model.reset()
            assert model.frame_ind == 0

            tmp_path.unlink()


if __name__ == '__main__':
    unittest.main()
