"""Тесты для модуля boxmot_utils.py.

********************************************
*        Тестирование работы с boxmot.     *
*                                          *
********************************************
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, call, patch

import cv2
import numpy as np
import pytest

# Моки для зависимостей перед импортом модуля
sys.modules['boxmot'] = MagicMock()
sys.modules['boxmot.trackers'] = MagicMock()
sys.modules['boxmot.trackers.tracker_zoo'] = MagicMock()

# Импортируем после установки моков
from boxmot_utils import Tracker, suppress_module_logs  # noqa: E402


class TestSuppressModuleLogs:
    """Тесты для контекстного менеджера suppress_module_logs."""

    def test_suppress_logs(self) -> None:
        """Тест подавления логов модуля."""
        mock_logger = Mock()
        with patch('boxmot_utils.logger', mock_logger):
            with suppress_module_logs('test_module'):
                pass

            # Проверяем вызовы отключения/включения логов
            assert mock_logger.disable.called
            assert mock_logger.enable.called
            assert mock_logger.disable.call_args == call('test_module')
            assert mock_logger.enable.call_args == call('test_module')


# Моки для BBox и Mask - они используются в boxmot_utils
class MockBBox:
    """Мок для класса BBox."""

    def __init__(self, xyxy: list, attribs: dict[str, Any]) -> None:
        """Инициализация мока BBox.

        Args:
            xyxy: Координаты [x1, y1, x2, y2]
            attribs: Атрибуты объекта

        """
        self.xyxy = np.array(xyxy)
        self.attribs = attribs.copy()


class MockMask:
    """Мок для класса Mask."""

    def __init__(self, bbox_xyxy: list, attribs: dict[str, Any]) -> None:
        """Инициализация мока Mask.

        Args:
            bbox_xyxy: Координаты bbox [x1, y1, x2, y2]
            attribs: Атрибуты объекта

        """
        self._bbox_xyxy = np.array(bbox_xyxy)
        self.attribs = attribs.copy()

    def asbbox(self, format_str: str) -> list:
        """Возвращает координаты bbox.

        Args:
            format_str: Формат координат (должен быть 'xyxy')

        Returns:
            Список координат [x1, y1, x2, y2]

        """
        assert format_str == 'xyxy'
        return self._bbox_xyxy.tolist()


# Подменяем реальные BBox и Mask нашими моками в импортируемом модуле
import boxmot_utils  # noqa: E402

boxmot_utils.BBox = MockBBox
boxmot_utils.Mask = MockMask


class TestTrackerInit:
    """Тесты инициализации трекера."""

    def test_init_default(self) -> None:
        """Тест инициализации с параметрами по умолчанию."""
        mock_tracker_class = Mock()
        mock_tracker_instance = Mock()
        mock_tracker_class.return_value = mock_tracker_instance

        # Мокаем атрибуты класса Tracker
        with (
            patch.object(Tracker, 'trackers', {'ocsort': mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', []),
            patch('boxmot_utils.logger'),
        ):
            tracker = Tracker()

        assert tracker.tracker_type == 'ocsort'
        assert not tracker.store_untracked
        assert 'reid_weights' in tracker.tracker_kwargs
        assert tracker.tracker is not None

    def test_init_custom_params(self) -> None:
        """Тест инициализации с пользовательскими параметрами."""
        mock_tracker_class = Mock()
        mock_tracker_instance = Mock()
        mock_tracker_class.return_value = mock_tracker_instance

        with (
            patch.object(Tracker, 'trackers', {'ocsort': mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', []),
            patch('boxmot_utils.logger'),
        ):
            tracker = Tracker(
                tracker_type='ocsort',
                store_untracked=True,
                det_thresh=0.5,
                max_age=30,
            )

        assert tracker.tracker_type == 'ocsort'
        assert tracker.store_untracked
        assert tracker.tracker_kwargs['det_thresh'] == 0.5
        assert tracker.tracker_kwargs['max_age'] == 30

    def test_init_reid_tracker(self) -> None:
        """Тест инициализации трекера с ReID."""
        mock_tracker_class = Mock()
        mock_tracker_instance = Mock()
        mock_tracker_class.return_value = mock_tracker_instance

        with (
            patch.object(Tracker, 'trackers', {'botsort': mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', ['botsort']),
            patch('boxmot_utils.logger'),
        ):
            tracker = Tracker(
                tracker_type='botsort',
                reid_weights=Path('/custom/path/model.pt'),
                device='cuda',
            )

        assert tracker.tracker_type == 'botsort'
        assert tracker.tracker_kwargs['reid_weights'] == Path(
            '/custom/path/model.pt',
        )
        assert tracker.tracker_kwargs['device'] == 'cuda'


class TestTrackerMethods:
    """Тесты методов класса Tracker."""

    def setup_method(self) -> None:
        """Настройка перед каждым тестом."""
        # Мокаем зависимости
        self.mock_tracker_class = Mock()
        self.mock_tracker_instance = Mock()
        self.mock_tracker_class.return_value = self.mock_tracker_instance

        # Патчим атрибуты класса Tracker
        self.tracker_patcher1: Any = patch.object(
            Tracker,
            'trackers',
            {'ocsort': self.mock_tracker_class},
        )
        self.tracker_patcher2: Any = patch.object(Tracker, 'reid_trackers', [])
        self.tracker_patcher1.start()
        self.tracker_patcher2.start()

        # Создаем трекер
        with patch('boxmot_utils.logger'):
            self.tracker = Tracker(tracker_type='ocsort')

    def teardown_method(self) -> None:
        """Очистка после каждого теста."""
        self.tracker_patcher1.stop()
        self.tracker_patcher2.stop()

    def test_reset(self) -> None:
        """Тест сброса трекера."""
        # Сохраняем старый трекер
        old_tracker = self.tracker.tracker

        # Создаем новый мок для нового трекера после reset
        new_mock_tracker_instance = Mock()
        self.mock_tracker_class.return_value = new_mock_tracker_instance

        # Вызываем reset
        self.tracker.reset()

        # Проверяем, что создан новый трекер
        assert self.tracker.tracker is not None
        assert self.tracker.tracker is not old_tracker
        assert self.tracker._labels == []  # noqa: SLF001

    def test_raise_on_tracked_obj_without_track(self) -> None:
        """Тест проверки объекта без трека."""
        obj = MockBBox([0, 0, 10, 10], {'label': 'person', 'confidence': 0.9})

        # Не должно вызывать исключение
        self.tracker._raise_on_tracked_obj(obj)  # noqa: SLF001

    def test_raise_on_tracked_obj_with_track(self) -> None:
        """Тест проверки объекта с уже назначенным треком."""
        obj = MockBBox(
            [0, 0, 10, 10],
            {'label': 'person', 'confidence': 0.9, 'track_id': 5},
        )

        # Должно вызвать AttributeError
        with pytest.raises(AttributeError, match='Объект содержит номер трека = 5!'):
            self.tracker._raise_on_tracked_obj(obj)  # noqa: SLF001

    def test_label2cls_new_label(self) -> None:
        """Тест добавления новой метки класса."""
        label = 'person'

        cls_id = self.tracker._label2cls(label)  # noqa: SLF001

        assert cls_id == 0
        assert self.tracker._labels == ['person']  # noqa: SLF001

    def test_label2cls_existing_label(self) -> None:
        """Тест получения ID существующей метки класса."""
        # Добавляем метку
        self.tracker._labels = ['person', 'car']  # noqa: SLF001

        cls_id = self.tracker._label2cls('car')  # noqa: SLF001

        assert cls_id == 1

    def test_obj2det_bbox(self) -> None:
        """Тест конвертации BBox в детекцию."""
        obj = MockBBox([10, 20, 30, 40], {'label': 'person', 'confidence': 0.95})

        # Добавляем метку в список
        self.tracker._labels = ['person']  # noqa: SLF001

        result = self.tracker._obj2det(obj)  # noqa: SLF001

        assert len(result) == 6
        assert result[0] == 10  # x1
        assert result[1] == 20  # y1
        assert result[2] == 30  # x2
        assert result[3] == 40  # y2
        assert result[4] == 0.95  # confidence
        assert result[5] == 0  # class ID

    def test_obj2det_mask(self) -> None:
        """Тест конвертации Mask в детекцию."""
        obj = MockMask([15, 25, 35, 45], {'label': 'car', 'confidence': 0.87})

        # Добавляем метку в список
        self.tracker._labels = ['car']  # noqa: SLF001

        result = self.tracker._obj2det(obj)  # noqa: SLF001

        assert len(result) == 6
        assert result[0] == 15  # x1
        assert result[1] == 25  # y1
        assert result[2] == 35  # x2
        assert result[3] == 45  # y2
        assert result[4] == 0.87  # confidence
        assert result[5] == 0  # class ID

    def test_obj2det_unsupported_type(self) -> None:
        """Тест конвертации неподдерживаемого типа объекта."""

        # Создаем объект с атрибутом attribs, но не BBox и не Mask
        class FakeObj:
            def __init__(self) -> None:
                self.attribs: dict[str, Any] = {}

        obj = FakeObj()

        with pytest.raises(TypeError, match='Неподдерживаемый тип объекта:'):
            self.tracker._obj2det(obj)  # noqa: SLF001

    def test_obj2det_already_tracked(self) -> None:
        """Тест конвертации объекта с уже назначенным треком."""
        obj = MockBBox(
            [0, 0, 10, 10],
            {'label': 'person', 'confidence': 0.9, 'track_id': 1},
        )

        with pytest.raises(AttributeError, match='Объект содержит номер трека = 1!'):
            self.tracker._obj2det(obj)  # noqa: SLF001


class TestTrackerUpdate:
    """Тесты метода update."""

    def setup_method(self) -> None:
        """Настройка перед каждым тестом."""
        # Мокаем зависимости
        self.mock_tracker_class = Mock()
        self.mock_tracker_instance = Mock()
        self.mock_tracker_class.return_value = self.mock_tracker_instance

        # Патчим атрибуты класса Tracker
        self.tracker_patcher1: Any = patch.object(
            Tracker,
            'trackers',
            {'ocsort': self.mock_tracker_class},
        )
        self.tracker_patcher2: Any = patch.object(Tracker, 'reid_trackers', [])
        self.tracker_patcher1.start()
        self.tracker_patcher2.start()

        # Создаем трекер
        with patch('boxmot_utils.logger'):
            self.tracker = Tracker(tracker_type='ocsort')

            # Создаем тестовые объекты
            self.obj1 = MockBBox(
                [10, 20, 30, 40],
                {'label': 'person', 'confidence': 0.95},
            )
            self.obj2 = MockBBox([50, 60, 70, 80], {'label': 'car', 'confidence': 0.87})
            rng = np.random.default_rng()
            self.img = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    def teardown_method(self) -> None:
        """Очистка после каждого теста."""
        self.tracker_patcher1.stop()
        self.tracker_patcher2.stop()

    def test_update_empty_objects(self) -> None:
        """Тест update с пустым списком объектов."""
        # Мокаем возвращаемое значение трекера
        self.mock_tracker_instance.update.return_value = np.zeros((0, 8))

        result = self.tracker.update([], self.img)

        assert result == []
        assert self.mock_tracker_instance.update.called

    def test_update_single_object_tracked(self) -> None:
        """Тест update с одним объектом, который успешно отслежен."""
        self.mock_tracker_instance.update.return_value = np.array(
            [[10, 20, 30, 40, 1, 0.95, 0, 0]],
        )

        result = self.tracker.update([self.obj1], self.img)

        # Проверяем, что объекту назначен track_id (id - 1)
        assert self.obj1.attribs['track_id'] == 0
        # Когда store_untracked=False, метод возвращает только объекты с track_id
        assert len(result) == 1
        assert result[0] is self.obj1

    def test_update_multiple_objects_mixed_tracking(self) -> None:
        """Тест update с несколькими объектами, некоторые отслежены, некоторые нет."""
        # Только первый объект отслежен
        self.mock_tracker_instance.update.return_value = np.array(
            [
                [10, 20, 30, 40, 1, 0.95, 0, 0],  # id=1, ind=0
                # Второй объект (ind=1) не отслежен
            ],
        )

        result = self.tracker.update([self.obj1, self.obj2], self.img)

        # Проверяем track_id
        assert self.obj1.attribs['track_id'] == 0
        assert self.obj2.attribs['track_id'] is None

        # Проверяем результат (store_untracked=False)
        assert len(result) == 1
        assert result[0] is self.obj1

    def test_update_store_untracked_true(self) -> None:
        """Тест update с флагом store_untracked=True."""
        # Создаем трекер с store_untracked=True
        with (
            patch.object(Tracker, 'trackers', {'ocsort': self.mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', []),
            patch('boxmot_utils.logger'),
        ):
            tracker = Tracker(tracker_type='ocsort', store_untracked=True)

        # Мокаем возвращаемое значение трекера
        self.mock_tracker_instance.update.return_value = np.array(
            [
                [10, 20, 30, 40, 1, 0.95, 0, 0],  # id=1, ind=0
            ],
        )

        result = tracker.update([self.obj1, self.obj2], self.img)

        # Проверяем track_id
        assert self.obj1.attribs['track_id'] == 0
        assert self.obj2.attribs['track_id'] is None

        # Проверяем результат (store_untracked=True)
        assert len(result) == 2  # Возвращаются все объекты
        assert self.obj1 in result
        assert self.obj2 in result

    def test_update_track_id_numbering(self) -> None:
        """Тест правильности нумерации треков (начинается с 0)."""
        # Мокаем возвращаемое значение трекера с разными id
        self.mock_tracker_instance.update.return_value = np.array(
            [
                [10, 20, 30, 40, 5, 0.95, 0, 0],  # id=5 → track_id=4
                [50, 60, 70, 80, 10, 0.87, 1, 1],  # id=10 → track_id=9
            ],
        )

        result = self.tracker.update([self.obj1, self.obj2], self.img)

        # Проверяем track_id (id - 1)
        assert self.obj1.attribs['track_id'] == 4
        assert self.obj2.attribs['track_id'] == 9
        assert len(result) == 2

    def test_update_image_conversion(self) -> None:
        """Тест конвертации изображения из RGB в BGR."""
        self.mock_tracker_instance.update.return_value = np.zeros((0, 8))

        with patch('boxmot_utils.cv2.cvtColor') as mock_cvt:
            self.tracker.update([], self.img)

            # Проверяем вызов конвертации
            assert mock_cvt.called
            assert mock_cvt.call_args[0][0] is self.img
            assert mock_cvt.call_args[0][1] == cv2.COLOR_RGB2BGR

    def test_call_method(self) -> None:
        """Тест, что __call__ делегирует вызов методу update."""
        # Проверяем, что на уровне класса __call__ и update - одна и та же функция
        assert self.tracker.__class__.__call__ is self.tracker.__class__.update

        # Дополнительно проверяем фактическое поведение
        # Мокаем возвращаемое значение трекера
        self.mock_tracker_instance.update.return_value = np.zeros((0, 8))

        # Вызываем через update
        result1 = self.tracker.update([], self.img)

        # Вызываем через __call__
        result2 = self.tracker([], self.img)

        # Оба вызова должны дать одинаковый результат
        assert result1 == result2 == []
        # Оба вызова должны быть зафиксированы моком
        assert self.mock_tracker_instance.update.call_count == 2


class TestTrackerIntegration:
    """Интеграционные тесты трекера."""

    def setup_method(self) -> None:
        """Настройка перед каждым тестом."""
        self.mock_tracker_class = Mock()

    def test_tracker_workflow(self) -> None:
        """Тест полного рабочего процесса трекера."""
        mock_tracker_instance = Mock()
        self.mock_tracker_class.return_value = mock_tracker_instance

        with (
            patch.object(Tracker, 'trackers', {'ocsort': self.mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', []),
            patch('boxmot_utils.logger'),
        ):
            # Создаем трекер
            tracker = Tracker(tracker_type='ocsort', store_untracked=True)

            # Создаем тестовые объекты для нескольких кадров
            rng = np.random.default_rng()
            img1 = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
            img2 = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

            obj1_frame1 = MockBBox(
                [10, 20, 30, 40],
                {'label': 'person', 'confidence': 0.95},
            )
            obj2_frame1 = MockBBox(
                [50, 60, 70, 80],
                {'label': 'car', 'confidence': 0.87},
            )

            # Первый кадр - оба объекта отслежены
            mock_tracker_instance.update.return_value = np.array(
                [
                    [10, 20, 30, 40, 1, 0.95, 0, 0],
                    [50, 60, 70, 80, 2, 0.87, 1, 1],
                ],
            )

            result1 = tracker.update([obj1_frame1, obj2_frame1], img1)

            # Проверяем первый кадр
            assert len(result1) == 2
            assert obj1_frame1.attribs['track_id'] == 0
            assert obj2_frame1.attribs['track_id'] == 1

            # Второй кадр - появляется новый объект, старый теряется
            obj1_frame2 = MockBBox(
                [12, 22, 32, 42],
                {'label': 'person', 'confidence': 0.93},
            )
            obj3_frame2 = MockBBox(
                [100, 110, 120, 130],
                {'label': 'person', 'confidence': 0.88},
            )

            mock_tracker_instance.update.return_value = np.array(
                [
                    [12, 22, 32, 42, 1, 0.93, 0, 0],  # id=1 (тот же трек)
                ],
            )

            result2 = tracker.update([obj1_frame2, obj3_frame2], img2)

            # Проверяем второй кадр
            assert len(result2) == 2  # store_untracked=True
            assert obj1_frame2.attribs['track_id'] == 0  # Тот же track_id
            assert obj3_frame2.attribs['track_id'] is None  # Не отслежен

    def test_reset_clears_labels(self) -> None:
        """Тест, что reset очищает список меток."""
        mock_tracker_instance = Mock()
        self.mock_tracker_class.return_value = mock_tracker_instance

        with (
            patch.object(Tracker, 'trackers', {'ocsort': self.mock_tracker_class}),
            patch.object(Tracker, 'reid_trackers', []),
            patch('boxmot_utils.logger'),
        ):
            tracker = Tracker(tracker_type='ocsort')

            # Добавляем метки
            tracker._labels = ['person', 'car', 'bicycle']  # noqa: SLF001
            tracker._label2cls('person')  # noqa: SLF001

            # Сбрасываем
            tracker.reset()

            # Проверяем, что список меток очищен
            assert tracker._labels == []  # noqa: SLF001

            # Проверяем, что новая метка добавляется с индексом 0
            assert tracker._label2cls('dog') == 0  # noqa: SLF001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
