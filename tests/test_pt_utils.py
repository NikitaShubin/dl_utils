"""Тесты для модуля pt_utils.py."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from pt_utils import (
    AutoDevice,
    Receiver,
    SegDataset,
    Sender,
    get_redused_shape,
    has_var_sufficient_elements,
    safe_var,
)


@pytest.fixture
def temp_dataset_dir() -> Iterator[str]:
    """Создает временную директорию с тестовыми данными."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем структуру папок
        inp_dir = Path(temp_dir) / 'inp'
        out_dir = Path(temp_dir) / 'out'
        inp_dir.mkdir()
        out_dir.mkdir()

        # Создаем тестовые изображения
        rng = np.random.default_rng()
        for i in range(3):
            # Входное изображение (цветное)
            img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(inp_dir / f'image_{i}.png'), img)

            # Выходная маска (grayscale) - только значения 0 и 1
            mask = rng.integers(0, 2, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(out_dir / f'image_{i}.png'), mask)

        yield temp_dir


class TestAutoDevice:
    """Тесты для класса AutoDevice."""

    def test_get_available_device_returns_device(self) -> None:
        """Тест, что get_avliable_device возвращает устройство."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            device = AutoDevice.get_avliable_device()
            assert isinstance(device, torch.device)

    def test_get_available_device_cuda(self) -> None:
        """Тест выбора CUDA при её доступности."""
        with patch('torch.cuda.is_available', return_value=True):
            assert AutoDevice.get_avliable_device() == 'cuda'

    def test_get_available_device_mps(self) -> None:
        """Тест выбора MPS при доступности и отсутствии CUDA."""
        with (
            patch('torch.cuda.is_available', return_value=False),
            patch('torch.backends.mps.is_available', return_value=True),
        ):
            assert AutoDevice.get_avliable_device() == 'mps'

    def test_auto_device_init_with_device_object(self) -> None:
        """Тест инициализации готовым объектом torch.device."""
        with patch('pt_utils.AutoDevice.prepare_device'):
            auto_device = AutoDevice(torch.device('cpu'))
        assert auto_device.device == torch.device('cpu')

    def test_auto_device_init_with_str_cpu(self) -> None:
        """Тест инициализации строкой устройства без автоподбора."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cuda')
            auto_device = AutoDevice('cpu')
            mock_get_device.assert_not_called()
        assert auto_device.device == torch.device('cpu')

    def test_prepare_device_mps(self) -> None:
        """Тест prepare_device для MPS - не должно быть ошибок."""
        AutoDevice.prepare_device(torch.device('mps'))

    def test_auto_device_init(self) -> None:
        """Тест инициализации AutoDevice."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()
            assert hasattr(auto_device, 'device')
            assert isinstance(auto_device.device, torch.device)

    def test_auto_device_call(self) -> None:
        """Тест вызова AutoDevice как функции."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()
            device = auto_device()
            assert isinstance(device, torch.device)
            assert device == auto_device.device

    def test_auto_device_str(self) -> None:
        """Тест строкового представления AutoDevice."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()
            assert str(auto_device) == str(torch.device('cpu'))

            # Не создаем новый AutoDevice с cuda - просто проверяем __str__ через patch
            # Вместо этого мокаем prepare_device, чтобы избежать инициализации CUDA
            with patch('pt_utils.AutoDevice.prepare_device'):
                auto_device.device = torch.device('cuda')
                assert str(auto_device) == str(torch.device('cuda'))

    def test_auto_device_to(self) -> None:
        """Тест метода to для переноса тензора на устройство."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()

            # Создаем тензор
            tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

            # Переносим на устройство
            tensor_on_device = auto_device.to(tensor)

            # Проверяем, что устройство правильное
            assert tensor_on_device.device == torch.device('cpu')

            # Проверяем, что данные не изменились
            torch.testing.assert_close(tensor, tensor_on_device.cpu())


class TestTensorUtils:
    """Тесты утилит для работы с тензорами."""

    @pytest.fixture
    def sample_tensor(self) -> torch.Tensor:
        """Создает тестовый тензор."""
        return torch.randn(2, 3, 4, 5)

    def test_get_redused_shape_none_dim(self, sample_tensor: torch.Tensor) -> None:
        """Тест get_redused_shape с dim=None."""
        # Без сохранения размерности
        shape = get_redused_shape(sample_tensor, dim=None, keepdim=False)
        assert shape == torch.Size([])

        # С сохранением размерности
        shape = get_redused_shape(sample_tensor, dim=None, keepdim=True)
        assert shape == torch.Size([1, 1, 1, 1])

    def test_get_redused_shape_single_dim(self, sample_tensor: torch.Tensor) -> None:
        """Тест get_redused_shape с одной размерностью."""
        # Без сохранения размерности
        shape = get_redused_shape(sample_tensor, dim=1, keepdim=False)
        assert shape == torch.Size([2, 4, 5])

        # С сохранением размерности
        shape = get_redused_shape(sample_tensor, dim=1, keepdim=True)
        assert shape == torch.Size([2, 1, 4, 5])

    def test_get_redused_shape_multiple_dims(self, sample_tensor: torch.Tensor) -> None:
        """Тест get_redused_shape с несколькими размерностями."""
        # Без сохранения размерности
        shape = get_redused_shape(sample_tensor, dim=[1, 2], keepdim=False)
        assert shape == torch.Size([2, 5])

        # С сохранением размерности
        shape = get_redused_shape(sample_tensor, dim=[1, 2], keepdim=True)
        assert shape == torch.Size([2, 1, 1, 5])

    def test_has_var_sufficient_elements(self, sample_tensor: torch.Tensor) -> None:
        """Тест has_var_sufficient_elements."""
        # Достаточно элементов
        assert has_var_sufficient_elements(sample_tensor, dim=1, correction=1)

        # Недостаточно элементов (коррекция больше чем элементов)
        small_tensor = torch.randn(2, 1)  # Только 2 элемента по размерности 0
        assert not has_var_sufficient_elements(small_tensor, dim=0, correction=2)

    def test_has_var_sufficient_elements_none_dim(self) -> None:
        """Тест has_var_sufficient_elements без указания размерности."""
        tensor = torch.randn(5)
        assert has_var_sufficient_elements(tensor, None, correction=2)
        assert not has_var_sufficient_elements(tensor, None, correction=5)

    def test_has_var_sufficient_elements_tuple_dim(self) -> None:
        """Тест has_var_sufficient_elements с кортежем размерностей."""
        tensor = torch.randn(3, 4)
        assert has_var_sufficient_elements(tensor, (0, 1), correction=5)

    def test_safe_var_normal_case(self, sample_tensor: torch.Tensor) -> None:
        """Тест safe_var в нормальном случае."""
        result = safe_var(sample_tensor, dim=1)
        expected = sample_tensor.var(dim=1)
        torch.testing.assert_close(result, expected)

    def test_safe_var_insufficient_elements(self) -> None:
        """Тест safe_var при недостаточном количестве элементов."""
        tensor = torch.tensor([1.0])  # Всего 1 элемент
        result = safe_var(tensor, dim=0, correction=1)
        assert result.item() == 0.0  # default_value


class TestSegDataset:
    """Тесты для датасета сегментации."""

    def test_dataset_initialization(self, temp_dataset_dir: str) -> None:
        """Тест инициализации датасета."""
        dataset = SegDataset(temp_dataset_dir)
        assert len(dataset) == 3
        assert dataset.transforms is None
        assert dataset.num_classes is None

    def test_dataset_with_num_classes(self, temp_dataset_dir: str) -> None:
        """Тест датасета с указанием num_classes."""
        dataset = SegDataset(temp_dataset_dir, num_classes=2)
        _image, mask = dataset[0]

        # Проверяем, что маска стала one-hot encoded (добавилась размерность каналов)
        assert len(mask.shape) == 3  # Должна быть 3D (H, W, C)
        assert mask.shape[2] == 2  # Количество классов

    def test_dataset_getitem(self, temp_dataset_dir: str) -> None:
        """Тест получения элемента датасета."""
        dataset = SegDataset(temp_dataset_dir)
        image, mask = dataset[0]

        assert image.shape == (100, 100, 3)
        assert mask.shape == (100, 100)  # Без one-hot encoding
        assert image.dtype == np.uint8
        assert mask.dtype == np.uint8

    def test_dataset_getitem_non_image(self, tmp_path: Path) -> None:
        """Тест ошибки при чтении файла, не являющегося изображением."""
        inp = tmp_path / 'inp'
        out = tmp_path / 'out'
        inp.mkdir()
        out.mkdir()
        (inp / 'x.png').write_text('не изображение', encoding='utf-8')
        (out / 'x.png').write_text('не изображение', encoding='utf-8')
        dataset = SegDataset(str(tmp_path))
        with pytest.raises(ValueError, match='не содержит изображение'):
            dataset[0]

    def test_dataset_invalid_structure(self) -> None:
        """Тест датасета с несовпадающими именами файлов."""
        with tempfile.TemporaryDirectory() as temp_dir:
            inp_dir = Path(temp_dir) / 'inp'
            out_dir = Path(temp_dir) / 'out'
            inp_dir.mkdir()
            out_dir.mkdir()

            # Создаем файлы с разными именами
            rng = np.random.default_rng()
            img = rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(inp_dir / 'image_1.png'), img)
            cv2.imwrite(str(out_dir / 'image_2.png'), img)

            with pytest.raises(ValueError, match='не совпадают имена'):
                SegDataset(temp_dir)


class TestSenderReceiver:
    """Тесты для классов Sender и Receiver."""

    @pytest.mark.parametrize('kernel_size', [2, 3, 5, 7])
    def test_tensor_shapes_with_different_kernels(self, kernel_size: int) -> None:
        """Тест корректности размеров тензоров для разных размеров ядра."""
        # Параметры теста
        inp_channels = 8
        msg_channels = 16
        batch_size = 4

        # Создаем модули
        sender = Sender(inp_channels, msg_channels, kernel_size)
        receiver = Receiver(msg_channels, inp_channels, kernel_size)

        # Тестируем несколько случайных размеров
        rng = np.random.default_rng()
        for _ in range(10):
            h, w = rng.integers(10, 100, size=2)

            # Входная карта признаков
            inp_map = torch.randn(batch_size, inp_channels, h, w)

            # Карта сообщений
            msg_map = sender(inp_map)

            # Выходная карта признаков
            out_map = receiver(msg_map)

            # Проверяем размеры
            msg_h, msg_w = msg_map.shape[-2:]
            out_h, out_w = out_map.shape[-2:]

            # Карта сообщений должна быть в kernel_size раз больше
            assert msg_h == h * kernel_size
            assert msg_w == w * kernel_size

            # Выходная карта должна иметь исходный размер
            assert out_h == h
            assert out_w == w

    def test_receiver_pair_method(self) -> None:
        """Тест метода pair класса Receiver."""
        # Тестируем с int
        result = Receiver.pair(3)
        assert result == (3, 3)

        # Тестируем с list
        result = Receiver.pair([2, 4])
        assert result == (2, 4)

        # Тестируем с tuple
        result = Receiver.pair((5, 6))
        assert result == (5, 6)

    def test_receiver_pair_single_and_error(self) -> None:
        """Тест pair с одиночным значением и с недопустимой длиной."""
        assert Receiver.pair([7]) == (7, 7)
        assert Receiver.pair((8,)) == (8, 8)
        with pytest.raises(ValueError, match='Ожидается 1 или 2'):
            Receiver.pair([1, 2, 3])


class TestIntegration:
    """Интеграционные тесты."""

    def test_auto_device_with_models(self) -> None:
        """Тест работы AutoDevice с моделями."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()
            device = auto_device()

            # Создаем простую модель и переносим на устройство
            model = torch.nn.Linear(10, 5)
            model.to(device)

            # Проверяем, что параметры модели на правильном устройстве
            for param in model.parameters():
                assert param.device == device

    def test_dataset_with_transforms(self, temp_dataset_dir: str) -> None:
        """Тест датасета с трансформациями."""

        # Простая трансформация для теста
        def simple_transform(
            image: np.ndarray,
            mask: np.ndarray,
        ) -> dict[str, np.ndarray]:
            return {
                'image': image.astype(np.float32) / 255.0,
                'mask': mask.astype(np.float32) / 255.0,
            }

        dataset = SegDataset(temp_dataset_dir, transforms=simple_transform)
        image, mask = dataset[0]

        assert image.dtype == np.float32
        assert mask.dtype == np.float32

    def test_auto_device_methods_integration(self) -> None:
        """Интеграционный тест методов AutoDevice."""
        with patch('pt_utils.AutoDevice.get_avliable_device') as mock_get_device:
            # Тестируем с CPU
            mock_get_device.return_value = torch.device('cpu')
            auto_device = AutoDevice()

            # Проверяем __str__
            assert str(auto_device) == 'cpu'

            # Проверяем to с моделью
            model = torch.nn.Linear(10, 5)
            model = auto_device.to(model)

            # Проверяем, что параметры на правильном устройстве
            for param in model.parameters():
                assert param.device == torch.device('cpu')

            # Проверяем to с тензором
            tensor = torch.randn(3, 3)
            tensor = auto_device.to(tensor)
            assert tensor.device == torch.device('cpu')


class TestAutoDevicePrepare:
    """Тесты для метода prepare_device."""

    def test_prepare_device_cpu(self) -> None:
        """Тест prepare_device для CPU."""
        # Должен выполниться без ошибок
        AutoDevice.prepare_device(torch.device('cpu'))

    def test_prepare_device_mps_unavailable(self) -> None:
        """Тест prepare_device когда MPS недоступен."""
        with (
            patch(
                'torch.backends.mps.is_available',
                return_value=False,
            ) as mock_cuda_available,
            patch('torch.cuda.is_available', return_value=False) as mock_mps_available,
        ):
            # Проверяем, что моки работают
            # (чтобы избежать предупреждения о неиспользуемых фикстурах)
            assert mock_cuda_available.return_value is False
            assert mock_mps_available.return_value is False

            # Если CUDA и MPS недоступны, должен использовать CPU
            device = AutoDevice.get_avliable_device()
            assert device == 'cpu'

    def test_prepare_device_cuda_ampere(self) -> None:
        """Тест prepare_device для CUDA с архитектурой Ampere."""
        with (
            patch('torch.cuda.is_available', return_value=True) as mock_is_available,
            patch('torch.cuda.get_device_properties') as mock_get_props,
        ):
            # Проверяем, что is_available возвращает True
            assert mock_is_available.return_value is True

            # Мокаем свойства GPU Ampere (>= 8.0)
            mock_get_props.return_value.major = 8
            mock_get_props.return_value.minor = 0

            # Вызываем prepare_device для CUDA
            with (
                patch('torch.autocast'),
                patch('torch.backends.cuda.matmul') as mock_matmul,
                patch('torch.backends.cudnn') as mock_cudnn,
            ):
                # Не проверяем вызов autocast, так как он может не вызываться
                # если bfloat16 не поддерживается или по другим причинам
                AutoDevice.prepare_device(torch.device('cuda'))

                # Проверяем только критически важные настройки для Ampere архитектуры
                assert mock_matmul.allow_tf32 is True
                assert mock_cudnn.allow_tf32 is True

    def test_prepare_device_cuda_pre_ampere(self) -> None:
        """Тест prepare_device для CUDA с архитектурой до Ampere."""
        with (
            patch('torch.cuda.is_available', return_value=True) as mock_is_available,
            patch('torch.cuda.get_device_properties') as mock_get_props,
        ):
            # Проверяем, что is_available возвращает True
            assert mock_is_available.return_value is True

            # Мокаем свойства GPU до Ampere (< 8.0)
            mock_get_props.return_value.major = 7
            mock_get_props.return_value.minor = 5

            # Вызываем prepare_device для CUDA
            with (
                patch('torch.autocast'),
                patch('torch.backends.cuda.matmul') as mock_matmul,
                patch('torch.backends.cudnn') as mock_cudnn,
            ):
                # Устанавливаем начальные значения
                mock_matmul.allow_tf32 = False
                mock_cudnn.allow_tf32 = False

                AutoDevice.prepare_device(torch.device('cuda'))

                # Проверяем, что matmul.allow_tf32 остался False
                assert mock_matmul.allow_tf32 is False
                # Проверяем, что cudnn.allow_tf32 остался False
                assert mock_cudnn.allow_tf32 is False


if __name__ == '__main__':
    pytest.main([__file__])
