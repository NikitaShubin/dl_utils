"""pt_utils.py.

********************************************
*   Набор самописных утилит для PyTorch.   *
*                                          *
********************************************
.
"""

# if using Apple MPS, fall back to CPU for unsupported ops
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class AutoDevice:
    """Объект, упрощающий работу с вычислительными устройствами.

    Основан на https://github.com/facebookresearch/sam2/blob/main/notebooks/
    video_predictor_example.ipynb (Он же в коллабе: colab.research.google.com/
    github/facebookresearch/sam2/blob/main/notebooks/
    video_predictor_example.ipynb)
    """

    @staticmethod
    def get_avliable_device() -> torch.device:
        """Возвращает лучшее из доступных устройств для вычислений."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        return device

    @staticmethod
    def prepare_device(device: torch.device) -> None:
        """Подготавливает Torch к использованию заданного устройства."""
        if device.type == 'cuda':
            # use bfloat16 for the entire notebook:
            torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

            # Определяем константу для минимальной архитектуры Ampere:
            min_ampere_arch = (8, 0)  # (major, minor)
            """
            turn on tfloat32 for Ampere GPUs https://pytorch.org/docs/stable/
            /notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            """
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(0) >= min_ampere_arch
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == 'mps':
            pass

    def __init__(self) -> None:
        """Инициализирует AutoDevice с лучшим доступным устройством."""
        self.device = self.get_avliable_device()
        self.prepare_device(self.device)

    def __call__(self) -> torch.device:
        """Возвращает текущее вычислительное устройство."""
        return self.device


def get_redused_shape(
    tensor: torch.Tensor,
    dim: None | int | list | tuple,
    keepdim: bool = False,  # noqa: FBT001, FBT002
) -> torch.Size:
    """Получает форму выходного тензора после операции reduction."""
    if dim is None:
        # Возвращаем форму с единичными размерностями:
        if keepdim:
            return torch.Size([1] * tensor.dim())

        # Скалярный результат:
        return torch.Size([])

    # Приводим к кортежу и Нормализуем индексы размерностей:
    if isinstance(dim, int):
        dim = (dim,)
    dim = tuple(d if d >= 0 else tensor.dim() + d for d in dim)

    # Заменяем указанные размерности на 1:
    if keepdim:
        return torch.Size(1 if i in dim else s for i, s in enumerate(tensor.shape))

    # Убираем указанные размерности:
    return torch.Size(s for i, s in enumerate(tensor.shape) if i not in dim)


def has_var_sufficient_elements(
    tensor: torch.Tensor, dim: None | int | list | tuple, correction: int
) -> bool:
    """Проверяет, достаточно ли элементов для вычисления дисперсии.

    Используется в safe_var.
    """
    # Для случая без размерности:
    if dim is None:
        return tensor.numel() > correction

    # Для случая с размерностью:
    # Делаем размерности кортежем:
    if isinstance(dim, int):
        dim = (dim,)

    # Оценка prod(size(dim)):
    normalized_dims = [d if d >= 0 else tensor.dim() + d for d in dim]
    total_elements = np.prod([tensor.size(d) for d in normalized_dims])

    return bool(total_elements > correction)


def safe_var(
    tensor: torch.Tensor,
    dim: None | int | list | tuple = None,
    keepdim: bool = False,  # noqa: FBT001, FBT002
    correction: int = 1,
    default_value: float = 0.0,
) -> torch.Tensor:
    """Безопасное вычисление дисперсии с проверкой достаточности количества элементов.

    Аргументы:
        tensor: входной тензор;
        dim: размерность или кортеж размерностей для вычисления дисперсии;
        keepdim: сохранять ли размерность;
        correction: поправка на степени свободы (аналог ddof);
        default_value: значение по умолчанию при недостаточном количестве элементов.

    Результат:
        Тензор с дисперсией или default_value.
    """
    # Элементов достаточно - вычисляем дисперсию:
    if has_var_sufficient_elements(tensor, dim, correction):
        return tensor.var(dim=dim, keepdim=keepdim, correction=correction)

    # Элементов недостаточно - возвращаем значение по умолчанию:

    # Оцениваем размер итогового тензора:
    redused_shape = get_redused_shape(tensor, dim=dim, keepdim=keepdim)

    # Формируем итоговый тензор с нужными значениями:
    return torch.full(
        redused_shape, default_value, device=tensor.device, dtype=tensor.dtype
    )


class SegDataset(Dataset):
    """Датасет для данных с сегментации."""

    def __init__(
        self,
        path: str,
        transforms: Callable | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Инициализирует датасет для сегментации.

        Аргументы:
            path: Путь к папке с данными;
            transforms: Трансформации для аугментации данных;
            num_classes: Количество классов для one-hot кодирования.
        """
        # Определяем имена подпапок:
        source_path = Path(path) / 'inp'  # Путь ко входным файлам
        target_path = Path(path) / 'out'  # Путь к выходным файлам

        # Создаём два списка файлов для входных и выходных файлов соответственно:
        source_files = [f for f in sorted(source_path.iterdir()) if f.is_file()]
        target_files = [f for f in sorted(target_path.iterdir()) if f.is_file()]

        # Фиксируем список пар файлов вход-выход:
        self.files = []
        for source_file, target_file in zip(source_files, target_files, strict=False):
            source_stem = source_file.stem
            target_stem = target_file.stem
            if source_stem != target_stem:
                msg = f'У файлов "{source_file}" и "{target_file}" не совпадают имена!'
                raise ValueError(msg)

            self.files.append((source_file, target_file))

        self.transforms = transforms  # Сохраняем трансформации
        self.num_classes = num_classes  # Сохраняем число классов

    def __len__(self) -> int:
        """Возвращает количество элементов в датасете."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает элемент датасета по индексу.

        Аргументы:
            idx: Индекс элемента.

        Результат:
            Кортеж (изображение, маска).
        """
        # Открываем изображения:
        source_file, target_file = self.files[idx]

        # Читаем изображения:
        image = cv2.imread(str(source_file), cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imread(str(target_file), cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Если указано число классов, то выполняем One-Hot Encoding:
        if self.num_classes:
            # Нормализуем маску до значений [0, num_classes-1]
            mask_normalized = mask // 255  # Конвертируем 255 в 1
            mask = np.eye(self.num_classes, dtype=np.float32)[mask_normalized]

        return image, mask


class Sender(nn.Module):
    """Отправитель адресных сообщений."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        """Инициализирует отправитель сообщений.

        Аргументы:
            in_channels: Количество входных каналов;
            out_channels: Количество выходных каналов;
            kernel_size: Размер ядра;
            **kwargs: Дополнительные аргументы для ConvTranspose2d.
        """
        super().__init__()
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs.pop('padding', None)
        kwargs.pop('output_padding', None)
        kwargs.pop('dilation', None)
        self.conv = nn.ConvTranspose2d(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через слой."""
        return self.conv(x)


class Receiver(nn.Module):
    """Получатель адресных сообщений."""

    @staticmethod
    def pair(inp: int | list[int] | tuple[int, ...]) -> tuple[int, int]:
        """Принудительно дублирует входную переменную если надо.

        Аналог troch.nn.modules.utils._pair.

        Аргументы:
            inp: Входное значение.

        Результат:
            Кортеж из двух одинаковых значений.
        """
        if isinstance(inp, (list, tuple)):
            # Определяем константы для лучшей читаемости:
            single_length = 1
            pair_length = 2

            if len(inp) == single_length:
                inp = inp[0]
                return inp, inp
            if len(inp) == pair_length:
                return inp[0], inp[1]
            msg = f'Ожидается 1 или 2 значения, получено {len(inp)}'
            raise ValueError(msg)
        return inp, inp

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        **kwargs: object,
    ) -> None:
        """Инициализирует получатель сообщений.

        Аргументы:
            in_channels: Количество входных каналов;
            out_channels: Количество выходных каналов;
            kernel_size: Размер ядра;
            **kwargs: Дополнительные аргументы для Conv2d.
        """
        super().__init__()

        kernel_h, kernel_w = self.pair(kernel_size)
        padding_h = (kernel_h**2 - kernel_h * 3) // 2 + 1
        padding_w = (kernel_w**2 - kernel_w * 3) // 2 + 1
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs['padding'] = (padding_h, padding_w)
        kwargs['dilation'] = (kernel_h - 1, kernel_w - 1)
        kwargs.pop('output_padding', None)
        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через слой."""
        return self.conv(x)
