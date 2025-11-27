'''
********************************************
*   Набор самописных утилит для PyTorch.   *
*                                          *
********************************************
'''

# if using Apple MPS, fall back to CPU for unsupported ops
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm
import numpy as np
import cv2


class AutoDevice:
    '''
    Объект, упрощающий работу с вычислительными устройствами.

    Основан на https://github.com/facebookresearch/sam2/blob/main/notebooks/
    video_predictor_example.ipynb (Он же в коллабе: colab.research.google.com/
    github/facebookresearch/sam2/blob/main/notebooks/
    video_predictor_example.ipynb)
    '''

    @staticmethod
    def get_avliable_device():
        '''
        Возвращает лучшее из доступных устройств для вычислений.
        '''
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        return device

    @staticmethod
    def prepare_device(device):
        '''
        Подготавливает Torch к использованию заданного устройства.
        '''
        if device.type == 'cuda':
            # use bfloat16 for the entire notebook
            torch.autocast('cuda', dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            # (https://pytorch.org/docs/stable/notes/cuda.
            # html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == 'mps':
            pass

    def __init__(self):
        self.device = self.get_avliable_device()
        self.prepare_device(self.device)

    def __call__(self):
        return self.device


def has_var_sufficient_elements(tensor, dim, correction):
    """Проверяет, достаточно ли элементов для вычисления дисперсии.

    Используется в safe_var.
    """
    # Для случая без размерности:
    if dim is None:
        return tensor.numel() > correction

    # Для случая с размерностью:
    else:
        # Делаем размерности кортежем:
        if isinstance(dim, int):
            dim = (dim,)

        # Оценка prod(size(dim)):
        total_elements = 1
        for d in dim:
            d = d if d >= 0 else tensor.dim() + d
            total_elements *= tensor.size(d)

        return total_elements > correction


class SegDataset(Dataset):
    '''
    Датасет для данных с сегментацией.
    '''

    def __init__(self, path, transforms=None, num_classes=None):
        # Определяем имена подпапок:
        source_path = os.path.join(path, 'inp')  # Путь ко входным файлам
        target_path = os.path.join(path, 'out')  # Путь к выходным файлам

        # Создаём два списка имён файлов:
        source_files = os.listdir(source_path)  # Датасет имён  входных файлов
        target_files = os.listdir(target_path)  # Датасет имён выходных файлов

        # Имена файлов должны совпадать:
        assert set(source_files) == set(target_files)

        # Дополняем имена путями до их папок:
        source_files = [os.path.join(source_path, file)
                        for file in sorted(source_files)]
        target_files = [os.path.join(target_path, file)
                        for file in sorted(target_files)]

        # Фиксируем список пар файлов вход-выход:
        self.files = list(zip(source_files, target_files))
        self.transforms = transforms    # Сохраняем трансформации
        self.num_classes = num_classes  # Сохраняем число классов

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # Открываем изображения:
        source_file, target_file = self.files[idx]

        # Читаем изображения:
        image = cv2.imread(source_file, cv2.IMREAD_COLOR)[..., ::-1]
        mask = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            image, mask = self.transforms(image=image, mask=mask).values()

        # Если указано число классов, то выполняем One-Hot Encoding:
        if self.num_classes:
            mask = np.eye(self.num_classes, dtype=np.float32)[mask]

        return image, mask


class Sender(nn.Module):
    '''
    Отправитель адресных сообщений.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Sender, self).__init__()
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs.pop('padding', None)
        kwargs.pop('output_padding', None)
        kwargs.pop('dilation', None)
        self.conv = nn.ConvTranspose2d(**kwargs)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


class Receiver(nn.Module):
    '''
    Получатель адресных сообщений.
    '''
    @staticmethod
    def pair(inp):
        '''
        Принудительно дублирует входную переменную если надо.
        Аналог troch.nn.modules.utils._pair.
        '''
        if isinstance(inp, (list, tuple)):
            return inp
        return (inp, inp)

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super(Receiver, self).__init__()

        kernel_h, kernel_w = self.pair(kernel_size)
        padding_h = (kernel_h ** 2 - kernel_h * 3) // 2 + 1
        padding_w = (kernel_w ** 2 - kernel_w * 3) // 2 + 1
        kwargs['in_channels'] = in_channels
        kwargs['out_channels'] = out_channels
        kwargs['kernel_size'] = kernel_size
        kwargs['stride'] = kernel_size
        kwargs['padding'] = (padding_h, padding_w)
        kwargs['dilation'] = (kernel_h - 1, kernel_w - 1)
        kwargs.pop('output_padding', None)
        self.conv = nn.Conv2d(**kwargs)

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


'''
#######################################################
# Проверка корректности размеров получаемых тензоров: #
#######################################################

inp_channels = 8
msg_channels = 16
hidden_channels = 32

# Перебираем разные размеры ядра:
for kernel_size in tqdm(range(2, 20)):

    # Отправитель сообщений:
    sender = Sender(inp_channels,
                    msg_channels,
                    kernel_size)

    # Получатель сообщений:
    receiver = Receiver(msg_channels,
                        inp_channels,
                        kernel_size)

    # 100 случайных размеров входной карты признаков:
    for _ in range(100):

        # Случайный размер входной карты признаков:
        h, w = np.random.randint(1, 128, size=2)

        # Входная карта признаков:
        inp_map = torch.randn((8, inp_channels, h, w))

        # Карта сообщений:
        msg_map = sender(inp_map)

        # Карта признаков после обмена сообщениями:
        out_map = receiver(msg_map)

        # Карта сообщений должна быть в kernel_size раз больше исходной по
        # каждой из осей:
        msg_h, msg_w = msg_map.shape[-2:]
        assert h * kernel_size == msg_h
        assert w * kernel_size == msg_w

        # Карта признаков после обмена сообщениями должна быть соразмерна
        # входной:
        hid_h, hid_w = out_map.shape[-2:]
        assert hid_h == h
        assert hid_w == w
'''