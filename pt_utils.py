'''
********************************************
*   Набор самописных утилит для PyTorch.   *
*                                          *
********************************************
'''

from torch.utils.data import Dataset

import numpy as np
import cv2
import os

class SegDataset(Dataset):
    '''
    Датасет для данных с сегментацией.
    '''
    def __init__(self, path, transforms=None, num_classes=None):
        
        # Определяем имена подпапок:
        source_path = os.path.join(path, 'inp') # Путь ко входным файлам
        target_path = os.path.join(path, 'out') # Путь к выходным файлам
        
        # Создаём два списка имён файлов:
        source_files = os.listdir(source_path) # Датасет имён  входных файлов
        target_files = os.listdir(target_path) # Датасет имён выходных файлов
        
        # Имена файлов должны совпадать:
        assert set(source_files) == set(target_files)
        
        # Дополняем имена путями до их папок:
        source_files = [os.path.join(source_path, file) for file in sorted(source_files)]
        target_files = [os.path.join(target_path, file) for file in sorted(target_files)]
        
        self.files = list(zip(source_files, target_files)) # Фиксируем список пар файлов вход-выход
        self.transforms  = transforms                      # Сохраняем трансформации
        self.num_classes = num_classes                     # Сохраняем число классов
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        # Открываем изображения:
        source_file, target_file = self.files[idx]
        
        # Читаем изображения:
        image = cv2.imread(source_file, cv2.IMREAD_COLOR    )[..., ::-1]
        mask  = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
        
        if self.transforms:
            image, mask = self.transforms(image=image, mask=mask).values()
        
        # Если указано число классов, то выполняем One-Hot Encoding:
        if self.num_classes:
            mask = np.eye(self.num_classes, dtype=np.float32)[mask]
        
        return image, mask
