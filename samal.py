'''
********************************************
*   Использование Segment Anything Model   *
*  (SAM) для автоматической разметки (AL). *
*                                          *
* Применяется для предварительной          *
* разметки изображений в CVAT.             *
*                                          *
*                                          *
* Основные функции и классы:               *
*   SAM - класс, инкапсулирующий Segment   *
*       Anything Model с несколькими       *
*       наиболее востребованными функциями *
*       обработки изображений на её        *
*       основе.                            *
*                                          *
********************************************
'''

import os
import cv2
import torch

import pandas as pd
import numpy  as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from urllib.request import urlretrieve

from utils import mkdirs, AnnotateIt
from cvat import CVATPoints, add_row2df
from cv_utils import Mask


class MasksFilter:
    '''
    Суперкласс для фильтров поэлементной обработки списков масок.
    Также может быть использован как декоратор для функций, обрабатывающих одну маску.
    '''
    def __init__(self, mask_filter=None):
        if mask_filter is not None:
            self.mask_filter = mask_filter
            self.__doc__ = mask_filter.__doc__
        else:
            raise ValueError('Функция "mask_filter" должна быть определена!')
    
    # Обработка всего списка масок:
    def __call__(self, masks):
        
        # Инициализация итогового списка масок:
        new_masks = []
        
        # Переборк каждой маски из старого списка:
        for mask in masks:
            
            # Применяем фильтра к маске:
            mask = self.mask_filter(mask)
            
            # Если результат является списком, кортежем или множеством, до добавляем его элементы в итоговый список:
            if isinstance(mask, (list, tuple, set)):
                new_masks += list(mask)
            # Сделано на случай если в результате фильтрации из одной маски в результате фильтрации создаётся целый список масок.
            
            # Если маска вообще возвращена, то добавляем её в итоговый список:
            elif mask is not None:
                new_masks.append(mask)
        
        return new_masks


def DropByArea(min_area=None, max_area=None):
    '''
    Создаёт функтор, выбрасывающий из списка масок те из них, чьи размеры выходят за рамки диапазона [min_area, max_area].
    '''
    # Если оба параметра остались неопределёнными, то выводим ошибку:
    if min_area is None:
        if max_area is None:
            raise ValueError('Должен быть определён хотябы один из двух порогов ("min_area" или "max_area")!')
        
        # Если определён только максимальный размер:
        else:
            
            # Объявляем функцию, отбрасывающую все сегменты больше заданного размера:
            @MasksFilter
            def drop_by_area(mask, max_area=max_area):
                '''
                Исключение сегментов, площади которых больше заданного максимума "{max_area}".
                '''
                # Переводим порог из относительного в абсолютный, если надо:
                if 0. < max_area < 1.: max_area = max_area * mask['segmentation'].size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if mask['area'] <= max_area:
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых больше заданного максимума {max_area}.'
    
    else:
        
        # Если объявлен только минимальный размер:
        if max_area is None:
            
            # Объявляем функцию, отбрасывающую все сегменты меньше заданного размера:
            @MasksFilter
            def drop_by_area(mask, min_area=min_area):
                '''
                Исключение сегментов, площади которых меньше заданного минимума "min_area".
                '''
                # Переводим порог из относительного в абсолютный, если надо:
                if 0. < min_area < 1.: min_area = min_area * mask['segmentation'].size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if min_area <= mask['area']:
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых меньше заданного минимума {min_area}.'
        
        # Если заданы оба размера:
        else:
            
            # Объявляем функцию, отбрасывающую все сегменты, размеры которых выходят за заданный диапазон:
            @MasksFilter
            def drop_by_area(mask, min_area=min_area, max_area=max_area):
                '''
                Исключение сегментов, площади которых выходят за заданный диапазон ["min_area", "max_area"].
                '''
                # Переводим пороги из относительных в абсолютные, если надо:
                if 0. < min_area < 1.: min_area = min_area * mask['segmentation'].size
                if 0. < max_area < 1.: max_area = max_area * mask['segmentation'].size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if min_area <= mask['area'] <= max_area:
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых выходят за заданный диапазон [{min_area}, {max_area}].'
    
    # Возвращаем построенный функтор:
    return drop_by_area


class DropDuplicates:
    '''
    Функтор исключения дубликатов в списках сегментов.
    
    Решение о совпадении двух сегментов принимается при
    превышении IoU (Intersection-Over-Union, он же
    индекс Жаккара) заданного порогового значения
    max_iou. Из двух сегментов остаётся бOльший, если
    не менять параметр exclude_smaller.
    '''
    def __init__(self, max_iou=0.85, exclude_smaller=True, desc=None):
        self.max_iou = max_iou
        self.exclude_smaller = exclude_smaller
        self.desc = desc
    
    def __call__(self, masks):
        # Инициализация множества индексов исключённых масок:
        drop_inds = set()
        
        # Перебираем маски в качестве первого объекта для сравнения:
        for i in tqdm(range(len(masks) - 1), desc=self.desc, disable=not self.desc):
            
            # Пропускаем все маски, чьи индексы уже исключёны:
            if i in drop_inds: continue
            
            # Получаем основные параметры маски:
            iarea =                       masks[i]['area'        ]               # Площадь
            imask =                       masks[i]['segmentation']               # Сама маска
            ibbox = CVATPoints.from_bbox(*masks[i]['bbox'        ], imask.shape) # Обрамляющий прямоугольник
            
            # Перебираем маски в качестве второго объекта для сравнения:
            for j in range(i + 1, len(masks)):
                
                # Пропускаем все маски, чьи индексы уже исключёны:
                if j in drop_inds: continue
                
                jarea =                       masks[j]['area'        ]               # Площадь
                jmask =                       masks[j]['segmentation']               # Сама маска
                jbbox = CVATPoints.from_bbox(*masks[j]['bbox'        ], jmask.shape) # Обрамляющий прямоугольник
                
                # Пропускаем, если даже обрамляющие прямоугольники сегментов пропускаются:
                if (ibbox & jbbox) is None: continue
                # Это позволяет не выполнять тяжёлую операцию подсчёта IoU в большенстве ненужных случаев.
                
                intercection = imask & jmask          # Область пересечения двух сегментов
                intercection_sum = intercection.sum() # Площать области пересечения
                
                # Пропускаем обработку, если область пересечения равна нулю:
                if intercection_sum == 0: continue
                
                union = imask | jmask   # Область объединения двух сегментов
                union_sum = union.sum() # Площать области объединения
                
                # Рассчёт IoU:
                iou = intercection_sum / union_sum
                
                # Если два сегмента почти идентичны (т.е. IoU превышает пороговое значение):
                if iou > self.max_iou:
                    
                    # Выбираем для исключения контур бОльшего или меньшего размера:
                    if self.exclude_smaller:
                        drop_ind = j if iarea > jarea else i # Исключаем меньшие сегменты
                    else:
                        drop_ind = j if iarea < jarea else i # Исключаем бОльшие сегменты
                    
                    # Вносим во множество исключений:
                    drop_inds.add(drop_ind)
                    
                    # Если исключён i-й сегмент, то выходим из вложенного цикла:
                    if drop_ind == i: break
        
        # Возвращаем список с отброшенными дубликатами:
        return [mask for ind, mask in enumerate(masks) if ind not in drop_inds]


@MasksFilter
def isolate_segments(mask):
    '''
    Разбиение отдельных сегментов в отдельные маски.
    Полезно в случае если SAM относит два отдельных
    сегмента к одному объекту (в одну маску).
    '''
    # Разбиение маски на сегменты:
    splitted_masks = Mask(mask['segmentation']).split_segments()
    
    # Если найдено не больше одного сегмента, то возвращаем список из одного элемента - оригинальной маски:
    if len(splitted_masks) < 2: return [mask]
    
    new_masks = []
    for splitted_mask in splitted_masks:
        
        # Переводим маску в бинарный формат:
        splitted_mask = splitted_mask.astype(bool)
        
        # Копируем исходную маску, чтобы изминять лишь некоторые параметры:
        m = dict(mask)
        
        # Обновляем изменившиеся параметры:
        m['segmentation'] = splitted_mask.array    # Сама маска
        m['bbox'        ] = splitted_mask.asbbox() # Обрамляющий прямоугольник (xl yt, w, h)
        m['area'        ] = splitted_mask.area()   # Площадь сегмента в пикселях
        
        # Вносим обновлённую маску в итоговый список:
        new_masks.append(m)
    
    return new_masks


class SAM:
    '''
    Класс предварительной сегментации с помощью Segment Anything Model.
    '''
    def __init__(self,
                 model_path          = '../sam_vit_h_4b8939.pth',
                 device              = 'auto'                   ,
                 use_tta             = True                     ,
                 postprocess_filters = []                       ):
        
        # Определяем, доступно ли GPU:
        if device == 'auto':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Качаем модель, если её не оказалось в указанном месте:
        if not os.path.exists(model_path):
            
            # Определяем имя и путь до файла-модели:
            file_dir, file_name = os.path.split(model_path)
            
            # Создаём папку, если её не было:
            mkdirs(file_dir)
            
            # Путь до модели в вебе:
            url = os.path.join('https://dl.fbaipublicfiles.com/segment_anything/', file_name)
            
            # Загрузка:
            with AnnotateIt(f'Загрузка модели {url} в "{file_dir}" ...',
                            f'Загрузка модели {url} в "{file_dir}" завершена!'):
                urlretrieve(url, model_path)
        
        # Загружаем модель (на GPU, если оно доступно)
        sam = sam_model_registry['default'](checkpoint=model_path).to(device=device)
        
        # Создаём метод однократной обработки изображения:
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
        # Использовать ли Test Time Augmentation вместо обычного инференса?
        self.use_tta = use_tta
        
        # Сохраняем список фильтров:
        self.postprocess_filters = postprocess_filters
    
    # Обычное применение модели:
    def inference(self, img):
        return self.mask_generator.generate(img)
    
    # Test Time Augmentation:
    def tta(self, img, leave=False, desc='Инференс для обратимо аугментированных данных'):
        '''
        Отражение по вертикали и горизонтали, а также поворот на
        90 градусов образует 8 вариантов обратимой аугментации.
        '''
        # Инициализация списка масок:
        masks = []
        
        # Оборачиваем вложенные циклы в менеджер контекста статусбара:
        with tqdm(total=8, desc=desc, disable=not desc) as pbar:
            
            # Перебор вариантов применения горизонтального отражения:
            for hflip in [True, False]:
                
                # Применяем горизонтальную аугментацию:
                aug_img_h = np.fliplr(img) if hflip else img
                
                # Перебор вариантов применения вертикального отражения:
                for vflip in [True, False]:
                    
                    # Применяем вертикальную аугментацию:
                    aug_img_hv = np.flipud(aug_img_h) if vflip else aug_img_h
                    
                    # Перебор количества поворотов на 90 градусов:
                    for k in range(2):
                        
                        # Применяем поворот на нужный угол:
                        aug_img_hvk = np.rot90(aug_img_hv, k)
                        
                        # Получаем маски для преобразованного изображения:
                        aug_masks = self.inference(aug_img_hvk)
                        
                        # Выполняем обратные преобразования над полученными масками:
                        for mask in aug_masks:
                            m    = mask['segmentation']
                            bbox = mask['bbox'        ]
                            pc   = mask['point_coords']
                            cb   = mask['crop_box'    ]
                            
                            imsize = m.shape[:2]
                            bbox =  CVATPoints.from_bbox(*bbox, imsize)
                            pc   = [CVATPoints.from_bbox( *_*2, imsize) for _ in pc]
                            cb   =  CVATPoints.from_bbox(  *cb, imsize)
                            
                            # Обращение поворота:
                            if k:
                                m    =   np.rot90(m, -k)
                                bbox = bbox.rot90(-k)
                                pc   = [  _.rot90(-k) for _ in pc]
                                cb   =   cb.rot90(-k)
                            
                            # Обращение вертиеального отражения:
                            if vflip:
                                m    =   np.flipud(m)
                                bbox = bbox.flipud()
                                pc   = [  _.flipud() for _ in pc]
                                cb   =   cb.flipud()
                            
                            # Обращение горизонтального отражения:
                            if hflip:
                                m    =   np.fliplr(m)
                                bbox = bbox.fliplr()
                                pc   = [  _.fliplr() for _ in pc]
                                cb   =   cb.fliplr()
                            
                            # Сборка исправлений в новую маску:
                            mask['segmentation'] = m
                            mask['bbox'        ] = bbox.asbbox()
                            mask['point_coords'] = [  _.asbbox()[:2] for _ in pc]
                            mask['crop_box'    ] =   cb.asbbox()
                            
                            # Внесение маски в список:
                            masks.append(mask)
                        
                        # Обновление статусбара:
                        pbar.update()
        
        return masks
    
    # Постобработка списка масок:
    def postprocess(self, masks):
        
        # Последовательное применение к списку сегментов каждого из фильтров:
        for postprocess_filter in self.postprocess_filters:
            masks = postprocess_filter(masks)
        
        return masks
    
    # Применение сети к изображению:
    def img2masks(self, img, *args, **kwargs):
        
        # Применяем разовый инференс или Test Time Augmentation:
        masks = self.tta(img, desc=None) if self.use_tta else self.inference(img)
        
        # Постобработка масок:
        return self.postprocess(masks)
    
    # Конвертация масок в датафрейм формата cvat-задач:
    @staticmethod
    def masks2df(masks, source='SAM', cup_holes=True, **kwarg):
        
        # Список датафреймов для последующего объединения:
        dfs = []
        
        # Перебор всех найденных масок:
        for mask in masks:
            
            # Векторизируем маску:
            points = CVATPoints.from_mask(mask['segmentation']).reducepoly()
            
            # Если надо исключить все внутренние контуры:
            if cup_holes:
                
                # Получаем Расщеплённые контуры:
                multipoly = points.split_multipoly()
                
                # Если контуров действительно больше одного, то оставляем наибольший:
                if len(multipoly) > 0:
                    
                    # Инициируем бОльшую площадь:
                    max_area = 0
                    
                    # Перебираем контуры:
                    for p in multipoly:
                        
                        # Если фигура является точкой, то пропускаем:
                        if len(p) == 2:
                            continue
                        # У точки нулевая площадь.
                        
                        # Подсчитываем площадь обрамляющего прямоугольника:
                        _, _, w, h = p.asbbox()
                        area = w * h
                        a
                        # Выбираем наибольший сегмент:
                        if area > max_area:
                            max_area = area
                            points = p
            # Предпологается, что сегмент всего один, и все контуры, кроме самого большого являются внутренними (дырами).
            
            # Собираем новый датафрейм из одной строки:
            df = add_row2df(None, points=points.flatten(), type=points.type, rotation=points.rotation, source=source, **kwarg)
            
            # Вносим строку в список:
            dfs.append(df)
        
        # Объединение всех строк в один датафрейм:
        df = pd.concat(dfs)
        
        # Меняем тип метки, если число точек мало для многоугольника:
        df.loc[df['points'].apply(len) == 4, 'type'] = 'polyline' # Если в многоугольнике 2 точки, то это - линия
        df.loc[df['points'].apply(len) == 2, 'type'] = 'points'   # Если в многоугольнике 1 точка, то это - точки
        
        return df
    
    # Формирование разметки в cvat-формате на основе полученного изображения:
    def img2df(self, img):
        
        # Формирование масок:
        masks = self.img2masks(img)
        
        # Перевод масок в датафрейм cvat-формата:
        return self.masks2df(masks)
    
    # Формируем разметку в формате CVAT-подзадачи по входному файлу с изображением:
    def img_file2subtask(self, img_file):
        
        # Читаем исходное изображение в формате RGB:
        img = cv2.imread(img_file)[..., ::-1]
        
        # Применяем модель для получения списка масок:
        masks = self.img2masks(img)
        
        # Переводим маски в датафрейм:
        df = self.masks2df(masks)
        
        # Возвращаем собранную подзадачу:
        return df, img_file, {0:0}
    
    # Файл с изображением -> разметка в формате CVAT-подзадачи:
    __call__ = img2df

