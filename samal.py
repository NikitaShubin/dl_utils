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

import pandas as pd
import numpy as np
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from urllib.request import urlretrieve

from utils import mkdirs, AnnotateIt
from cvat import CVATPoints, add_row2df
from cv_utils import Mask
from pt_utils import AutoDevice


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
            device = AutoDevice()

        # Если передан только один путь, то делаем из него список:
        if isinstance(model_path, str):
            model_paths = [model_path]
        else:
            model_paths = model_path

        # Инициируем список генераторов масок?
        self.mask_generators = []

        for model_path in tqdm(model_paths,
                               desc='Загружаем модели' if \
                               len(model_paths) > 1 else None):

            # Качаем модель, если её не оказалось в указанном месте:
            if not os.path.exists(model_path):

                # Определяем имя и путь до файла-модели:
                file_dir, file_name = os.path.split(model_path)

                # Создаём папку, если её не было:
                mkdirs(file_dir)

                # Путь до модели в вебе:
                url = os.path.join(
                    'https://dl.fbaipublicfiles.com/segment_anything/',
                    file_name)

                # Загрузка:
                prefix = f'Загрузка модели {url} в "{file_dir}" '
                with AnnotateIt(prefix + '...', prefix + 'завершена!'):
                    urlretrieve(url, model_path)

            # Определяем тип модели по имени файла:
            model_types = []
            for model_type in ['vit_h', 'vit_l', 'vit_b', 'vit_tiny']:
                if model_type in model_path.lower():
                    model_types.append(model_type)
            if len(model_types) == 1:
                model_type = model_types.pop()
            else:
                raise ValueError('Не удалось определить тип модели по ' +
                                 f'имени "{model_path}"!')

            # Загружаем модель (на GPU, если оно доступно)
            sam = sam_model_registry[model_type](checkpoint=model_path).to(
                device=device)

            # Создаём метод однократной обработки изображения:
            self.mask_generators.append(SamAutomaticMaskGenerator(sam))

        # Использовать ли Test Time Augmentation вместо обычного инференса?
        self.use_tta = use_tta

        # Сохраняем список фильтров:
        self.postprocess_filters = postprocess_filters

    # Обычное применение моделей:
    def inference(self, img):
        masks = []
        for mask_generator in self.mask_generators:
            masks += mask_generator.generate(img)
        return masks

    # Test Time Augmentation:
    def tta(self, img,
            desc='Инференс для обратимо аугментированных данных'):
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

                        # Выполняем обратные преобразования над полученными
                        # масками:
                        for mask in aug_masks:
                            m    = mask['segmentation']
                            bbox = mask['bbox'        ]
                            pc   = mask['point_coords']
                            cb   = mask['crop_box'    ]

                            imsize = m.shape[:2]
                            bbox =  CVATPoints.from_bbox(bbox, imsize)
                            pc   = [CVATPoints.from_bbox( _*2, imsize) for _ in pc]
                            cb   =  CVATPoints.from_bbox(  cb, imsize)

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
        if len(self.postprocess_filters):

            # Оборачиваем маски COCO-формата из SAM в экземпляры класса Mask:
            masks = list(map(Mask.from_COCO_annotation, masks))

            # Выполняем последовательное приминение ф-ий постобработки:
            for postprocess_filter in self.postprocess_filters:
                masks = postprocess_filter(masks)

            # Возвращение из Mask в COCO:
            masks = [mask.as_COCO_annotation() for mask in masks]

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
            #points = points.re
            #m = points.split_multipoly()
            '''
            for _ in m:
                if isinstance(_, list):
                    print(_)
                    raise

                try:
                    m.reducepoly()
                    print('ok')
                except:
                    print('[', end='')
                    for __ in points.flatten():
                        print()
                    print(points.points)
                    points.show()
                    raise
            try:
                points = points.reducepoly()
            except:
                from PIL.Image import fromarray
                fromarray(mask['segmentation'])
                raise
            '''
            # Если надо исключить все внутренние контуры:
            if cup_holes:

                # Получаем Расщеплённые контуры:
                multipoly = points.split_multipoly()

                # Если контуров действительно больше одного, то оставляем
                # наибольший:
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

                        # Выбираем наибольший сегмент:
                        if area > max_area:
                            max_area = area
                            points = p
            # Предпологается, что сегмент всего один, и все контуры, кроме
            # самого большого являются внутренними (дырами).

            # Собираем новый датафрейм из одной строки:
            df = add_row2df(None, points=points.flatten(), type=points.type,
                            rotation=points.rotation, source=source, **kwarg)

            # Вносим строку в список:
            dfs.append(df)

        # Объединение всех строк в один датафрейм:
        df = pd.concat(dfs)

        # Меняем тип метки, если число точек мало для многоугольника:

        # Если в многоугольнике 2 точки, то это - линия:
        df.loc[df['points'].apply(len) == 4, 'type'] = 'polyline'
        # Если в многоугольнике 1 точка, то это - точки:
        df.loc[df['points'].apply(len) == 2, 'type'] = 'points'

        # SAM не различает объекты:
        df['label'] = kwarg.get('label', None)
        # Вынесено вне add_row2df, чтобы None не переводился в строку.
        # Иначе потом task_auto_annottation её не заменит.

        return df

    # Формирование разметки в cvat-формате на основе полученного изображения:
    def img2df(self, img):

        # Формирование масок:
        masks = self.img2masks(img)

        # Перевод масок в датафрейм cvat-формата:
        return self.masks2df(masks)

    # Формируем разметку в формате CVAT-подзадачи по входному файлу с
    # изображением:
    def img_file2subtask(self, img_file):

        # Читаем исходное изображение в формате RGB:
        img = cv2.imread(img_file)[..., ::-1]

        # Применяем модель для получения списка масок:
        masks = self.img2masks(img)

        # Переводим маски в датафрейм:
        df = self.masks2df(masks)

        # Возвращаем собранную подзадачу:
        return df, img_file, {0: 0}

    # Файл с изображением -> разметка в формате CVAT-подзадачи:
    __call__ = img2df

    def reset(self):
        '''
        Сброс внутренних состояний.
        '''
        [_.reset for _ in self.postprocess_filters if hasattr(_, 'reset')]