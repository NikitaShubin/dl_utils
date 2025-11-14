'''
********************************************
*     Работа с библиотекой ultralytics.    *
*                                          *
********************************************
'''
import os

import numpy as np

from ultralytics import YOLO

from cv_utils import BBox, Mask
from cvat import obj2dfrow, concat_dfs


def _ul_boxes2boxes(boxes):
    xyxys = boxes.xyxy.numpy()
    clses = boxes.cls.numpy()
    confs = boxes.conf.numpy()
    if boxes.is_track:
        ids = boxes.id.numpy().astype(int)
    else:
        ids = [None] * len(confs)

    bboxes = []
    for xyxy, cls, conf, id in zip(xyxys, clses, confs, ids):
        bboxes.append(BBox(xyxy, attribs={'label': cls,
                                          'confidence': conf,
                                          'track_id': id}))
    return bboxes


def _ul_masks2masks(masks):
    raise NotImplementedError('Конвертация в маски не реализована!')


def _result2objs(result):
    objs = []
    if result.boxes is not None:
        objs += _ul_boxes2boxes(result.boxes)
    if result.masks is not None:
        objs += _ul_masks2masks(result.masks)
    if result.obb is not None:
        raise NotImplementedError('Повёрнутые прямоугольники не реализованы!')
    if result.keypoints is not None:
        raise NotImplementedError('Скелеты не реализованы!')

    # Заменяем номера классов каждого объекта на их имена:
    for obj in objs:
        if 'label' in obj.attribs:
            obj.attribs['label'] = result.names[obj.attribs['label']]

    return objs


class UltralyticsModel:
    '''
    Обёртка вокруг моделей от Ultralytics для инференса.

    mode:
        'preannotation' - создаётся датафрейм, поддерживаемый cvat.py;
        'preview' - Рисует на исходном изображении все обнаруженные объекты.
    '''

    def __init__(self,
                 model: str | YOLO = 'rtdetr-x.pt',
                 tracker=None,
                 mode: str = 'preannotation',
                 postprocess_filters: list = [],
                 *args, **kwargs):

        # Если передано название файла модели:
        if isinstance(model, str):

            # Eсли файл torch-модели не существует, а путь не задан явно,
            # то скачаем моель в ~/models/:
            if os.path.splitext(model)[1].lower() not in {'.pt', 'pth'} and \
                    not os.path.isfile(model) and \
                    model == os.path.basename(model):
                model = os.path.join(os.path.expanduser('~user'),
                                     'models', model)

                model = YOLO(model)

            model = YOLO(model, *args, **kwargs)

        # Если уже передана YOLO-модель - ничего не делаем:
        elif isinstance(model, YOLO):
            pass

        else:
            raise TypeError('Объект model должен быть строкой или ' +
                            f'YOLO-моделью. Получено: {type(mode)}!')

        self.model = model
        self.tracker = tracker
        self.mode = mode.lower()
        self.postprocess_filters = postprocess_filters

        self.frame_ind = 0

    def result2df(self, result):
        # Формируем список объектов:
        objs = _result2objs(result)

        # Обрабатываем список объектов, если надо:
        for postprocess_filter in self.postprocess_filters:
            objs = postprocess_filter(objs)

        # Формируем датафрейм:
        df = concat_dfs(list(map(obj2dfrow, objs)))

        # Указываем номер кадра:
        df['frame'] = df['true_frame'] = self.frame_ind
        
        return df

    def _img2result(self, img: np.ndarray):
        if self.tracker is None:
            result = self.model(img, verbose=False)[0]
        else:
            result = self.model.track(img,
                                      tracker=self.tracker,
                                      persist=True,
                                      verbose=False)[0]
        return result.cpu()

    def img2df(self, img: np.ndarray):
        '''
        Ф-ия применения модели с представлением результатов в виде датафрейма.
        '''
        df = self.result2df(self._img2result(img))
        self.frame_ind += 1
        return df

    def draw(self, img: np.ndarray):
        '''
        Создаёт превью обработанного кадра.
        '''
        return self._img2result(img).plot()

    def reset(self):
        '''
        Сбрасывает состояние трекера, если есть.
        '''
        # Сбрасываем номер кадра:
        self.frame_ind = 0
        
        # Сбрасываем фильтры, если надо:
        for postprocess_filter in self.postprocess_filters:
            if hasattr(postprocess_filter, 'reset'):
                postprocess_filter.reset()


    def __call__(self, img):
        # Определяем способ обработы в зависимости от режима:
        if self.mode == 'preannotation':
            processor = self.img2df
        elif self.mode == 'preview':
            processor = self.draw
        else:
            raise ValueError(f'Неподдерживаемый режим: {self.mode}!')

        # Применяем выборанный способ обработки:
        return processor(img)