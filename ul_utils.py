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
from cvat import concat_dfs, CVATPoints


def _decrop_ul_masks(masks, orig_shape):
    '''
    Удаляет рамку у масок ultralytics.
    '''
    # Определяем исходные размеры масок и изображения:
    orig_shape = np.array(orig_shape[:2])
    mask_shape = np.array(masks.shape[1:])

    # Оцениваем размер масок после обрезки:
    target_shape = orig_shape * (mask_shape / orig_shape).min()
    target_shape = np.round(target_shape, decimals=1).astype(int)

    pad = mask_shape - target_shape  # Размер рамок

    # Параметры обрезки:
    top_left = np.round(pad / 2, decimals=1).astype(int)
    # top_left = np.zeros_like(pad)
    bottom_right = target_shape + top_left + 1

    # Обрезка:
    return masks[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


def _result2objs(result, attribs: dict = {}):
    '''
    Переводим результаты из YOLO-формата в список объектов,
    поддерживаемый cv_utils.
    '''
    # Фиксируем грубые описания объектов:
    xyxys = result.boxes.xyxy.numpy()
    clses = result.boxes.cls.numpy()
    confs = result.boxes.conf.numpy()
    if result.boxes.is_track:
        ids = result.boxes.id.numpy().astype(int)
    else:
        ids = [None] * len(confs)

    # Строим сами объекты:
    objs = []
    if result.masks is not None:

        # Удаление рамок у масок:
        masks = result.masks.data.numpy() * 255  # [0, 1] -> [0, 255] (uint8)
        masks = _decrop_ul_masks(masks, result.orig_shape)

        # Размер масок может отличаться от размеров исходного изображения,
        # так что для прямоугольников нужно масштабирование:
        scale_y = masks.shape[1] / result.orig_shape[0]
        scale_x = masks.shape[2] / result.orig_shape[1]
        scale = np.array([scale_x, scale_y, scale_x, scale_y])

        for mask, xyxy, cls, conf, id in zip(masks, xyxys * scale,
                                             clses, confs, ids):
            attribs_ = attribs | {'label': result.names[cls],
                                  'confidence': conf,
                                  'track_id': id}
            objs.append(Mask(mask, rect=xyxy, attribs=attribs_))

    elif result.obb is not None:
        raise NotImplementedError('Повёрнутые прямоугольники не реализованы!')

    elif result.keypoints is not None:
        raise NotImplementedError('Скелеты не реализованы!')

    # Если есть только обычные обрамляющие прямоугольники:
    else:
        for xyxy, cls, conf, id in zip(xyxys, clses, confs, ids):
            attribs_ = attribs | {'label': result.names[cls],
                                  'confidence': conf,
                                  'track_id': id}
            objs.append(BBox(xyxy, imsize=result.orig_shape,
                             attribs=attribs_))

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
            if os.path.splitext(model)[1].lower() in {'.pt', 'pth'} and \
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

        # Сбрасываем все внутренние состояния:
        self.reset()

    def result2df(self, result):
        # Формируем список объектов:
        attribs = {'frame': self.frame_ind, 'true_frame': self.frame_ind}
        objs = _result2objs(result, attribs)

        # Обрабатываем список объектов, если надо:
        for postprocess_filter in self.postprocess_filters:
            objs = postprocess_filter(objs)

        # Формируем датафрейм:
        dfs = []
        for obj in objs:

            if isinstance(obj, BBox):
                points = CVATPoints.from_bbox(obj)

            elif isinstance(obj, Mask):
                points = CVATPoints.from_mask(obj)

                # Масштабируем маску до размера исходного изображения,
                # если надо:
                scale_y = result.orig_shape[0] / obj.array.shape[0]
                scale_x = result.orig_shape[1] / obj.array.shape[1]
                if not (scale_y == scale_x == 1.):
                    points = points.scale((scale_x, scale_y))

            else:
                raise TypeError(f'Неподдерживаемый тип: {type(obj)}!')

            dfs.append(points.to_dfrow())

        return concat_dfs(dfs)

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

        # Сбрасываем состояния трекеров, если надо:
        predictor = self.model.predictor
        if hasattr(predictor, 'trackers'):
            for tracker in predictor.trackers:
                tracker.reset()

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