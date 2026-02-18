#!/usr/bin/env python3
"""ul_utils.py.

********************************************
*     Работа с библиотекой ultralytics.    *
*                                          *
*   Предоставляет обёртку для инференса    *
* моделей детекции/сегментации из          *
* фреймворка Ultralytics с конвертацией    *
* результатов в формат, совместимый с      *
* cv_utils и cvat.                         *
*                                          *
* Основные компоненты:                     *
*     UltralyticsModel - основной класс    *
*         для инференса моделей YOLO с     *
*         поддержкой трекинга,             *
*         постобработки и разных режимов   *
*         вывода.                          *
*                                          *
********************************************
.
"""

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import ultralytics
from tqdm import tqdm
from ultralytics import YOLO, engine

from cv_utils import BBox, Mask
from cvat import CVATPoints, Subtask, concat_dfs
from video_utils import VideoGenerator

# Класс моделей:
ULModel: TypeAlias = (
    ultralytics.engine.model.Model | ultralytics.engine.predictor.BasePredictor
)


def _unpad_ul_masks(masks: np.ndarray, orig_shape: tuple[int, ...]) -> np.ndarray:
    """Удаляет паддинг (рамку) из масок, сгенерированных Ultralytics.

    Ultralytics при сегментации создаёт маски фиксированного размера с
    паддингом вокруг объекта. Эта функция обрезает паддинг, приводя маску
    к пропорциональному масштабу исходного изображения.

    Args:
        masks: Тензор масок shape (N, H, W) с паддингом
        orig_shape: Размеры исходного изображения (height, width, ...)

    Returns:
        Тензор масок без паддинга shape (N, H', W') где H' и W' пропорциональны
        оригинальным размерам

    """
    # Определяем исходные размеры масок и изображения:
    orig_shape = np.array(orig_shape[:2])
    mask_shape = np.array(masks.shape[1:])

    # Оцениваем размер масок после обрезки:
    target_shape = orig_shape * (mask_shape / orig_shape).min()
    target_shape = np.round(target_shape, decimals=1).astype(int)

    pad = mask_shape - target_shape  # Размер рамок

    # Параметры обрезки:
    top_left = np.round(pad / 2, decimals=1).astype(int)
    bottom_right = target_shape + top_left + 1

    # Обрезка:
    return masks[:, top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]


def _result2objs(
    result: engine.results,
    attribs: dict | None = None,
) -> list[BBox | Mask]:
    """Конвертирует результат работы YOLO модели в список объектов cv_utils.

    Поддерживает различные типы выходов модели:
        - боксы (bounding boxes)
        - маски сегментации
        - повёрнутые боксы (не реализовано)
        - скелеты (не реализовано)

    Args:
        result: Результат работы YOLO модели (объект ultralytics.engine.results)
        attribs: Дополнительные атрибуты для добавляемые к каждому объекту

    Returns:
        Список объектов BBox или Mask с атрибутами из модели

    Raises:
        NotImplementedError: Для неподдерживаемых типов вывода (OBB, keypoints)

    """
    # Фиксируем грубые описания объектов:
    if attribs is None:
        attribs = {}
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
        masks = (result.masks.data.numpy() * 255).astype(np.uint8)  # bool -> uint8
        masks = _unpad_ul_masks(masks, result.orig_shape)

        # Размер масок может отличаться от размеров исходного изображения,
        # так что для прямоугольников нужно масштабирование:
        scale_y = masks.shape[1] / result.orig_shape[0]
        scale_x = masks.shape[2] / result.orig_shape[1]
        scale = np.array([scale_x, scale_y, scale_x, scale_y])

        for mask, xyxy, cls, conf, id_ in zip(
            masks,
            xyxys * scale,
            clses,
            confs,
            ids,
            strict=False,
        ):
            attribs_ = attribs | {
                'label': result.names[int(cls)],
                'confidence': conf,
                'track_id': id_,
            }
            objs.append(Mask(mask, rect=xyxy, attribs=attribs_))

    elif result.obb is not None:
        msg = 'Повёрнутые прямоугольники не реализованы!'
        raise NotImplementedError(msg)

    elif result.keypoints is not None:
        msg = 'Скелеты не реализованы!'
        raise NotImplementedError(msg)

    # Если есть только обычные обрамляющие прямоугольники:
    else:
        for xyxy, cls, conf, id_ in zip(xyxys, clses, confs, ids, strict=False):
            attribs_ = attribs | {
                'label': result.names[int(cls)],
                'confidence': conf,
                'track_id': id_,
            }
            objs.append(BBox(xyxy, imsize=result.orig_shape, attribs=attribs_))

    return objs


# Минимально и максимально допустимые числоа агрументов для postprocess_filter:
_one_filter_arg = 1
_two_filter_args = 2


class UltralyticsModel:
    """Обёртка для моделей Ultralytics (YOLO) с поддержкой CVAT-формата.

    Инкапсулирует работу с моделями детекции/сегментации Ultralytics,
    предоставляет единый интерфейс для инференса с конвертацией результатов
    в формат, совместимый с cv_utils и cvat модулями.

    Поддерживает:
        - автоматическую загрузку моделей в ~/models/
        - трекинг объектов
        - различные режимы вывода (разметка, превью)
        - цепочки постобработки

    Attributes:
        model: Загруженная YOLO модель
        tracker: Название трекера (если используется)
        mode: Режим работы ('preannotation' или 'preview')
        postprocess_filters: Список фильтров постобработки
        frame_ind: Текущий номер кадра (для трекинга)

    Примеры:
        >>> model = UltralyticsModel('yolo11n.pt', mode='preannotation')
        >>> df = model.img2df(image)  # Датафрейм разметки
        >>> preview = model.draw(image)  # Изображение с визуализацией

    """

    def __init__(
        self,
        model: str | Path | ULModel = 'rtdetr-x.pt',
        tracker: str | None = None,
        mode: str = 'preannotation',
        postprocess_filters: list[Callable] | None = None,
        **kwargs: object,
    ) -> None:
        """Инициализирует Ultralytics модель.

        Args:
            model:
                Путь к файлу модели (.pt/.pth), имя модели для загрузки
                из ~/models/ или готовый объект YOLO.

            tracker:
                Название трекера (например, 'bytetrack.yaml') или None.

            mode:
                Режим работы:
                    'preannotation' - возвращает датафрейм разметки;
                    'preview' - возвращает изображение с визуализацией.

            postprocess_filters:
                Список функций для постобработки объектов
                (например, [NMS(), Tracker()])

            **kwargs:
                Дополнительные именованные аргументы для:
                    - конструктора модели, если задан лишь путь до неё;
                    - для инференса модели, если передан экземпляр класса
                    ultralytics.engine.model.Model.

        Raises:
            TypeError: Если передан неверный тип model

        """
        # Отстутствие фильтров соответствует пустому списку:
        if postprocess_filters is None:
            postprocess_filters = []

        # Если передано название файла модели:
        if isinstance(model, (str, Path)):
            model_path = Path(model)

            # Если файл torch-модели не существует, а путь не задан явно,
            # то скачаем модель в ~/models/:
            if (
                model_path.suffix.lower() in {'.pt', '.pth'}
                and not model_path.is_file()
                and str(model) == model_path.name
            ):
                models_dir = Path.home() / 'models'  # Путь к директории ~/models/
                model_path = models_dir / model_path.name

            # Загружаем YOLO-модель:
            self.model = YOLO(str(model_path), **kwargs)

            # Фиксируем отсутствия доп. параметров для инференса:
            self.kwargs: dict[str, object] = {}

        # Если передана уже загруженная YOLO-модель - используем её:
        elif isinstance(model, ULModel):
            self.model = model

            # Фиксируем доп. параметры для инференса:
            self.kwargs = kwargs

        else:
            msg = (
                'Объект model должен быть строкой или '
                f'зкземпляром класса {ULModel}. Получен объект типа: {type(model)}!'
            )
            raise TypeError(msg)

        self.tracker = tracker
        self.mode = mode.lower()
        self.postprocess_filters = postprocess_filters

        # Сбрасываем все внутренние состояния:
        self.reset()

    def result2df(self, result: engine.results) -> pd.DataFrame | None:
        """Конвертирует результат работы YOLO в датафрейм CVAT-формата.

        Args:
            result: Результат работы YOLO модели (после вызова model())

        Returns:
            pd.DataFrame | None: Датафрейм с колонками как в df_columns_type из cvat

        Note:
            Для масок автоматически применяется масштабирование к исходному
            размеру изображения

        """
        # Формируем список объектов:
        attribs = {'frame': self.frame_ind, 'true_frame': self.frame_ind}
        objs = _result2objs(result, attribs)

        # Обрабатываем список объектов, если надо:
        for postprocess_filter in self.postprocess_filters:
            nargs = len(inspect.signature(postprocess_filter).parameters)
            if nargs == _one_filter_arg:
                objs = postprocess_filter(objs)
            elif nargs == _two_filter_args:
                objs = postprocess_filter(objs, result.orig_img)
            else:
                msg = (
                    'Фильтр постобработки должен принимать аргументы (objs) или '
                    f'(objs, img), но {postprocess_filter} имеет {nargs} '
                    'аргументов!'
                )
                raise ValueError(msg)

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
                if not (scale_y == scale_x == 1.0):
                    points = points.scale((scale_x, scale_y))

            else:
                msg = f'Неподдерживаемый тип: {type(obj)}!'
                raise TypeError(msg)

            # Отмечаем объект только если он имеется:
            if points is not None and len(points):
                dfs.append(points.to_dfrow())

        return concat_dfs(dfs)

    def _img2result(self, img: np.ndarray) -> engine.results:
        """Обрабатывает изображение и возвращает разметку во внутреннем формате."""
        if self.tracker is None:
            result = self.model(img, verbose=False, **self.kwargs)[0]
        else:
            result = self.model.track(
                img,
                tracker=self.tracker,
                persist=True,
                verbose=False,
                **self.kwargs,
            )[0]
        return result.cpu()

    def img2df(self, img: np.ndarray) -> pd.DataFrame | None:
        """Обрабатывает изображение и возвращает разметку в виде датафрейма.

        Args:
            img: Входное изображение

        Returns:
            pd.DataFrame | None: Датафрейм с обнаруженными объектами в формате CVAT

        """
        df = self.result2df(self._img2result(img))
        self.frame_ind += 1
        return df

    def draw(self, img: np.ndarray) -> np.ndarray:
        """Создаёт превью обработанного кадра."""
        return self._img2result(img).plot()

    def reset(self) -> None:
        """Сбрасывает состояние трекера, если есть."""
        # Сбрасываем номер кадра:
        self.frame_ind = 0

        # Сбрасываем фильтры, если надо:
        for postprocess_filter in self.postprocess_filters:
            if hasattr(postprocess_filter, 'reset'):
                postprocess_filter.reset()

        # Сбрасываем состояния трекеров, если надо:
        predictor = self.model
        if hasattr(predictor, 'predictor'):
            predictor = predictor.predictor
        if hasattr(predictor, 'trackers'):
            for tracker in predictor.trackers:
                tracker.reset()

    def __call__(self, img: np.ndarray) -> pd.DataFrame | None | np.ndarray:
        """Обрабатывает изображение в соответствии с установленным режимом.

        Args:
            img: Входное изображение

        Returns:
            Зависит от mode:
                'preannotation' -> pd.DataFrame | None
                'preview' -> np.ndarray (изображение)

        Raises:
            ValueError: При неподдерживаемом режиме работы

        """
        # Применяем способ обработки в зависимости от режима:
        if self.mode == 'preannotation':
            return self.img2df(img)
        if self.mode == 'preview':
            return self.draw(img)

        msg = f'Неподдерживаемый режим: {self.mode}!'
        raise ValueError(msg)

    def video2subtask(self, file: str | Path, desc: str | None = None) -> Subtask:
        """Формирует CVAT-подзадачу с объектами в заданном видеофайле."""
        # Сбрасываем внутренние состояния:
        self.reset()

        # Создаём генератор, формирующий результаты обработки кадров:
        results_generator = self.model(
            source=file,
            stream=True,
            **self.kwargs,
        )

        # Оцениваем общее число кадров:
        total = len(VideoGenerator(file))

        # Оборачиваем генератор в tqdm, если указан сопроводительный текст:
        if desc:
            results_generator = tqdm(results_generator, desc, total)

        # Выполняем обработку:
        dfs = []
        results_iter = iter(results_generator)
        while True:
            # Обрабатываем следующий кадр, если есть:
            try:
                result = next(results_iter)
            except (IndexError, StopIteration):
                break

            # Преобразовываем результат обработки в датафйермф и вносим его в список:
            df = self.result2df(result.cpu())
            dfs.append(df)
            self.frame_ind += 1

        # Возвращаем подзадачу со всеми найденными объектами:
        return concat_dfs(dfs), file, {frame: frame for frame in range(self.frame_ind)}


# При автономном запуске закачиваем ряд моделей в "~/models/":
if __name__ == '__main__':
    models = {
        'yolo26x.pt',
        'yolo12x.pt',
        'rtdetr-x.pt',
        'yolo26x-seg.pt',
        'yolo11x-seg.pt',
    }
    for model in models:
        UltralyticsModel(model)
