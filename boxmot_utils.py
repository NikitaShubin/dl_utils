#!/usr/bin/env python3
"""boxmot_utils.py.

********************************************
*        Работа с библиотекой boxmot.      *
*                                          *
********************************************
.
"""

import inspect
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import boxmot  # type: ignore[import-untyped]
import cv2
import numpy as np
from loguru import logger

from cv_utils import BBox, Mask


@contextmanager
def suppress_module_logs(module_name: str = 'boxmot') -> Iterator[None]:
    """Контекст, подавляющий логи модуля."""
    logger.disable(module_name)
    try:
        yield
    finally:
        logger.enable(module_name)


class Tracker:
    """Обвязка вокруг трекера из boxmot для работы с CVAT-данными.

    Класс предоставляет удобный интерфейс для работы с трекерами из библиотеки
    boxmot, адаптированный для обработки объектов в формате CVAT (BBox, Mask).

    Пример использования:
    ```
    tracker = Tracker(tracker_type='botsort')
    for frame_idx, (frame, objects) in enumerate(dataset):
        tracked_objects = tracker.update(objects, frame)
        # Каждому объекту добавлен атрибут 'track_id'
    ```
    """

    # Словарь перехода от имён поддерживаемых трекеров к их конструкторам:
    trackers: ClassVar[dict] = {}
    for name, constructor_name in boxmot.trackers.tracker_zoo.TRACKER_MAPPING.items():
        trackers[name] = eval(constructor_name)  # noqa: S307

    # Трекеры, требующие модель ReID для работы:
    reid_trackers: ClassVar[list[str]] = boxmot.trackers.tracker_zoo.REID_TRACKERS

    # Параметры ReID по-умолчанию:
    default_reid_kwargs: ClassVar[dict] = {
        'reid_weights': Path('~/models/osnet_x0_25_msmt17.pt').expanduser(),
        'device': None,
        'half': False,
    }

    def __init__(
        self,
        tracker_type: str = 'ocsort',
        *,
        store_untracked: bool = False,
        **tracker_kwargs: dict[str, int | float | str | Path | None],
    ) -> None:
        """Инициализирует трекер.

        Параметры:
        ----------
        """
        # Строки документации будут дополнены вне функции

        # Фиксируем параметры:
        self.tracker_type = tracker_type
        self.store_untracked = store_untracked  # Флаг сохранения объектов без треков
        self.tracker_kwargs = self.default_reid_kwargs | tracker_kwargs

        # Установка самого трекера через сброс:
        self.reset()

    # Инициируем одполнительые строки с информацией из самого boxmot:
    add_doc = f"""
        tracker_type
            Тип трекера. Доступные варианты: [
                {(',' + chr(10) + '    ' * 4).join(trackers)}
            ]

        store_untracked
            Если True, метод update() возвращает все объекты (включая те,
            которым не назначен track_id). Если False - только оттреченные.

        **tracker_kwargs:
            Параметры трекера. Включает параметры ReID для трекеров из списка: [
                {(',' + chr(10) + '    ' * 4).join(reid_trackers)}
            ]
            По-умолчанию, зачения для ReID следующие:

                reid_weights = '{default_reid_kwargs['reid_weights']}'
                    Путь к весам модели ReID.

                device = {default_reid_kwargs['device']}
                    Устройство для вычислений ('cuda', 'cpu'). Если None, определяется
                    автоматически.

                half = {default_reid_kwargs['half']}
                    Использовать половинную точность (float16) для вычислений.

        Параметры для каждого из трекеров:"""
    add_doc = inspect.cleandoc(add_doc)  # Убираем лишний отступ

    # Добавляем с отступом внешней справочной строки:
    for tracker_name, tracker_constructor in trackers.items():
        # Формируем строки:
        if tracker_constructor.__doc__:
            tracker_doc = inspect.cleandoc(tracker_constructor.__doc__)
        else:
            tracker_doc = 'Информация отсутствует.'
        tracker_doc = textwrap.indent(tracker_doc, '    ')  # Отступ

        # Добавляем к основному корпусу:
        add_doc = (
            add_doc + '\n\n' + '=' * 88 + '\n\n' + tracker_name + ':\n\n' + tracker_doc
        )

    # Добалвяем результат к исходному документу:
    __init__.__doc__ = inspect.cleandoc(__init__.__doc__ or '') + '\n' + add_doc

    def reset(self) -> None:
        """Сбрасывает внутренние состояния трекера.

        Выполняет:
        1. Очистку списка известных меток классов
        2. Удаление текущего экземпляра трекера
        3. Создание нового экземпляра трекера с исходными параметрами

        Примечание:
        -----------
        Используется пересоздание трекера вместо вызова tracker.reset(),
        так как последний в текущей версии boxmot не работает.
        См. https://github.com/mikel-brostrom/boxmot/issues/1461
        """
        self._labels: list[str] = []

        # Удаляем старый трекер и создаём новый:
        if hasattr(self, 'tracker'):
            delattr(self, 'tracker')
        with suppress_module_logs():
            self.tracker = self.trackers[self.tracker_type](**self.tracker_kwargs)
        # tracker.reset() пока не работает
        # (см. https://github.com/mikel-brostrom/boxmot/issues/1461),
        # поэтому приходится сбрасывать через создание нового экземпляра класса.

    @staticmethod
    def _raise_on_tracked_obj(obj: BBox | Mask) -> None:
        """Проверяет, не принадлежит ли объект уже какому-либо треку.

        Параметры:
        ----------
        obj
            Объект для проверки.

        Исключения:
        -----------
        AttributeError
            Если объект уже имеет атрибут 'track_id'.

        Примечание:
        -----------
        Метод используется для предотвращения повторного трекинга
        уже отслеживаемых объектов.
        """
        track_id = obj.attribs.get('track_id', None)
        if track_id is not None:
            msg = f'Объект содержит номер трека = {track_id}!'
            raise AttributeError(msg)

    def _label2cls(self, label: str) -> int:
        """Переводит строковую метку в числовой идентификатор класса.

        Параметры:
        ----------
        label
            Строковая метка класса (например, 'person', 'car').

        Возвращает:
        -----------
        int
            Числовой идентификатор класса. Если метка встречается впервые,
            она добавляется во внутренний список меток.

        Примечание:
        -----------
        Метод используется для согласования строковых меток CVAT с числовыми
        классами, ожидаемыми трекерами boxmot.
        """
        try:
            return self._labels.index(label)
        except ValueError as e:
            if 'is not in list' in str(e):
                self._labels.append(label)
                return len(self._labels) - 1
            raise

    def _obj2det(
        self,
        obj: BBox | Mask,
    ) -> tuple[float, float, float, float, float, int]:
        """Конвертирует объект Mask или BBox в формат детекции для трекера.

        Параметры:
        ----------
        obj
            Объект детекции. Должен содержать атрибуты:
            - confidence (float): уверенность детекции
            - label (str): метка класса

        Возвращает:
        -----------
        tuple
            Кортеж (x1, y1, x2, y2, confidence, cls_id) в формате,
            ожидаемом трекерами boxmot.

        Исключения:
        -----------
        AttributeError
            Если объект уже имеет атрибут 'track_id'.
        TypeError
            Если передан объект неподдерживаемого типа.
        """
        # У объекта не должен быть номер трека:
        self._raise_on_tracked_obj(obj)

        # Получаем координаты прямоугольника:
        if isinstance(obj, Mask):
            xyxy = np.array(obj.asbbox('xyxy'))
        elif isinstance(obj, BBox):
            xyxy = obj.xyxy.copy()
        else:
            msg = f'Неподдерживаемый тип объекта: {type(obj)}!'
            raise TypeError(msg)

        # Получаем уверенность и класс объекта:
        conf = float(obj.attribs['confidence'])
        cls = self._label2cls(obj.attribs['label'])

        return *xyxy, conf, cls

    def update(
        self,
        objs: list[BBox | Mask] | tuple[BBox | Mask, ...],
        img: np.ndarray,
    ) -> list[BBox | Mask]:
        """Применяет трекер к очередному кадру и его объектам.

        Параметры:
        ----------
        objs
            Список объектов детекции (BBox или Mask) для текущего кадра.
            Каждый объект должен иметь атрибуты:
            - attribs['confidence']: float
            - attribs['label']: str
        img
            Изображение текущего кадра в формате RGB (H, W, 3).
            Некоторые трекеры используют изображение для извлечения
            визуальных признаков.

        Возвращает:
        -----------
        list
            Список объектов с добавленным атрибутом 'track_id'.
            В зависимости от параметра store_untracked:
            - Если True: все исходные объекты
            - Если False: только объекты с назначенным track_id

        Примечания:
        -----------
        - Каждому объекту добавляется атрибут 'track_id'.
        - Нумерация треков начинается с 0 (у boxmot с 1)
        - Объекты без назначенного трека получают track_id = None
        - Изображение конвертируется из RGB в BGR, так как boxmot ожидает BGR
        """
        # Собираем детекции:
        dets = np.array(list(map(self._obj2det, objs)))

        # Применяем трекер:
        tracks = self.tracker.update(dets, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #   M X (x, y, x, y, conf, cls) -> N X (x, y, x, y, id, conf, cls, ind)

        # Исправляем размер пустой таблицы:
        if len(tracks) == 0:
            tracks = np.zeros((0, 8))

        # Словарь перехода от индекса объекта в списке к его track_id:
        ind2track_id = {
            int(ind): int(track_id) - 1 for track_id, ind in tracks[:, [4, 7]]
        }
        # id на выходе из трекеров начинаются с 1, поэтому мы берём id - 1.

        # Прописываем номера треков в атрибутах каждого объекта:
        for ind, obj in enumerate(objs):
            obj.attribs['track_id'] = ind2track_id.get(ind)

        # Возвращаем все объекты, либо только оттреченные:
        if self.store_untracked:
            return list(objs)
        return [obj for obj in objs if obj.attribs['track_id'] is not None]

    __call__ = update


# При автономном запуске закачиваем модель osnet_x0_25_msmt17.pt в "~/models/":
if __name__ == '__main__':
    Tracker('botsort', device='cpu')
