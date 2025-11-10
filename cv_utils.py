'''
********************************************
* Набор утилит для работы с изображениями. *
*                                          *
*                                          *
* Основные функции и классы:               *
*   Mask - класс бинарных и полутоновых    *
*       масок.                             *
*                                          *
********************************************
'''


import cv2
import time
import datetime

import numpy as np

from matplotlib import pyplot as plt
from IPython.display import Image
from PIL.Image import fromarray
from tqdm import tqdm
from collections import defaultdict

from utils import apply_on_cartesian_product, isfloat, draw_mask_on_image


def float2uin8(img):
    '''
    Перевод изображения из float в uint8.
    '''
    return (img * 255).astype(np.uint8)


def video2gif(video,
              file='tmp.gif',
              duration=30,
              loop=True,
              desc=None):
    '''
    Сохранить последовательность кадров в gif-файл.
    '''
    # Превращаем тензор BxHxWxC в список изображений HxWxC:
    video = list(video)
    # Если video - уже список, то ничего не изменится.

    # Переводим кадры в uint8, если надо:
    if isfloat(video[0]):
        video = list(map(float2uin8, video))

    # Конвертация всех кадров в PIL-формат:
    images = [fromarray(frame) for frame in video]

    # Собираем кадры в GIF-файл:
    images[0].save(file,
                   format='GIF',
                   save_all=True,
                   append_images=tqdm(images[1:], desc=desc,
                                      disable=desc is None),
                   optimize=True,
                   duration=duration,
                   loop=0 if loop else None)

    # Вывод GIF в ячейку Jupyter-а:
    return Image(url=file)


class Mask:
    '''
    Класс масок выделения объектов на изображении.
    Предоставляет набор позезных методов для работы с картами.
    '''

    def __init__(self, array, area='auto', rect='auto', attribs={}):
        # Проверка входного параметра:
        assert isinstance(array, np.ndarray)
        # Маска собирается только из Numpy-массива.
        assert array.ndim == 2
        # Массив должен быть двумерным.

        self.array = array.copy()
        self._rect = rect
        self._area = area
        self.attribs = attribs  # Доп. атрибуты

    '''
    # Конвертация в numpy-массив:
    def astype(self, type_=None):
        return self.array if type_ is None else self.array.astype(type_)
    '''

    def _reset(self):
        '''
        Забывает положение обрамляющего прямоугольника и площадь сегмента.
        Используется в случае явного изменения array, т.к. эти параметры
        должны быть перерасчитаны.
        '''
        self._rect = 'auto'
        self._area = 'auto'

    def copy(self):
        return type(self)(self.array, self._area, self._rect, self.attribs)

    # Изменение типа массива:
    def astype(self, dtype):
        return type(self)(self.to_image(dtype), attribs=self.attribs)

    # Создаёт изображение из маски:
    def to_image(self, dtype=np.uint8):

        if self.array.dtype == bool:
            if dtype == bool:
                return self.array  # bool -> bool
            elif dtype == np.uint8:
                return self.array.astype(np.uint8) * 255  # bool -> uint8
            elif issubclass(dtype, np.floating):
                return self.array.astype(dtype)  # bool -> float
            else:
                raise ValueError(f'Неподдерживаемый целевой тип: {dtype}!')

        elif self.array.dtype == np.uint8:
            if dtype == bool:
                return self.array.astype(dtype)  # uint8 -> bool
            elif dtype == np.uint8:
                return self.array  # uint8 -> uint8
            elif issubclass(dtype, np.floating):
                return self.array.astype(dtype) / 255  # uint8 -> float
            else:
                raise ValueError(f'Неподдерживаемый целевой тип: {dtype}!')

        elif issubclass(self.array.dtype, np.floating):
            if dtype == bool:
                return self.array.astype(dtype)  # float -> bool
            elif dtype == np.uint8:
                return self.array.astype(dtype)  # float -> uint8
            elif issubclass(dtype, np.floating):
                return self.array.astype(dtype)  # float -> float
            else:
                raise ValueError(f'Неподдерживаемый целевой тип: {dtype}!')

        else:
            raise ValueError('Неподдерживаемый исходный тип: ' +
                             f'{self.array.dtype}!')

    # Импорт из COCO-формата:
    @classmethod
    def from_COCO_annotation(cls, coco_annotation):

        # Конвертация формата обрамляющего прямоугольника
        # из bbox в rect (x0, y0, w, h -> x0, y0, x1, y1):
        xmin, ymin, dx, dy = coco_annotation['bbox']
        rect = [xmin, ymin, xmin + dx, ymin + dy]

        # Собираем объект:
        return cls(array=coco_annotation['segmentation'],
                   area=coco_annotation['area'],
                   rect=rect)

    # Экспорт в COCO-формат:
    def as_COCO_annotation(self):
        return {'segmentation': self.array,
                'bbox': self.asbbox(),
                'area': self.area()}

    # Приводим входные данны к классу Mask, если надо:
    @classmethod
    def __any2mask(cls, any):
        if not isinstance(any, cls):
            any = cls(any)
        return any

    # Подготовка аргумента к различным операциям с текущим классом:
    def __preproc_other(self, other):

        # Приводим тип второго операнда к классу Mask, если надо:
        other = self.__any2mask(other)

        # Типы данных обеих масок должны быть одинаковыми:
        assert self.array.dtype == other.array.dtype

        return other

    # Объединение масок эквивалентно попиксельному взятию максимального
    # значения из двух вариантов:
    def __or__(self, other):

        # Подготавливаем второй аргумент к различным операциям с текущим
        # экземпляром класса:
        other = self.__preproc_other(other)

        # Возвращаем поэлементный максимум:
        return type(self)(np.dstack([self.array, other.array]).max(-1))

    # __radd__ работает некорректно с другими типами данных, так что он не
    # определён.

    # Пересечение масок эквивалентно попиксельному взятию минимального
    # значения из двух вариантов:
    def __and__(self, other):

        # Подготавливаем второй аргумент к различным операциям с текущим
        # экземпляром класса:
        other = self.__preproc_other(other)

        # Возвращаем поэлементный минимум:
        return type(self)(np.dstack([self.array, other.array]).min(-1))

    # Вычитание масок эквивалентно пересечению первой маски с инверсией
    # второй:
    def __sub__(self, other):

        # Подготавливаем второй аргумент к различным операциям с текущим
        # экземпляром класса:
        other = self.__preproc_other(other)

        # Возвращаем Результат вычитания:
        return self & ~other

    # Создаёт структурный элемент:
    @staticmethod
    def make_kernel(kernel):

        # Если вместо ядра задано целое число, то создаём ядро виде круга с
        # заданным диаметром.
        if isinstance(kernel, int):

            # Радиус:
            r = kernel / 2

            # Строим круглое ядро:
            kernel = np.fromfunction(
                lambda x, y: ((x - r + 0.5) ** 2 + (y - r + 0.5) ** 2 <
                              r ** 2) * 1,
                (kernel, kernel), dtype=int).astype(np.uint8)
            # Взято с https://stackoverflow.com/a/73159803 .
            # Это лучше стандартного
            # cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)).

        return kernel

    # Создаёт структурный элемент в форме окружности, ...
    # ... площадью в scale от площади текущего сегмента
    def make_scaled_kernel(self, scale=0.01):

        # Диаметр круга, площадью с текущий сегмент:
        D = 2 * np.sqrt(self.area() / np.pi)

        # Масштабируем, округляем размер и строим круг:
        return self.make_kernel(int(D * scale))

    # Морфологическая операция над текущей маской по структурному элементу
    # kernel:
    def morphology(self, cv_morph_type, kernel=3):

        # Собираем структурный элемент, если надо:
        if isinstance(kernel, int):
            kernel = self.make_kernel(kernel)
        elif isinstance(kernel, float):
            kernel = self.make_scaled_kernel(kernel)

        # Извлекаем исходную маску:
        array = self.array

        # Конвертируем её тип, если надо:
        if array.dtype == np.dtype('bool'):
            target_type = array.dtype
            array = array.astype(np.uint8)
        else:
            target_type = None

        # Выполняем морфологическое преобразование:
        array = cv2.morphologyEx(array, cv_morph_type, kernel)

        # Возвращаем результату исходный тип:
        if target_type:
            array = array.astype(target_type)

        # Оборачиваем полученную маску в нвый объект класса Mask:
        return type(self)(array)

    # Морфология бинарных операций:

    # m * k = дилатация m по k:
    def __mul__(self, kernel):
        return self.morphology(cv2.MORPH_DILATE, kernel)

    # m / k = эрозия m по k:
    def __truediv__(self, kernel):
        return self.morphology(cv2.MORPH_ERODE, kernel)

    # m ** k = закрытие m по k:
    def __pow__(self, kernel):
        return self.morphology(cv2.MORPH_CLOSE, kernel)

    # m // k = открытие m по k:
    def __floordiv__(self, kernel):
        return self.morphology(cv2.MORPH_OPEN, kernel)

    # Инверсия маски через унарный оператор "~" :
    def __invert__(self):

        # Определяем текущий тип маски:
        dtype = self.array.dtype

        # Определяем максимальное допустимое значение масти текущего типа:
        try:
            max_val = np.iinfo(dtype).max
        except ValueError:
            max_val = 1

        # Инвертируем маску:
        new_array = max_val - self.array

        # Приводим её к старому типу, если нужно:
        if new_array.dtype != dtype:
            new_array = new_array.astype(dtype)

        return type(self)(new_array)

    # Поворот k раз на 90 градусов против часовой стрелки:
    def rot90(self, k=1):
        return type(self)(np.rot90(self.array, k))

    # Отражения:

    # Отражение вдоль заданной оси:
    def flip(self, axis=None):
        return type(self)(np.flip(self.array, axis))

    # Отражение вдоль горизонтали:
    def fliplr(self):
        return type(self)(np.fliplr(self.array))

    # Отражение вдоль вертикали:
    def flipud(self):
        return type(self)(np.flipud(self.array))

    # Возвращает внешние или внутренние границы:
    def edges(self, external=True):
        return self * 3 - ~self if external else ~self * 3 - self

    # Возвращает список масок каждого из сегментов, входящих в текущую маску:
    def split_segments(self):

        # Получаем непосредственно саму маску:
        array = self.array

        # Если маска бинарная, то переводим её в uint8.
        if array.dtype == bool:
            array = array.astype(np.uint8)
        # Следующая функция не работает с бинарными масками.

        # Формируем карту разбиения на сегменты:
        ns, indexed_mask = cv2.connectedComponents(array)

        # Возвращаем списки отделённых сегментов:
        return [type(self)(self.array * (val == indexed_mask))
                for val in range(1, ns)]

    # Обрамляющий прямоугольник (левый верхний угол, правый нижний угол):
    def rectangle(self):

        # Рассчитываем параметры прямоугольника, если он ещё не рассчитан:
        if isinstance(self._rect, str) and self._rect == 'auto':

            # Определяем границы ненулевых элементов с каждой из сторон:
            mask = self.array
            for xmin in range(mask.shape[1]):
                if mask[:, xmin].any():
                    break

            # Ecли цикл дошёл до конца, значит, маска пуста:
            else:
                self._rect = None
                return None

            for ymin in range(mask.shape[0]):
                if mask[ymin, :].any():
                    break

            for xmax in reversed(range(mask.shape[1])):
                if mask[:, xmax].any():
                    break

            for ymax in reversed(range(mask.shape[0])):
                if mask[ymax, :].any():
                    break

            # Сохранение параметров обрамляющего прямоугольника во внутренней
            # переменной:
            self._rect = xmin, ymin, xmax + 1, ymax + 1
            # Нужно для снятия необходимости вычислять параметры при каждом
            # вызове.

        return self._rect

    # Обрамляющий прямоугольник (левый верхний угол, размеры):
    def asbbox(self, format='xywh'):
        xmin, ymin, xmax, ymax = self.rectangle()
        if format == 'xywh':
            return xmin, ymin, xmax - xmin, ymax - ymin
        elif format == 'xyxy':
            return xmin, ymin, xmax, ymax
        else:
            raise ValueError(f'Неподдерживаемый формат: {format}!')

    # Подсчёт площади сегмента в пикселях:
    def area(self):

        # Подсчитываем только если до того не считалось:
        if self._area == 'auto':
            self._area = self.array.astype(bool).sum()

        return self._area

    # Есть ли пересечение обрамляющих прямоугольников двух масок:
    def is_rect_intersection_with(self, other):

        # Получаем сами обрамляющие прямоугольники:
        a_rect = self.rectangle()
        if a_rect is None:
            return False

        b_rect = other.rectangle()
        if b_rect is None:
            return False
        # Если хоть у одной из масок метод rectangle возвращает None,
        # значит, маска пуста!

        a_xmin, a_ymin, a_xmax, a_ymax = a_rect
        b_xmin, b_ymin, b_xmax, b_ymax = b_rect

        # Если любое из следующих условий не совпадает, то пересечений нет:
        if a_xmin >= b_xmax:
            return False
        if b_xmin >= a_xmax:
            return False
        if a_ymin >= b_ymax:
            return False
        if b_ymin >= a_ymax:
            return False

        # Если все вышеперечисленные условия соблюдены, то пересечение есть:
        return True

    def Jaccard_with(self, other):
        '''
        Индекс Жаккара (Он же Intersection over Union).
        '''
        # Если даже обрамляющие прямоугольники не пересекаются, то и самих
        # масок тем более пересечений не будет:
        if not self.is_rect_intersection_with(other):
            return 0.

        # Рассчитываем площадь пересечения:
        intercection_area = (self & other).area()

        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта
        # объединения:
        if intercection_area == 0:
            return 0.

        # Рассчитываем площадь объединения:
        overunion_area = (self | other).area()

        # Рассчёт индекса Жаккара:
        return intercection_area / overunion_area

    def Overlap_with(self, other):
        '''
        Коэффициент перекрытия.
        '''
        # Если даже обрамляющие прямоугольники не пересекаются, то и самих
        # масок тем более пересечений не будет:
        if not self.is_rect_intersection_with(other):
            return 0.

        # Рассчитываем площадь пересечения:
        intercection_area = (self & other).area()

        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта меньшей
        # фигуры:
        if intercection_area == 0:
            return 0.

        # Рассчитываем площадь меньшей фигуры:
        min_area = min(self.area(), other.area())

        # Рассчёт коэффициента перекрытия:
        return intercection_area / min_area

    def Dice_with(self, other):
        '''
        Коэффициент Дайеса.
        '''
        # Рассчитываем индекс Жаккара:
        J = self.Jaccard_with(other)

        # Переводим Жаккара в Дайеса:
        return 2 * J / (J + 1)

    def JaccardDiceOverlap_with(self, other):
        '''
        Все три метрики совпадения масок разом.
        '''
        # Если даже обрамляющие прямоугольники не пересекаются, то и самих
        # масок тем более пересечений не будет:
        if not self.is_rect_intersection_with(other):
            return 0., 0., 0.

        # Рассчитываем площадь пересечения:
        intercection_area = (self & other).area()

        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта объединения
        # и меньшей фигуры:
        if intercection_area == 0:
            return 0., 0., 0.

        # Рассчитываем вспомогательные величины:
        overunion_area = (self | other).area()     # Площадь объединения
        min_area = min(self.area(), other.area())  # Площадь меньшей фигуры

        # Рассчёт метрик:
        J = intercection_area / overunion_area  # Индекс      Жаккара
        D = 2 * J / (J + 1)                     # Коэффициент Дайеса
        O = intercection_area / min_area        # Коэффициент перекрытия

        return J, D, O
    # Очерёдность метрик такова потому, что
    # всегда справедливо неравенство J <= D <= O.

    # Intersection over Union:
    IoU_with = Jaccard_with

    # Вынос внутренних методов в КЛАССОВЫЕ бинарные ф-ии:
    is_rect_intersection = staticmethod(is_rect_intersection_with)
    IoU = staticmethod(IoU_with)
    Dice = staticmethod(Dice_with)
    Overlap = staticmethod(Overlap_with)
    JaccardDiceOverlap = staticmethod(JaccardDiceOverlap_with)
    Jaccard = IoU

    # Отрисовка маски:
    def draw(self, img=None, color=255, alpha=1.):
        return draw_mask_on_image(self.array, img, color, alpha)

    # Отображение маски:
    def show(self, now=True, borders=True, *args, **kwargs):
        img = self.draw(*args, **kwargs)
        fig = plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        if borders:
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        else:
            plt.axis(False)

        if now:
            plt.show()
    '''
    Для все остальные атрибуты берутся из array:
    def __getattr__(self, name):
        if name in {'dtype', 'shape'}:
            return getattr(self.array, name)
        #     raise ValueError(f'Атрибут {name} не поддерживается!')
    '''

    def fitEllipse(self, est=False):
        '''
        Возвращает параметры эллипсов, аппроксимирующих контуры маски.

        Если est == True, то возвращается список словарей, где, помимо
        параметров эллипса, есть ряд других полезных величин, по которым можно
        оценить степень совпадения аппроксимации и получить сами маски.
        '''

        # Получаем список контуров для текущей маски:
        array = self.to_image()
        contours, hierarchy = cv2.findContours(array,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Инициируем список выводов:
        ellipses = []

        # Перебираем все найденные контуры:
        for contour, connection in zip(contours, hierarchy):

            # Для оценки параметров эллипса в контуре должно быть
            # минимум 5 точек:
            if len(contour) < 10:
                continue

            # Получаем оценку параметров эллипса:
            ellipse = cv2.fitEllipse(contour)

            if est:

                # Если контур всего один, берём исходную маску для проверки:
                if len(contours) == 1:
                    source = array

                # Если контуров несколько, растеризируем каждый отдельно:
                else:
                    source = np.zeros_like(array)
                    source = cv2.fillPoly(source, contour, 255)

                # Оборачиваем маску в соответствующий класс для сопоставления:
                source_mask = Mask(source)

                # Создаём маску по оцененным параметрам:
                target = cv2.ellipse(np.zeros_like(array), ellipse, 255, -1)
                target_mask = type(self)(target)

                # Подсчитываем метрики совпадения сегментов:
                JDO = source_mask.JaccardDiceOverlap_with(target_mask)

                # Упаковываем все результаты в словарь и добавляем его в
                # итоговый список:
                ellipses.append({'contour': contour,
                                 'connection': connection,
                                 'ellipse': ellipse,
                                 'source': source_mask,
                                 'target': target_mask,
                                 'IoU': JDO[0],
                                 'Jaccard': JDO[0],
                                 'Dice': JDO[1],
                                 'Overlap': JDO[2]})

            else:
                ellipses.append(ellipse)

        return ellipses


class BBox:
    '''
    Класс обрамляющих прямоугольников Вокруг объектов на изображении.
    Предоставляет набор ползезных методов для работы с прямоугольниками.
    '''

    def __init__(self, xyxy=None, imsize=None, attribs={}):
        self.xyxy = np.array(xyxy)
        self.imsize = imsize
        self.attribs = attribs

    @classmethod
    def from_format(cls, box, imsize=None, attribs={}, format='xywh'):
        '''
        Инициация из прямоугольников в разных форматах.
        '''
        if format == 'xywh':
            x, y, w, h = box
            box = x, y, x + w, y + h
            return cls(box, imsize, attribs)

        if format == 'xyxy':
            return cls(box, imsize, attribs)

        raise ValueError(f'Неподдерживаемый формат: {format}!')

    @classmethod
    def from_yolo(cls, box, imsize=None, attribs={}):
        '''
        Из обрамляющего прямоугольника в формате CVAT.
        '''
        # Если размер изображения не указан, придётся работать с
        # отностительными координатами:
        if imsize is None:
            imsize = (1., 1.)

        cx, cy, w, h = box
        cx = cx * imsize[1]
        cy = cy * imsize[0]
        hw = w * imsize[1] / 2
        hh = h * imsize[0] / 2
        box = cx - hw, cy - hh, cx + hw, cy + hh

        return cls(box, imsize, attribs)

    def copy(self):
        '''
        Создание дубликата.
        '''
        return type(self)(xyxy=self.xyxy,
                          imsize=self.imsize,
                          attribs=dict(self.attribs))

    def straight(self):
        '''
        Упорядочивает последовательность углов прямоугольника.
        Т.е. Делает левый верхний угол действительно левым верхним и т.п.
        '''
        copy = self.copy()
        x0, y0, x1, y1 = copy.xyxy
        copy.xyxy = np.array(
            [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        )
        return copy

    def area(self):
        '''
        Подсчёт площади прямоугольника.
        '''
        x0, y0, x1, y1 = self.xyxy
        return (x1 - x0) * (y1 - y0)

    def __and__(self, other):
        '''
        BBox пересечения текущего BBox-а с заданным либо None.
        '''
        if other is None:
            return

        copy = self.straight()
        other = other.straight()

        ax0, ay0, ax1, ay1 = copy.xyxy
        bx0, by0, bx1, by1 = other.xyxy

        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)

        # Если пересечения нет - возвращаем None:
        if x0 > x1 or y0 > y1:
            return

        copy.xyxy = np.array([x0, y0, x1, y1])

        return copy

    def __or__(self, other):
        '''
        BBox объединения текущего BBox-а с заданным.
        '''
        if other is None:
            return self.straight()

        ax0, ay0, ax1, ay1 = self.xyxy
        bx0, by0, bx1, by1 = other.xyxy

        x0 = min(ax0, bx0, ax1, bx1)
        y0 = min(ay0, by0, ay1, by1)
        x1 = max(ax0, bx0, ax1, bx1)
        y1 = max(ay0, by0, ay1, by1)

        copy = self.copy()
        copy.xyxy = np.array([x0, y0, x1, y1])

        return copy

    def intersection_area_with(self, other):
        '''
        Площадь пересечения текущего BBox-а с заданным.
        '''
        intersection = self & other
        return 0 if intersection is None else intersection.area()

    def Jaccard_with(self, other):
        '''
        Индекс Жаккара (Он же Intersection over Union).
        '''
        # Рассчитываем площадь пересечения:
        intercection_area = self.intersection_area_with(other)

        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта
        # объединения:
        if intercection_area == 0:
            return 0.

        # Рассчитываем площадь объединения:
        overunion_area = self.area() + other.area() - intercection_area

        # Рассчёт индекса Жаккара:
        return intercection_area / overunion_area

    def Overlap_with(self, other):
        '''
        Коэффициент перекрытия.
        '''
        # Рассчитываем площадь пересечения:
        intercection_area = self.intersection_area_with(other)

        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта меньшей
        # фигуры:
        if intercection_area == 0:
            return 0.

        # Рассчитываем площадь меньшей фигуры:
        min_area = min(self.area(), other.area())

        # Рассчёт коэффициента перекрытия:
        return intercection_area / min_area

    def Dice_with(self, other):
        '''
        Коэффициент Дайеса.
        '''
        # Рассчитываем индекс Жаккара:
        J = self.Jaccard_with(other)

        # Переводим Жаккара в Дайеса:
        return 2 * J / (J + 1)

    def JaccardDiceOverlap_with(self, other):
        '''
        Все три метрики совпадения прямоугольников разом.
        '''
        # Рассчитываем площадь пересечения:
        intercection_area = self.intersection_area_with(other)

        # Если площадь пересечения = 0, то возвращаем сразу 0 без
        # рассчёта объединения и меньшей фигуры:
        if intercection_area == 0:
            return 0., 0., 0.

        self_area = self.area()
        other_area = other.area()
        # Рассчитываем вспомогательные величины:
        overunion_area = self_area + other_area - intercection_area
        # Площадь объединения.
        min_area = min(self_area, other_area)
        # Площадь меньшей фигуры.

        # Рассчёт метрик:
        J = intercection_area / overunion_area  # Индекс      Жаккара
        D = 2 * J / (J + 1)                     # Коэффициент Дайеса
        O = intercection_area / min_area        # Коэффициент перекрытия

        return J, D, O
    # Очерёдность метрик такова потому, что
    # всегда справедливо неравенство J <= D <= O.

    # Intersection over Union:
    IoU_with = Jaccard_with

    # Вынос внутренних методов в КЛАССОВЫЕ бинарные ф-ии:
    IoU = staticmethod(IoU_with)
    Dice = staticmethod(Dice_with)
    Overlap = staticmethod(Overlap_with)
    JaccardDiceOverlap = staticmethod(JaccardDiceOverlap_with)
    Jaccard = IoU

    def draw(self, img=None, color=255, alpha=1., thickness=1):
        '''
        Отрисовка прямоугольника.
        '''
        # Доопределяем изображение:
        if img is None:
            if self.imsize is None:
                img = (self.xyxy[3], self.xyxy[2])
                # В крайнем случае ориентируемся на правый нижний угол.
            else:
                img = self.imsize

        # Если вместо изображения задан его размер, создаём его:
        if isinstance(img, (list, tuple)) or img.ndim == 1:
            if len(img) not in {2, 3}:
                raise ValueError('Размер должен содержать 2 или 3 размера:' +
                                 f'{img}!')
            img = np.zeros(img, dtype=np.uint8)

        # Рисуем на изображении:

        if alpha == 1.:
            # Делаем изображение цветным, если надо:
            if hasattr(color, '__len__') and len(color) == 3 and \
                    (img.ndim < 3 or img.shape[2] == 1):
                img = np.dstack([img] * 3)

            # Выполняем отрисовку:
            out = cv2.rectangle(img,
                                self.xyxy[:2], self.xyxy[2:],
                                color, thickness)

        else:
            # Строим маску, содержащую прямоугольник:
            mask = np.zeros(img.shape[:2], dtype=img.dtype)
            mask = cv2.rectangle(mask,
                                 self.xyxy[:2], self.xyxy[2:],
                                 255 if img.dtype == np.uint8 else 1, thickness)

            # Ннаносим маску на изображение:
            out = draw_mask_on_image(mask, img, color, alpha)

        return out

    # Отображение прямоугольника:
    def show(self, now=True, borders=True, *args, **kwargs):
        img = self.draw(*args, **kwargs)
        fig = plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        if borders:
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
        else:
            plt.axis(False)

        if now:
            plt.show()


def build_masks_IoU_matrix(masks1,
                           masks2=None,
                           diag_val=1.,
                           desc=None,
                           num_procs=0):
    '''
    Построение матрицы, в ячейках которой хранятся
    значения IoU для двух масок, которым соответствуют
    столбцы и строки этой матрицы. Если второй список
    масок не задан, то в его качестве берётся первый
    список.
    '''
    return apply_on_cartesian_product(Mask.Jaccard,
                                      masks1, masks2,
                                      symmetric=True,
                                      diag_val=diag_val,
                                      desc=desc,
                                      num_procs=num_procs).astype(float)


def build_masks_JaccardDiceOverlap_matrixs(masks1,
                                           masks2=None,
                                           diag_val=(1., 1., 1.),
                                           **mpmap_kwargs):
    '''
    Строит сразу 3 матрицы связностей для одного или двух списков масок.
    '''
    JDO = apply_on_cartesian_product(Mask.JaccardDiceOverlap,
                                     masks1,
                                     masks2,
                                     symmetric=True,
                                     diag_val=diag_val,
                                     **mpmap_kwargs)

    # Расфасовываем результаты в отдельные матрицы:
    j_mat = np.zeros_like(JDO, dtype=float)
    d_mat = np.zeros_like(JDO, dtype=float)
    o_mat = np.zeros_like(JDO, dtype=float)
    for i in range(JDO.shape[0]):
        for j in range(JDO.shape[1]):
            j_mat[i, j], d_mat[i, j], o_mat[i, j] = JDO[i, j]

    return j_mat, d_mat, o_mat


def build_bboxes_IoU_matrix(bboxes1,
                            bboxes2=None,
                            diag_val=1.,
                            desc=None):
    '''
    Построение матрицы, в ячейках которой хранятся
    значения IoU для двух масок, которым соответствуют
    столбцы и строки этой матрицы. Если второй список
    масок не задан, то в его качестве берётся первый
    список.
    '''
    return apply_on_cartesian_product(BBox.Jaccard,
                                      bboxes1, bboxes2,
                                      symmetric=True,
                                      diag_val=diag_val,
                                      desc=desc,
                                      num_procs=1).astype(float)


def build_bboxes_JaccardDiceOverlap_matrixs(bboxes1,
                                            bboxes2=None,
                                            diag_val=(1., 1., 1.),
                                            **mpmap_kwargs):
    '''
    Строит сразу 3 матрицы связностей для одного или двух списков масок.
    '''
    # Отключаем параллельность, т.к. на транзакционные издержки уйдёт ббольше
    # времени:
    mpmap_kwargs['num_procs'] = 1

    JDO = apply_on_cartesian_product(BBox.JaccardDiceOverlap,
                                     bboxes1, bboxes2,
                                     symmetric=True,
                                     diag_val=diag_val,
                                     **mpmap_kwargs)

    # Расфасовываем результаты в отдельные матрицы:
    j_mat = np.zeros_like(JDO, dtype=float)
    d_mat = np.zeros_like(JDO, dtype=float)
    o_mat = np.zeros_like(JDO, dtype=float)
    for i in range(JDO.shape[0]):
        for j in range(JDO.shape[1]):
            j_mat[i, j], d_mat[i, j], o_mat[i, j] = JDO[i, j]

    return j_mat, d_mat, o_mat


def build_IoU_matrix(objs1,
                     objs2=None,
                     diag_val=1.,
                     desc=None,
                     num_procs=0):
    '''
    Аналогичен build_masks_IoU_matrix и build_bboxes_IoU_matrix, но работает
    с объектами обоих типов (Mask или BBox).
    Требуется, чтобы список(и) содержал гомогенные элементы
    (были строго одного типа)!
    '''
    # Составляем множество типов объектов:
    types = set(map(type, objs1))
    if objs2 is not None:
        types |= set(map(type, objs2))

    if len(types) > 1:
        raise ValueError(f'Элементы списка(ов) негомогенны: {types}!')

    # Применяем подходящую функцию:
    if Mask in types:
        return build_masks_IoU_matrix(objs1, objs2, diag_val, desc,
                                      num_procs)
    elif BBox in types:
        return build_bboxes_IoU_matrix(objs1, objs2, diag_val, desc)
    else:
        raise TypeError(f'Неподдерживаемый тип элементов: {types.pop()}!')


def build_JaccardDiceOverlap_matrixs(objs1,
                                     objs2=None,
                                     diag_val=(1., 1., 1.),
                                     **mpmap_kwargs):
    '''
    Аналогичен build_masks_JaccardDiceOverlap_matrixs и 
    build_bboxes_JaccardDiceOverlap_matrixs, но работает
    с объектами обоих типов (Mask и BBox).
    '''
    # Составляем множество типов объектов:
    types = set(map(type, objs1))
    if objs2 is not None:
        types |= set(map(type, objs2))

    if len(types) > 1:
        raise ValueError(f'Элементы списка(ов) негомогенны: {types}!')

    # Применяем подходящую функцию:
    if Mask in types:
        return build_masks_JaccardDiceOverlap_matrixs(objs1, objs2, diag_val,
                                                      **mpmap_kwargs)
    elif BBox in types:
        return build_bboxes_JaccardDiceOverlap_matrixs(objs1, objs2, diag_val,
                                                       **mpmap_kwargs)
    else:
        raise TypeError(f'Неподдерживаемый тип элементов: {types.pop()}!')

    return builder(objs1, objs2, diag_val, **mpmap_kwargs)


def split_by_attrib(objs, attrib='label', as_dict=False):
    '''
    Разбивает список объектов на подгруппы по значению заданного
    атрибута. По-умолчанию группирует объекты по меткам класса.

    Т.е.:
    [obj1, obj2, ...] -> [[obj1, obj3, ...], [obj2, ...], ...]
    '''
    objs_dict = defaultdict(list)  # Словарь для прямоугольников с атрибутом
    nonmarked = []                 # Список для безатрибутных прямоугольников

    for obj in objs:

        # Если атрибут есть, вносим прямоугольник в одноимённый список:
        if attrib in obj.attribs:
            attrib_val = obj.attribs[attrib]
            objs_dict[attrib_val].append(obj)

        # Если атрибут не указан, заносим прямоугольник в отдельный список:
        else:
            nonmarked.append(obj)

    # Если нужен словарь + остаток:
    if as_dict:
        return objs_dict, nonmarked

    # Если нужны просто списки списков:
    else:
        return list(objs_dict.values()) + [nonmarked]


def sort_by_attrib(objs,
                   attrib='confidence',
                   key=None,
                   descending=True,
                   nonmarked='first'):
    '''
    Сортирует спиоск объектов по значению какого-либо атрибута.

    key - ключ сортировки (как в sorted).

    nonmarked - флаг обработки безатрибутных объектов:
            nonmarked = 'first':
        помещает объекты беэ этого аттрибута впереди всех;
            nonmarked = 'last':
        помещает объекты беэ этого аттрибута позади всех;
            для всех других значениях nonmarked при встерече с таким
        объектом возникает ошибка!
    '''
    # Разбиваем на подсписки по значениям атрибута:
    objs_dict, nonmarked_ = split_by_attrib(objs, attrib, True)

    # Сортируем значения атрибута:
    keys = sorted(objs_dict.keys(), key=key, reverse=not descending)

    # Упорядочиваем сами прямоугольники по их атрибутам:
    objs = []
    for key in keys:
        objs += objs_dict[key]

    nonmarked = nonmarked.lower()
    if nonmarked == 'first':
        return nonmarked_ + objs
    if nonmarked == 'last':
        return objs + nonmarked_

    if nonmarked_:
        raise ValueError(f'Найдено {len(nonmarked_)} безатрибутных объекта!')

    return objs


class PrintInfo:
    '''
    Декоратор-класс для отслеживания вызовов callable-объектов,
    преобразующих список в список.
    '''

    def __init__(self, wrapped=None, name: str | None = None):
        '''
        Args:
                wrapped: Объект с методом __call__, принимающим список и
            возвращающим список;
                name: Имя для отображения в выводе
            (по умолчанию: имя класса wrapped).
        '''
        if wrapped is not None and not hasattr(wrapped, '__call__'):
            raise ValueError(f'{wrapped} должен быть вызываемым объектом!')

        self._wrapped = wrapped
        self._name = name or wrapped.__class__.__name__

    def __call__(self, input_list: list) -> list:
        '''
        Вызывает обернутый объект, измеряет время и выводит статистику.
        '''
        # Если объект для декорации не задан,
        # просто выводим время и размер списка:
        if self._wrapped is None:
            now = datetime.datetime.now()
            print(f'[{now}]: len = {len(input_list)}')
            return input_list

        # Если объект для декорации задан, выводим размеры
        # списка до и после обработки, и время на обработку:
        else:
            # Длина входного списка
            input_len = len(input_list)

            # Засекаем время
            start_time = time.time()
            result = self._wrapped(input_list)
            end_time = time.time()

            # Длина выходного списка и время обработки
            output_len = len(result)
            dtime = end_time - start_time

            # Вывод информации
            print(f'{input_len} -> [{self._name} ({dtime:.2f} с.)] ' +
                  f'-> {output_len}')

            return result


class NMS:
    '''
    Классический Non-maximum Suppression для списка обнаруженных
    объектов в кадре.
    '''

    def __init__(self, minIoU=0.5):
        self.minIoU = 0.5

    def __call__(self, objs):
        # Список из менее двух объектов оставляем без изменения:
        if len(objs) < 2:
            return objs

        # Группируем по классам:
        objs_list = split_by_attrib(objs)

        # Обрабатываем список для каждого класса отдельно:
        objs = []  # Итоговый список объектов
        for label_objs in objs_list:

            # Если объектов текущего класса меньше двух - переносим его в
            # итоговый без изменений:
            if len(label_objs) < 2:
                objs += label_objs
                continue

            # Упорядочиваем по убыванию уверенности:
            label_objs = sort_by_attrib(label_objs, nonmarked='raise')

            # Строим матрицу связностей:
            j_mat = build_IoU_matrix(label_objs)

            # Перебираем все пары:
            excluded_inds = []  # Индексы исключённых объектов
            for i in range(len(label_objs) - 1):
                if i in excluded_inds:
                    continue  # Исключённые индексы пропускаем.

                for j in range(i + 1, len(label_objs)):
                    if j in excluded_inds:
                        continue  # Исключённые индексы пропускаем.

                    # Если пересечение текущей пары >= порогового, то исключаем
                    # тот, что с меньшей уверенностью:
                    if j_mat[i, j] >= self.minIoU:
                        excluded_inds.append(j)

            # Пополняем итоговый список объектов неисключёнными позициями:
            objs += [obj for ind, obj in enumerate(label_objs)
                       if ind not in excluded_inds]

        return objs