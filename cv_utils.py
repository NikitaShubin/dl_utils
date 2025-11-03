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

import numpy as np

from matplotlib import pyplot as plt
from IPython.display import Image
from PIL.Image import fromarray
from tqdm import tqdm

from utils import (apply_on_cartesian_product, isfloat, isint,
                   draw_mask_on_image)


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

    '''
    @property
    def array(self):
        if hasattr(self, '_array') and self._array is not None:
            return self._array
        else: 

    @array.setter
    def array(self, val):
        self._array = val
    '''

    def __init__(self, array, area='auto', rect='auto'):
        # Проверка входного параметра:
        assert isinstance(array, np.ndarray)
        # Маска собирается только из Numpy-массива.
        assert array.ndim == 2
        # Массив должен быть двумерным.

        self.array = array.copy()
        self._rect = rect
        self._area = area

    '''
    # Конвертация в numpy-массив:
    def astype(self, type_=None):
        return self.array if type_ is None else self.array.astype(type_)
    '''

    def copy(self):
        return type(self)(self.array, self._area, self._rect)

    # Изменение типа массива:
    def astype(self, dtype):
        return type(self)(self.array.astype(dtype))

    # Создаёт изображение из маски:
    def to_image(self, dtype=np.uint8):
        if issubclass(dtype, np.floating):
            if self.array.dtype == bool:
                return self.array.astype(float)
            else:
                return self.array.copy()
        elif dtype == np.uint8:
            if self.array.dtype == bool:
                return self.array.astype(np.uint8) * 255
            else:
                return (self.array * 255).astype(np.uint8)

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
    def asbbox(self):
        xmin, ymin, xmax, ymax = self.rectangle()
        return xmin, ymin, xmax - xmin, ymax - ymin

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

    # Индекс Жаккара:
    def Jaccard_with(self, other):

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

    # Коэффициент перекрытия:
    def Overlap_with(self, other):

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

    # Коэффициент Дайеса:
    def Dice_with(self, other):

        # Рассчитываем индекс Жаккара:
        J = self.Jaccard_with(other)

        # Переводим Жаккара в Дайеса:
        return 2 * J / (J + 1)

    # Все три метрики совпадения масок разом:
    def JaccardDiceOverlap_with(self, other):

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


'''
mask = np.zeros((10, 10), np.uint8)
mask[3:5, 2: 4] = 1
mask[6:9, 3: 8] = 1
mask = Mask(mask)


for m in mask.split_segments():
    m.show()
    plt.show()
    print(m.asbbox())
'''
