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
import imageio

import numpy as np

from matplotlib import pyplot as plt
from IPython.display import Image
from PIL.Image import fromarray

def float2uin8(img):
    '''
    Перевод изображения из float в uint8.
    '''
    return (frame * 255).astype(np.uint8)


def video2gif(video, save2file='tmp.gif', duration=30, dither=False):
    '''
    Сохранить последовательность кадров в gif-файл.
    '''
    # Переводим кадры в uint8, если надо:
    video = list(map(float2uin8, video))

    # Сборка GIF- файла
    if dither:
        images = [fromarray(frame) for frame in video]
        images[0].save(save2file, save_all=True, append_images=images[1:], optimize=True, duration=duration, loop=0)
    else:
        imageio.mimsave(save2file, video, duration=duration)
    
    # Вывод GIF в тетрадку
    return Image(url=save2file)


class Mask:
    '''
    Класс масок выделения объектов на изображении.
    Предоставляет набор позезных методов для работы с картами.
    '''
    def __init__(self, array, rect=None):
        
        # Проверка входного параметра:
        assert isinstance(array, np.ndarray) # Маска собирается только из Numpy-массива
        assert array.ndim == 2               # Массив должен быть двумерным
        
        self.array = array
        self._rect = rect
        self._area = None
    
    # Приводим входные данны к классу Mask, если надо:
    @classmethod
    def __any2mask(cls, any):
        if type(any) != cls:
            any = cls(any)
        return any
    
    # Подготовка аргумента к различным операциям с текущим классом:
    def __preproc_other(self, other):
        
        # Приводим тип второго операнда к классу Mask, если надо:
        other = self.__any2mask(other)
        
        # Типы данных обеих масок должны быть одинаковыми:
        assert self.array.dtype == other.array.dtype
        
        return other
    
    # Объединение масок эквивалентно попиксельному взятию максимального значения из двух вариантов:
    def __or__(self, other):
        
        # Подготавливаем второй аргумент к различным операциям с текущим экземпляром класса:
        other = self.__preproc_other(other)
        
        # Возвращаем поэлементный максимум:
        return type(self)(np.dstack([self.array, other.array]).max(-1))
    
    # __radd__ работает некорректно с другими типами данных, так что он не определён.
    
    # Пересечение масок эквивалентно попиксельному взятию минимального значения из двух вариантов:
    def __and__(self, other):
        
        # Подготавливаем второй аргумент к различным операциям с текущим экземпляром класса:
        other = self.__preproc_other(other)
        
        # Возвращаем поэлементный минимум:
        return type(self)(np.dstack([self.array, other.array]).min(-1))
    
    # Создаёт структурный элемент
    def __make_kernel(self, kernel):
        
        # Если вместо ядра задано целое число, то создаём ядро виде круга с заданным диаметром.
        if isinstance(kernel, int):
            
            # Радиус:
            r = kernel/2
            
            # Строим круглое ядро:
            kernel = np.fromfunction(
                lambda x, y:((x - r + 0.5) ** 2 + (y - r + 0.5) ** 2 < r ** 2) * 1,
                (kernel, kernel), dtype=int).astype(self.array.dtype)
            # Взято с https://stackoverflow.com/a/73159803 .
            # Это лучше стандартного cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)) .
        
        return kernel
    
    # Морфологическая операция над текущей маской по структурному элементу kernel:
    def morphology(self, cv_morph_type, kernel=3):
        
        # Собираем структурный элемент, если надо:
        kernel = self.__make_kernel(kernel)
        
        # Возвращаем результат морфологического преобразования:
        return type(self)(cv2.morphologyEx(self.array, cv_morph_type, kernel))
    
    # Морфология бинарных операций:
    def      __mul__(self, kernel): return self.morphology(cv2.MORPH_DILATE, kernel) # m *  k = дилатация m по k
    def  __truediv__(self, kernel): return self.morphology(cv2.MORPH_ERODE , kernel) # m /  k =    эрозия m по k
    def      __pow__(self, kernel): return self.morphology(cv2.MORPH_CLOSE , kernel) # m ** k =  закрытие m по k
    def __floordiv__(self, kernel): return self.morphology(cv2.MORPH_OPEN  , kernel) # m // k =  открытие m по k
    
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
    def flip  (self, axis=None): return type(self)(np.flip  (self.array, axis)) # Отражение вдоль заданной оси
    def fliplr(self           ): return type(self)(np.fliplr(self.array      )) # Отражение вдоль горизонтали
    def flipud(self           ): return type(self)(np.flipud(self.array      )) # Отражение вдоль вертикали
    
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
        return [type(self)(self.array * (val == indexed_mask)) for val in range(1, ns)]
    
    # Обрамляющий прямоугольник (левый верхний угол, правый нижний угол):
    def rectangle(self):
        
        # Рассчитываем параметры прямоугольника, если он ещё 
        if self._rect is None:
            
            # Определяем границы ненулевых элементов с каждой из сторон:
            mask = self.array
            for xmin in          range(mask.shape[1]) :
                if mask[:, xmin].any(): break
            for ymin in          range(mask.shape[0]) :
                if mask[ymin, :].any(): break
            for xmax in reversed(range(mask.shape[1])):
                if mask[:, xmax].any(): break
            for ymax in reversed(range(mask.shape[0])):
                if mask[ymax, :].any(): break
            
            # Сохранение параметров обрамляющего прямоугольника во внутренней переменной:
            self._rect = xmin, ymin, xmax + 1, ymax + 1
            # Нужно для снятия необходимости вычислять параметры при каждом вызове.
        
        return self._rect
    
    # Обрамляющий прямоугольник (левый верхний угол, размеры):
    def asbbox(self):
        xmin, ymin, xmax, ymax = self.asrectangle()
        return xmin, ymin, xmax - xmin, ymax - ymin
    
    # Подсчёт площади сегмента в пикселях:
    def area(self):
        
        # Подсчитываем только если до того не считалось:
        if self._area is None:
            self._area = self.array.astype(bool).sum()
        
        return self._area

    # Есть ли пересечение обрамляющих прямоугольников двух масок:
    def is_rect_intersection(self, other):
        
        # Получаем сами обрамляющие прямоугольники:
        a_xmin, a_ymin, a_xmax, a_ymax = self .rectangle()
        b_xmin, b_ymin, b_xmax, b_ymax = other.rectangle()
        
        # Если любое из следующих условий не совпадает, то пересечений нет:
        if a_xmin > b_xmax: return False
        if b_xmin > a_xmax: return False
        if a_ymin > b_ymax: return False
        if b_ymin > a_ymax: return False
        
        # Если все вышеперечисленные условия соблюдены, то пересечение есть:
        return True
    
    # Индекс Жаккара:
    def Jaccard(self, other):
        
        # Если даже обрамляющие прямоугольники не пересекаются, то и самих масок тем более пересечений не будет:
        if not self.is_rect_intersection(other):
            return 0.
        
        # Рассчитываем площадь пересечения:
        intercection_area = (self & other).area()
        
        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта объединения:
        if intercection_area == 0:
            return 0.
        
        # Рассчитываем площадь объединения:
        overunion_area = (self | other).area()
        
        # Рассчёт индекса Жаккара:
        return intercection_area / overunion_area
    
    # Коэффициент перекрытия:
    def Overlap(self, other):
        
        # Если даже обрамляющие прямоугольники не пересекаются, то и самих масок тем более пересечений не будет:
        if not self.is_rect_intersection(other):
            return 0.
        
        # Рассчитываем площадь пересечения:
        intercection_area = (self & other).area()
        
        # Если пересечение = 0, то возвращаем сразу 0 без рассчёта объединения:
        if intercection_area == 0:
            return 0.
        
        # Рассчитываем площадь меньшей фигуры:
        min_area = min(self.area(), other.area())
        
        # Рассчёт коэффициента перекрытия:
        return intercection_area / min_area
    
    # Коэффициент Дайеса:
    def Dice(self, other):
        
        # Рассчитываем индекс Жаккара:
        J = self.Jaccard(other)
        
        # Переводим Жаккара в Дайеса:
        return 2 * J / (J + 1)
    
    # Intersection over Union:
    IoU = Jaccard
    
    def astype(self, dtype):
        return type(self)(self.array.astype(dtype))
    
    # Отрисовка маски:
    def show(self, inplace=True):
        plt.imshow(self.array, cmap='gray')
        plt.axis(False)

        if inplace:
            plt.show()


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

