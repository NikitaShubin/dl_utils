#!/usr/bin/env python3
'''
********************************************
*   Различные утилиты общего назначения,   *
*       не использующие ни один из         *
*              DL-фреймворков.             *
*                                          *
*                                          *
* Работа с фото и видео:                   *
*   cv2_img_exts и cv2_vid_exts -          *
*       множества, содежращие              *
*       поддерживаемые OpenCV расширения   *
*       фото и видео соответственно.       *
*                                          *
*   fig2nparray - функция, выполняющая     *
*       захват содержимого  .              *
*       Matplotlib-фигуры, преобразуя его  *
*       изображение в виде numpy-массива.  *
*                                          *
*   draw_contrast_text - функция,          *
*       наносящая на изображение           *
*       контрастную надпись, читаемую на   *
*       любом фоне.                        *
*                                          *
*   resize_with_pad - функция, выполняющая *
*       изменение размера изображения с    *
*       сохранением соотношения сторон за  *
*       счёт использования паддинга.       *
*                                          *
*  img_dir2video - функция, собирающая     *
*       видео-превью из изображнений,      *
*       находящихся в заданной папке.      *
*                                          *
*  ImReadBuffer - класс, буферизирующий    *
*       чтение кадра из видео или фото.    *
*                                          *
*                                          *
* Работа с файловой системой:              *
*   mkdirs - функция, создающая все        *
*       недостающие папки для              *
*       существования полного пути (не     *
*       выводит ошибку, если путь уже      *
*       существует).                       *
*                                          *
*   rmpath - функция, удаляющая файл или   *
*       папку вместе со всем её            *
*       содержимым (не. выводит ошибку,    *
*       если путь уже не существует).      *
*                                          *
*   first_existed_path - функция,          *
*       возвращающая первый существующий   *
*       путь (полезна при использовании    *
*       скрипта на нескольких машинах, где *
*       ресурсы могут находиться в разных  * 
*       местах).                           *
*                                          *
*   unzip_dir - функция, распаковывающая   *
*       каждый zip-архив из заданной папки *
*       в свою подпапку в другом месте.    *
*       (полезно для распаковки            *
*        датасетов).                       *
*                                          *
*                                          *
* Работа с процессами:                     *
*   mpmap - функция, выполняющая обработку *
*       каждого элемента списка в          *
*       отдельном процессе (значительн     *
*       ускоряет обработку вычислительно   *
*       затратными функциями, способными   *
*       работать параллельно).             *
*                                          *
*                                          *
* Другие утилиты:                          *
*   rim2arabic - функция, переводящая      *
*       римские цифры в арабские.          *
*                                          *
*   restart_kernel_and_run_all_cells -     *
*       функция, перезагружающая ядро      *
*       текущего ноутбука.                 *
*                                          *
*   cls - функция, очистки консоли или     *
*       ячейки.                            *
*                                          *
*   timeit - Контекст, засекающий время    *
*       выполнения вложенного кода в       *
*       секундах.                          *
*                                          *
********************************************
'''


import os
import cv2
import yaml
import random
import tempfile
import ast
import numpy as np
import zipfile
import glob
import json
import time
import logging

# from inspect import isclass
from pathlib import Path
from functools import reduce
from shutil import rmtree, copyfile, move
from tqdm import tqdm
from multiprocessing import pool, Pool
from IPython.display import clear_output, HTML  # , Javascript, display
from matplotlib import pyplot as plt
from typing import Iterable, Self
from collections import defaultdict, deque
from collections.abc import Callable


########################
# Работа с фото/видео: #
########################

def autocrop(img):
    '''
    Автоматически удаляет края, представляющие собой монотонную заливку.
    '''
    # Определяем границы обрезки:

    for i_min in range(img.shape[0]):
        if not (img[0, 0, ...] == img[i_min, :, ...]).all():
            break

    for j_min in range(img.shape[1]):
        if not (img[0, 0, ...] == img[:, j_min, ...]).all():
            break

    for i_max in reversed(range(img.shape[0])):
        if not (img[-1, -1, ...] == img[i_max, :, ...]).all():
            break

    for j_max in reversed(range(img.shape[1])):
        if not (img[-1, -1, ...] == img[:, j_max, ...]).all():
            break

    # Возвращаем пустоту, если всё изображение было монотонным:
    if i_min >= i_max or j_min >= j_max:
        return None

    # Возвращаем обрезку:
    return img[i_min:i_max + 1, j_min:j_max + 1]


def df2img(df, file=None, title='index', show=False):
    '''
    Преобразоывает датафрейм в иизображение.
    '''
    # Доопределяем путь к файлу:
    rm_file = False
    if file is None:
        rm_file = True
        file = os.path.join(tempfile.gettempdir(), 'tmp.png')

    # Выводим таблицу через matplotlib:
    fig, ax = plt.subplots()
    ax.set_axis_off()
    table = ax.table(df.values, rowLabels=df.index, colLabels=df.columns,
                     cellLoc='center', loc='upper left')
    if title:
        ax.set_title(df.index.name if title=='index' else title,
                     fontweight="bold", loc='left')

    # Первичное созранение изображения таблицы:
    plt.savefig(file,
                bbox_inches='tight',
                transparent=True,
                dpi=200)

    img = plt.imread(file)  # Чтение первичного изображения
    img = autocrop(img)     # Обрезка полей

    # Удаляем временный файл или сохраняем окончательный вариант:
    if rm_file:
        rmpath(file)
    else:
        plt.imsave(file, img)

    # Фиксируем или убираем изображение:
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis(False)
        plt.show()
    else:
        plt.close()

    return img


def fig2nparray(fig=None):
    '''
    Возвращает содержимое фигуры из Matplotlib в виде numpy-массива
    RGB-изображения.
    Бездумно взято отсюда:
    https://stackoverflow.com/questions/7821518/
    matplotlib-save-plot-to-numpy-array
    '''
    if fig is None:
        fig = plt.gcf()

    # Принудительная отрисовка:
    fig.canvas.draw()

    # Захват данных в виде вектора:
    vector = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

    # Перевод вектора в изображение:
    image = vector.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.cla()
    plt.clf()
    plt.show()

    return image


# Неполный список видеофайлов, поддерживаемых OpenCV для чтения:
cv2_vid_exts = {'.mpg', '.mpeg', '.mp4', '.mkv', '.avi', '.mov', '.ts'}

# Неполный список изображений, поддерживаемых OpenCV для чтения:
cv2_img_exts = {'.bmp', '.jpg', '.jpeg', '.tif', '.tiff', '.png'}

cv2_exts = cv2_vid_exts | cv2_img_exts


def dtype_like(val):
    '''
    Берёт dtype заданной переменной, даже если она не класса np.ndarray.
    Используется в функциях для приведении типа выходного массива типу
    входного.

    Если передан список/кортеж, то берётся тип первого элемента.
    '''
    if hasattr(val, 'dtype'):
        return val.dtype
    elif isinstance(val, (tuple, list)):
        return dtype_like(val[0])
    else:
        return type(val)


def overlap_with_alpha(image, watermark, return_with_watermark=False):
    '''
    Накладывает на исходное изображение водяной знак, содержащий альфаканал.
    '''
    # Получаем размеры изображений:
    img_shape =     image.shape
    wtm_shape = watermark.shape

    # Фиксируем тип исходного изображения:
    img_dtype = image.dtype

    # Некоторые проверки входных данных:
    assert img_shape[:2] == wtm_shape[:2]
    assert wtm_shape[2] in {2, 4}  # Водяной знак обязан иметь альфаканал

    # Если исходное изображение или водяной знак представлено
    # целыми числами, то меняем его тип на тип с плавающей точкой:
    if isint(image    ): image     = image     / np.iinfo(image    .dtype).max
    if isint(watermark): watermark = watermark / np.iinfo(watermark.dtype).max
    # Вместе с этим нормализуем значения (максимально возможное значение = 1.)

    # Переводим оба изображения в float32, т.к. cv2 ...
    # ... не работает с другими типами плавающих точек:
    image     = image    .astype(np.float32)
    watermark = watermark.astype(np.float32)

    # Разделяем исходное изображение на содержимое и альфаканал, если он есть:

    # Если изображение не имеет альфаканал:
    if len(img_shape) == 2 or img_shape[2] in {1, 3}:
        img       = image
        img_alpha = None

    # Если изображение имеет альфаканал:
    elif img_shape[2] in {2, 4}:
        img       = image[..., :-1]
        img_alpha = image[...,  -1]

    else:
        raise ValueError(f'Некорректный размер изображения: {img_shape}!')

    # Разделяем исходное изображение на содержимое и альфаканал:
    wtm       = watermark[..., :-1]
    wtm_alpha = watermark[...,  -1]

    # Если размеры исходного изображения и водяного ...
    # ... знака не совпадают, то исправляем это:
    if img.shape != wtm.shape:

        # Если в исходном изображении вообще 2 измерения:
        if len(img.shape) == 2:

            # Если водяной знак цветной, переводим в монохромный:
            if wtm.shape[2] == 3:
                wtm = cv2.cvtColor(wtm, cv2.COLOR_RGB2GRAY)

            # Если в водяном знаке всего 1 канал, отбрасываем лишнее измерение:
            elif wtm.shape[2] == 1:
                wtm = wtm[..., 0]

            # До сюда код доходить не должен:
            else:
                raise Exception('В коде допущена ошибка!')

        # Если в исходном изображении всего 1 канал:
        elif img.shape[2] == 1:
            wtm = cv2.cvtColor(wtm, cv2.COLOR_RGB2GRAY)[..., np.newaxis]
        # Переводим водяной знак в оттенки ...
        # ... серого с числом каналов, равным 1.

        # Если исходное изображение цветное, ...
        # ... делаем цветным и водяной знак:
        elif img.shape[2] == 3:
            wtm = np.dstack([wtm] * 3)

        # До сюда код доходить не должен:
        else:
            raise Exception('В коде допущена ошибка!')

    # За основу маски наложения берём альфаканал водяного знака:
    mask = wtm_alpha

    # Если маска наложения не соразмерна исходному изображению, исправляем:
    if img.shape != mask.shape:
        if img.shape[2] == 1:
            mask = mask[..., np.newaxis]
        elif img.shape[2] == 3:
            mask = np.dstack([mask] * 3)

    # Рассчитываем конечное изображение (выполняем наложение):
    rzlt = img * (1 - mask) + wtm * mask

    # Обновляем альфаканал, если он был:
    if img_alpha is not None:

        # Выполняем наложение альфаканалов:
        rzlt_alpha = np.dstack([img_alpha[..., np.newaxis],
                                wtm_alpha[..., np.newaxis]])
        rzlt_alpha = rzlt_alpha.max(-1, keepdims=True)

        # Добавляем альфаканал к конечному изображению:
        rzlt = np.dstack([rzlt, rzlt_alpha])

    # Если типы текущего и исходного изображений не совпадают:
    if rzlt.dtype != img_dtype:

        # Обращаем нормализацию, если надо:
        if isint(img_dtype):
            rzlt *= np.iinfo(img_dtype).max

        # Возвращаем конечному изображению исходный тип:
        rzlt = rzlt.astype(img_dtype)

    # Возвращаем результат вместе с собранной маской
    # (например, чтобы использовать повторно), либо без неё:
    if return_with_watermark:
        return rzlt, np.dstack([wtm, wtm_alpha])
    else:
        return rzlt


def color2img(color, imsize):
    '''
    Создаёт изображение заданного размера, залитое указанным цветом.
    '''
    # Создаём изображение нужного цвета и берём образец цвета для определения
    # выходного типа:
    if hasattr(color, '__len__'):
        if len(color) == 1:
            img = np.ones(imsize) * color[0]
        elif len(imsize) == 2 or imsize[2] == 1:
            img = np.ones(list(imsize[:2]) + [len(color)]) * color
        elif imsize[2] == len(color):
            img = np.dstack([np.ones(imsize[:2]) * c for c in color])
        else:
            raise ValueError('Несовпадение числа каналов ' +
                             f'цвета ({len(color)}) и ' +
                             f'изображения ({imsize[2]})!')
        color_val_example = color[0]

    else:
        img = np.ones(imsize) * color
        color_val_example = color

    # Если цвет задан целым числом - используем тип uint8:
    if isint(color_val_example):
        img = img.astype(np.uint8)

    return img


def draw_mask_on_image(mask, img=None, color=None, alpha=1.):
    # Если не заданы ни изображение, ни цвет, то
    # нужна упрощённая отрисовка:
    if img is None and color is None:
        if alpha == 1.:
            return mask.copy()
        else:
            return (mask * alpha).astype(mask)

    # Если цвет не задан, то берём маскимально допустимое значение
    # яркости маски:
    if color is None:
        color = 255 if img is None or img.dtype == np.uint8 else 1.

    # Создаём чёрное полотно, если исходное изображение не задано:
    if img is None:
        img = color2img(0, mask.shape)

    # Строим заливку маски:
    watermark_color = color2img(color, mask.shape)
    if isint(watermark_color):
        max_dtype_value = np.iinfo(watermark_color.dtype).max
        watermark_color = watermark_color / max_dtype_value
    # Принудительно переводим во float.

    # Строим альфаканал маски:
    watermark_alpha = mask
    if watermark_alpha.dtype == bool:
        watermark_alpha = watermark_alpha.astype(float)
    elif isint(watermark_alpha):
        max_dtype_value = np.iinfo(watermark_alpha.dtype).max
        watermark_alpha = watermark_alpha / max_dtype_value
    if alpha != 1.:
        watermark_alpha *= alpha
    # Тоже во float.

    # Cобираем водяной знак:
    watermark = np.dstack([watermark_color, watermark_alpha])

    # Делаем исходное изображение цветным, если оно монохромное, а
    # водяной знак цветной:
    if (img.ndim == 2 or img.shape[2] == 1) and watermark.shape[2] == 4:
        img = np.dstack([img] * 3)
    # Это нужно т.к. в противном случае overlap_with_alpha выдаст
    # монохромный результат.

    # Наносим водяной знак на изображение и возвращаем:
    return overlap_with_alpha(img, watermark)


def put_text_carefully(text         : 'Растеризируемый текст'                             ,
                       img_or_imsize: 'Изображение или его размер'                        ,
                       coordinates  : 'Желаемые координаты центра текста'= (0, 0)         ,
                       scale        : 'Масштаб шрифта'                   = 0.6            ,
                       color        : 'Цвет текста'                      = (255, 255, 255),
                       alpha        : 'Прозрачность текста'              = 1.             ):
    '''
    Размещает текст на изображении так, чтобы он не вышел из рамок.
    '''
    # Растеризация текста:
    text_img = text2img(text, scale=scale)

    # Если передано само изображение, ничего не делаем:
    if isinstance(img_or_imsize, np.ndarray) and img_or_imsize.ndim in {2, 3}:
        img = img_or_imsize.copy()
        imsize = img.shape[:2]

    # Создаём изображение по размеру, если он задан:
    elif hasattr(img_or_imsize, '__len__') and len(img_or_imsize) in {2, 3}:
        img = np.zeros(img_or_imsize, np.uint8)
        imsize = img[:2]

    else:
        raise ValueError('В качестве параметра "img_or_imsize" должно быть' +
                         ' передано изображение, его размер или значение. ' +
                         f'Получено {img}!')

    # Размер текстового спрайта:
    textsize = text_img.shape[:2]

    # Уменьшаем размер спрайта, если он не влезает в изображение:
    resize_scale = 1.
    for dim in range(2):
        if textsize[dim] > imsize[dim]:
            resize_scale = min(resize_scale, imsize[dim] / textsize[dim])
    if resize_scale < 1.:
        textsize = (np.fix(resize_scale * textsize[0]).astype(int),
                    np.fix(resize_scale * textsize[1]).astype(int))
        text_img = cv2.resize(text_img, (textsize[1], textsize[0]))

    # Определяем координаты левого верхнего угла спрайта:
    my = coordinates[0] - textsize[0] / 2
    mx = coordinates[1] - textsize[1] / 2

    # Сдвигаем координаты, если спрайт вышел за границы:
    my = int(min(max(0, my), imsize[0] - textsize[0]))
    mx = int(min(max(0, mx), imsize[1] - textsize[1]))

    # Наносим спрайт на нужную часть исходного изображения:
    img[my: my + textsize[0], mx: mx + textsize[1], ...] = draw_mask_on_image(
        text_img,
        img[my: my + textsize[0], mx: mx + textsize[1], ...],
        color=color,
        alpha=alpha,
    )

    return img


def color_float_hsv_to_uint8_rgb(h: float,
                                 s: float = 1.,
                                 v: float = 1.,
                                 a: float | None = None) -> tuple:
    '''
    Переводит вещественные HSV(A) в целые RGB(А).
    Полезно для выбора цветов в визуализациях.
    Передрано с https://stackoverflow.com/a/26856771/14474616
    '''
    if a is not None:
        a = int(255*a)

    if s:
        if h == 1.0:
            h = 0.0
        i = int(h*6.0)
        f = h*6.0 - i

        w = int(255*(v * (1.0 - s)))
        q = int(255*(v * (1.0 - s * f)))
        t = int(255*(v * (1.0 - s * (1.0 - f))))
        v = int(255*v)

        if i == 0:
            return (v, t, w) if a is None else (v, t, w, a)
        if i == 1:
            return (q, v, w) if a is None else (q, v, w, a)
        if i == 2:
            return (w, v, t) if a is None else (w, v, t, a)
        if i == 3:
            return (w, q, v) if a is None else (w, q, v, a)
        if i == 4:
            return (t, w, v) if a is None else (t, w, v, a)
        if i == 5:
            return (v, w, q) if a is None else (v, w, q, a)

    else:
        v = int(255*v)
        return (v, v, v) if a is None else (v, v, v, a)


def get_n_colors(n):
    '''
    Генерирует список цветов максимальной различимости по тону.
    Полезен во всяких визуализациях.
    '''
    h = np.linspace(0, 1, n, endpoint=False)
    return list(map(color_float_hsv_to_uint8_rgb, h))


def text2img(text : 'Растеризируемый текст'                       ,
             img  : 'Изображение или его размер' = 'auto'         ,
             scale: 'Масштаб шрифта'             = 0.6            ,
             color: 'Цвет текста'                = (255, 255, 255)):
    '''
    Простой способ векторизации текста или нанесения надписи на изображение.
    По умолчанию формируется оптимальный для текста размер полутонового
    изображения.
    '''
    # Разделяем текст на строки:
    lines = text.split('\n')

    # Флаг автоматической обрезки итогового изображени:
    auto_crop = False

    # Размер символа в пикселях:
    scale_rate = scale / 0.6
    char_size  = int(scale_rate * 20)
    shift_size = int(scale_rate *  2)

    # Если размер изображения выбирается автоматически:
    if isinstance(img, str) and img.lower() == 'auto':
        # Высоту берём в зависимости от числа строк:
        h = shift_size + char_size * len(lines)
        # Ширину берём в зависимости от длины самой большой строки:
        w = shift_size + char_size * max(map(len, lines))
        # Инициализируем изображение:
        img = np.zeros((h, w), np.uint8)
        # Отмечаем, что изображение нужно потом обрезать:
        auto_crop = True

    # Если передано само изображение, ничего не делаем:
    elif isinstance(img, np.ndarray) and img.ndim in {2, 3}:
        pass

    # Создаём изображение по размеру, если он задан:
    elif hasattr(img, '__len__') and len(img) in {2, 3}:
        img = np.zeros(img, np.uint8)

    else:
        raise ValueError('В качестве параметра "img" должно быть передано' +
                         f' изображение, его размер или значение. Получено {img}!')

    # Наносим каждую строку текста:
    for line_ind, line in enumerate(lines, 1):
        img = cv2.putText(img                               ,
                          line                              ,
                          (shift_size, char_size * line_ind),
                          cv2.FONT_HERSHEY_COMPLEX          ,
                          scale                             ,
                          color                             ,
                          int(np.ceil(scale_rate))          ,
                          cv2.LINE_AA                       )

    # Кропим изображение, если надо:
    if auto_crop:
        img = autocrop(img)

    return img


def draw_contrast_text(image, text):
    '''
    Делает многострочные контрастные надписи на заданном изображении.
    '''
    # Отвязываем рабочее изображение от входного:
    image = image.copy()

    # Разбиваем текст на строки и отрисовываем каждую в отдельности:
    for line_ind, line in enumerate(text.split('\n'), 1):

        # Рисуем тёмную обводку вокруг строки:
        for i in [-1, 1]:
            for j in [-1, 1]:
                image = cv2.putText(image, line, (i, j + 20 * line_ind),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (0, 0, 0), 1, cv2.LINE_AA)

        # Рисуем саму белую строку:
        image = cv2.putText(image, line, (0, 0 + 20 * line_ind),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)

    return image


def resize_with_pad(image                    ,
                    new_shape                ,
                    padding_color = (0, 0, 0)):
    '''
    Масштабирование с сохранением соотношения сторон (используется паддинг).
    '''
    # Определяем размер исходного изображения:
    original_shape = image.shape[:2]

    # Определяем коэффициент увеличения исходного изображения:
    ratio = min(new_shape[0] / original_shape[0],
                new_shape[1] / original_shape[1])

    # Определяем целевой размер изображения без паддинга:
    new_size = [np.round(x * ratio, 0).astype(int) for x in original_shape]

    # Масштабируем исходное изображение с сохранением соотношения сторон:
    image = cv2.resize(image, new_size[::-1], interpolation=cv2.INTER_AREA)

    # Определяем размер рамки:
    delta_h = new_shape[0] - new_size[0]
    delta_w = new_shape[1] - new_size[1]
    top    = delta_h // 2
    bottom = delta_h - top
    left   = delta_w // 2
    right  = delta_w - left

    # Возвращаем изображение с паддингом:
    return cv2.copyMakeBorder(image, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=padding_color)


def rounded_rectangle(img,
                      top_left, bottom_right,
                      radius=1, color=255,
                      thickness=1, line_type=cv2.LINE_AA):
    '''
    Рисует прямоугольник со скруглёнными углами:
    Переписано https://stackoverflow.com/a/60210706 с помощью DeepSeek.
    '''

    top = min(top_left[0], bottom_right[0])
    bottom = max(top_left[0], bottom_right[0])
    left = min(top_left[1], bottom_right[1])
    right = max(top_left[1], bottom_right[1])

    height = min(bottom - top, right - left)

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:
        if corner_radius == 0:
            cv2.rectangle(img,
                          (left, top),
                          (right, bottom),
                          color, thickness, line_type)
        else:
            # рисуем заполненные прямоугольники для основных частей
            cv2.rectangle(img,
                          (left + corner_radius, top),
                          (right - corner_radius, bottom),
                          color, -1, line_type)
            cv2.rectangle(img,
                          (left, top + corner_radius),
                          (right, bottom - corner_radius),
                          color, -1, line_type)

            # рисуем заполненные эллипсы для углов
            cv2.ellipse(img,
                        (left + corner_radius, top + corner_radius),
                        (corner_radius, corner_radius),
                        180, 0, 90,
                        color, -1, line_type)
            cv2.ellipse(img,
                        (right - corner_radius, top + corner_radius),
                        (corner_radius, corner_radius),
                        270, 0, 90,
                        color, -1, line_type)
            cv2.ellipse(img,
                        (left + corner_radius, bottom - corner_radius),
                        (corner_radius, corner_radius),
                        90, 0, 90,
                        color, -1, line_type)
            cv2.ellipse(img,
                        (right - corner_radius, bottom - corner_radius),
                        (corner_radius, corner_radius),
                        0, 0, 90,
                        color, -1, line_type)

    else:
        # рисуем контур
        if corner_radius == 0:
            cv2.rectangle(img,
                          (left, top),
                          (right, bottom),
                          color, thickness, line_type)
        else:
            # рисуем прямые линии для сторон
            cv2.line(img,
                     (left + corner_radius, top),
                     (right - corner_radius, top),
                     color, thickness, line_type)
            cv2.line(img,
                     (left + corner_radius, bottom),
                     (right - corner_radius, bottom),
                     color, thickness, line_type)
            cv2.line(img,
                     (left, top + corner_radius),
                     (left, bottom - corner_radius),
                     color, thickness, line_type)
            cv2.line(img,
                     (right, top + corner_radius),
                     (right, bottom - corner_radius),
                     color, thickness, line_type)

            # рисуем дуги для углов
            cv2.ellipse(img,
                        (left + corner_radius, top + corner_radius),
                        (corner_radius, corner_radius),
                        180, 0, 90,
                        color, thickness, line_type)
            cv2.ellipse(img,
                        (right - corner_radius, top + corner_radius),
                        (corner_radius, corner_radius),
                        270, 0, 90,
                        color, thickness, line_type)
            cv2.ellipse(img,
                        (left + corner_radius, bottom - corner_radius),
                        (corner_radius, corner_radius),
                        90, 0, 90,
                        color, thickness, line_type)
            cv2.ellipse(img,
                        (right - corner_radius, bottom - corner_radius),
                        (corner_radius, corner_radius),
                        0, 0, 90,
                        color, thickness, line_type)
    return img


class Img2Film:
    '''
    Обрамляет изображения схематическим кадром из фотоплёнки.
    '''

    def __init__(self,
                 dpmm: 'Число пикселей на мм' = 300 / 25.4,  # dpi=600
                 inner_size: 'Размер внутреннего изображения' = (24, 36),
                 outer_size: 'Размер всего кадра плёнки' = (35, 37),
                 num_holes: 'Число перфораций с каждой стороны' = 8):
        self.dpmm = dpmm

        # Рассчитываем внутренние и итоговые размеры в пикселях:
        self.inner_size = (np.array(inner_size) * dpmm).astype(int)
        self.outer_size = (np.array(outer_size) * dpmm).astype(int)

        if (self.outer_size <= self.inner_size).any():
            raise ValueError('Изображение должно помещаться на плёнку!')

        # Рассчитываем сдвиг внутреннего изображения относительно итогового:
        self.shift = (self.outer_size - self.inner_size) // 2

        # Число перфораций:
        self.num_holes = num_holes

        # Остальные параметры инициируем через сброс:
        self.reset()

    def reset(self):
        self.target_size = None
        self.dtype = None
        self.film = None

    def _init_film(self, img):
        '''
        Создаёт шаблон итогового изображения.
        '''
        # Определяем размер и тип итогового изображения:
        self.target_size = list(self.outer_size) + list(img.shape[2:])
        self.dtype = img.dtype

        # Инициируем шаблон:
        self.film = np.zeros(self.target_size, dtype=self.dtype)

        # Определяем параметры перфораций:
        color = 255 if self.dtype == np.uint8 else 1.
        if len(self.target_size) == 3 and self.target_size[2] == 3:
            color = (color, color, color)
        radius = 0.7
        rel_cy1 = self.shift[0] / 2 / self.outer_size[0]
        rel_cy2 = 1. - rel_cy1
        rel_hw = 1 / 5 / self.num_holes
        rel_hh = rel_cy1 / 2
        top1 = int((rel_cy1 - rel_hh) * self.target_size[0])
        top2 = int((rel_cy2 - rel_hh) * self.target_size[0])
        bottom1 = int((rel_cy1 + rel_hh) * self.target_size[0])
        bottom2 = int((rel_cy2 + rel_hh) * self.target_size[0])

        # Создаём перфорацию:
        for ind in range(self.num_holes):
            rel_cx = (2 * ind + 1) / 2 / self.num_holes
            left = int((rel_cx - rel_hw) * self.target_size[1])
            right = int((rel_cx + rel_hw) * self.target_size[1])

            self.film = rounded_rectangle(self.film,
                                          (top1, left),
                                          (bottom1, right),
                                          radius, color, -1)
            self.film = rounded_rectangle(self.film,
                                          (top2, left),
                                          (bottom2, right),
                                          radius, color, -1)

    def apply2img(self, img):
        '''
        Обрамляет в кадр плёнки заданное изображение.
        '''
        # Масштабируем изображение до внутреннего размера:
        in_img = cv2.resize(img, self.inner_size[::-1],
                            interpolation=cv2.INTER_AREA)

        # Если текущий шаблон не соответствует необходимому или не создан, то
        # воссоздаём его:
        if in_img.dtype != self.dtype or \
                (self.target_size[:2] != list(self.film.shape[:2])) or \
                (self.target_size[2:] != list(in_img.shape[2:])):
            self._init_film(in_img)

        # Берём заготовку итогового изображения:
        out_img = self.film.copy()

        # Вписываем внутреннее изображение в итоговое:
        out_img[self.shift[0]: self.shift[0] + in_img.shape[0],
                self.shift[1]: self.shift[1] + in_img.shape[1],
                ...] = in_img

        return out_img

    def __call__(self, img):

        # Читаем изображение, если передано имя файла:
        if isinstance(img, str):
            img = cv2.imread(img)

            # BGR -> RGB
            if img.ndims == 3:
                img = img[..., ::-1]

        return self.apply2img(img)


def img_dir2video(img_dir,
                  video_file='preview.avi',
                  tmp_file=None,
                  desc=None,
                  imsize=None,
                  fps=5,
                  intra_frame_compression_only=False,
                  rm_after_add=False):
    '''
    Сборка видео из всек изображений в папке.
    Может объединять изображения разных размеров,
    сохраняя соотношение сторон за счёт паддинга.
    Используется для превью.
    '''
    # Получаем сортированный по имени список изображений:
    images = sorted(get_file_list(img_dir, extentions=cv2_img_exts))

    # Если папка не содержит изображений - возвращаем None:
    if len(images) == 0:
        return

    # Если конечный размер не задан, берём его из первого кадра:
    if imsize is None:
        imsize = cv2.imread(images[0]).shape[:2]

    # Принудительный перевод размера кадра в кортеж:
    imsize = tuple(imsize)

    if tmp_file is None:
        video_file_name, video_file_ext = os.path.splitext(video_file)
        tmp_file = video_file_name + '_tmp' + video_file_ext

    # Если изображений нет, то выходим:
    if len(images) == 0:
        return os.path.abspath(video_file)

    # Инициируем видеофайл:
    out = cv2.VideoWriter(tmp_file,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          fps,
                          imsize[::-1])

    # Для каждого кадра в папке:
    for file in tqdm(images, desc=desc, disable=desc is None):

        # Пытаемся считать, масштабировать и записать новый кадр:
        try:
            # Читаем очередной кадр:
            img = cv2.imread(file)

            # Масштабируем кадр, если надо:
            if img.shape[:2] != imsize:
                img = resize_with_pad(img, imsize)

            # Пишем кадр в видеофайл:
            out.write(img)

            # Удаляем файл, если нужно:
            if rm_after_add:
                rmpath(file)

        # Пропускаем кадр, если что-то пошло не так:
        except Exception as e:
            print(f'Пропущена запись кадра "{file}" в видео "{video_file}"!')
            print(e)
            continue

    # Закрываем записанный видеофайл:
    out.release()

    # Если нужно оставить лишь внутрикадровое сжатие, то просто переименовываем
    # файл:
    if intra_frame_compression_only:
        os.rename(tmp_file, video_file)

    # Если нужно и межкадровое сжатие, запускаем рекомпрессию:
    else:
        # Формируем команду конвертации:
        cmd_line = f'ffmpeg -i "{tmp_file}" -y -hide_banner ' + \
                   f'-c:v libx264 -crf 32 -preset slow "{video_file}"'

        '''
        # Отключаем вывод, если нужно:
        if desc is None:
            cmd_line += '>/dev/null 2>&1'
        '''
        cmd_line += '>/dev/null 2>&1'

        # Пересжимаем файл и удаляем непересжатую версию:
        os.system(cmd_line)
        rmpath(tmp_file)

    return os.path.abspath(video_file)


def get_video_info(file):
    '''
    Извлекает из видеофайла такие его параметры, как разрешение кадра,
    общее число кадров и число кадров в секунду.
    '''

    ext = os.path.splitext(file)[1]
    if ext not in cv2_vid_exts:
        raise ValueError(f'Непоодерживаемый тип файла: "{ext[1:]}"!')

    # Извлекаем основные параметры видео:
    info = {}
    cap = cv2.VideoCapture(file)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {file}")
    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info['length'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info['fps'] = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return info


class ImReadBuffer:
    '''
    Класс, буферизирующий чтение кадра из видео или фото.
    Сохраняет уже открытые файлы во внутренних состояниях, что позволяет
    ускорять процесс извлечения новых частей видеопоследовательности или
    повторного чтения изображений.
    '''

    # Сбрасывает внутенние состояния экземпляра класса:
    def reset_state(self):
        self.file      = None  # Текущий открытый файл
        self.vcap      = None  # Объект открытого видеофайла
        self.img       = None  # Текущий загруженный кадр
        self.frame_num = None  # Номер текущего загруженного кадра (для видео)

    def __init__(self):
        # Задаём внутенние состояния по умолчанию:
        self.reset_state()

    # Переводит BGR в RGB, если изображение цветной:
    @staticmethod
    def bgr2rgb(img):
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __call__(self, file, frame=0, save2file=None):

        # Если передана папка, берём все вложенные изображения:
        if isinstance(file, str) and os.path.isdir(file):
            file = [os.path.join(file, _) for _ in sorted(os.listdir(file)) if
                    os.path.splitext(_)[-1].lower() in cv2_img_exts]

        # Если передан не один файл, а целый набор изображений:
        if isinstance(file, (list, tuple)):

            # Берём тот файл, который соответствует номеру кадра:
            file = file[frame]

        # Определяем тип файла:
        file_ext = os.path.splitext(file)[-1].lower()

        # Если файл является изображением:
        if file_ext in cv2_img_exts:

            # Если текущее изображение ещё не загружалось, то загружаем:
            if file != self.file:
                self.close()  # Сбрасываем внутренние состояния
                self.img = cv2.imread(file)
                self.file = file

            # Если изображение надо сохранять:
            if save2file is not None:

                # Если тип исходного и конечного файла одинаков, то просто
                # копируем без пересжатия:
                if file_ext == os.path.splitext(save2file)[-1].lower():
                    copyfile(self.file, save2file)

                # Если типы не совпадают, придётся сохранять с пересжатием:
                else:
                    cv2.imwrite(save2file, self.img)

        # Если файл является видеопоследовательностью:
        elif file_ext in cv2_vid_exts:

            # Если текущее видео ещё не загружалось, или номер ...
            # ... загруженного кадра больше номера текущего кадра:
            if file != self.file or frame < self.frame:
                self.close()  # Сбрасываем внутренние состояния

                self.vcap = cv2.VideoCapture(file)
                if not self.vcap.isOpened():
                    raise ValueError(f'Невозможно открыть файл "{file}"!')

                self.file = file
                self.frame = -1

            # Ищем нужный кадр:
            while self.frame < frame:

                # Читаем очередной кадр:
                self.img = self.vcap.read()[1]

                # Инкримент номера кадра
                self.frame += 1

            # Сохраняем изображение, если надо:
            if save2file is not None:
                cv2.imwrite(save2file, self.img)

        # Если файл не является изображением или видео:
        else:
            raise TypeError(
                f'Файл "{file}" не является ни изображением, ни видео!')

        # Возвращаем загруженное изображение:
        if self.img is None:
            return
        else:
            return self.bgr2rgb(self.img)

    # Освобождает ресурсы:
    def close(self):
        if self.vcap:
            self.vcap.release()  # Закрываем открытый видеофайл
            self.reset_state()   # Сбрасываем внутенние состояния

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


###############################
# Работа с файловой системой: #
###############################


def mkdirs(path):
    '''
    Создаёт все нужные дирректории заданного пути.
    Возвращает False, если путь уже существовал.
    '''

    # Если передана пустая строка, здачит имелся ввиду текущий католог,
    # который уже создан:
    if not path:
        return False

    if not os.path.exists(path) or not os.path.isdir(path):
        try:
            os.makedirs(path)
            return True

        except PermissionError:
            raise PermissionError(f'Недостаточно прав создать "{path}"!')

    return False


def rmpath(paths, desc=None):
    '''
    Удаляет файл(ы) или папку(и) вместе с её(их) содержимым.
    Возвращает False, если путь не существовал.
    '''

    # Если передан целый список/кортеж/множество путей, удаляем каждый:
    if isinstance(paths, (list, tuple, set)):
        return mpmap(rmpath, paths, desc=desc, num_procs=1)

    # Если путь всего один, работаем с ним:
    path = paths

    # Ничего не делаем, если файла просто нет:
    if not os.path.exists(path):
        return False

    try:
        with AnnotateIt(desc):
            # Если это папка:
            if os.path.isdir(path):
                for file in get_file_list(path):
                    os.remove(file)
                rmtree(path)

            # Если это Файл:
            elif os.path.isfile(path):
                os.remove(path)

            else:
                msg = f'Не файл и не папка "{path}"!'
                raise ValueError(msg)

            return True

    except PermissionError:
        raise PermissionError(f'Недостаточно прав удалить "{path}"!')


def emptydir(path):
    '''
    Создаёт папку, если её не было или
    очищает всё её содержимое, если она была.
    '''
    rmpath(path)
    mkdirs(path)


def get_empty_dir(path: str | Path = None, clear: bool = False):
    '''
    Возвращает путь до пустой папки.
    Она может быть либо задана явно,
    либо будет создана автоматически
    по временному пути.

    Если путь задан явно, но папка не пуста, то
    она очищается, если clear == True, иначе
    возвращается ошибка.
    '''
    # Создаём временную дирректорию, если путь не указан:
    if path is None:
        path = tempfile.mkdtemp()

    # Если папка не создана - создаём:
    elif not os.path.isdir(path):
        mkdirs(path)

    # Если папка не пуста:
    elif os.listdir(path):

        # Очищаем содержимое, если можно:
        if clear:
            emptydir(path)

        # Если очищать нельзя - возвращаем исключение:
        else:
            raise ValueError(f'Папка "{path}" не пуста!')

    return path


def first_existed_path(paths):
    '''
    Возвращяет первый существующий путь из списка путей.
    '''
    for path in paths:
        if os.path.exists(path):
            return path


def unzip_file(zip_file, unzipped_files_subdir):
    '''
    Распаковывает заданный zip-файл в заданную дирректорию.
    '''
    # Распаковываем:
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(unzipped_files_subdir)

    return unzipped_files_subdir


def unzip_dir(zipped_files_dir    : 'Путь к папке с *.zip-файлами, подлежащими распаковке'                                  ,
              unzipped_files_dir  : 'Путь к папке, в поддиректории которой будут помещяться распакованные файлы'            ,
              desc                : 'Строка, выводящаяся в статусбаре (если None, то статусбар не выводится)' = 'Распаковка',
              use_multiprocessing : 'Многопроцессорный режим'                                                 = True        ):
    '''
    Распаковывает все zip-файлы в папке f"{zipped_files_dir}" (без рекурсии)
    в папку f"{unzipped_files_dir}" в подпапки с именем соответствующих архивов.
    '''

    # Список путей к распакованным папкам:
    unzipped_files_dirs = []

    # Формируем писок задач (zip-файлов и целевых дирректорий):
    zip_files = []
    unzipped_files_subdirs = []
    for zip_file in os.listdir(zipped_files_dir):

        # Получяем имя и расширение файла:
        name, ext = os.path.splitext(zip_file)

        # Пропускаем не zip-архивы:
        if ext not in ('.zip',):
            continue

        # Полный путь до архива:
        zip_file = os.path.join(zipped_files_dir, zip_file)

        # Путь до дирректории для распаковки:
        unzipped_files_subdir = os.path.join(unzipped_files_dir, name)

        # Добавляем пути в списки:
        zip_files.append(zip_file)
        unzipped_files_subdirs.append(unzipped_files_subdir)

    # Распаковываем во временную папку:
    unzipped_files_dirs = mpmap(unzip_file, zip_files, unzipped_files_subdirs,
                                num_procs=not use_multiprocessing, desc=desc)

    return unzipped_files_dirs


class Zipper:
    '''
    Позволяет сжимать один или несколько файлов или распаковать zip-архивы.
    При успешном выполнении возвращает путь к созданному архиву (при сжатии)
    или путь к распакованным файлам (при извлечении).

    Первый опыт вайб-кодинга. Использован DeepSeek R1.
    '''

    @staticmethod
    def flag(bool_val: bool | None, default: bool):
        '''
        Возвращает bool_val, если он не None, иначе default
        '''
        return bool_val if bool_val is not None else default

    def __init__(
        self,
        unzipped: str | Iterable[str] = '',
        zipped: str = '',
        remove_unzipped: bool | None = None,
        rewrtie_unzipped: bool | None = None,
        remove_zipped: bool | None = None,
        rewrtie_zipped: bool | None = None,
        remove_source: bool = False,
        rewrite_target: bool = False,
        compress_desc: str = '',
        extract_desc: str = '',
        desc: str = ''
    ):
        # Проверка наличия хотя бы одного параметра:
        if not unzipped and not zipped:
            raise ValueError('Должен быть задан хотя бы один из параметров!')

        self.unzipped = unzipped
        self.zipped = zipped
        self.remove_unzipped = self.flag(remove_unzipped, remove_source)
        self.rewrtie_unzipped = self.flag(rewrtie_unzipped, rewrite_target)
        self.remove_zipped = self.flag(remove_zipped, remove_source)
        self.rewrtie_zipped = self.flag(rewrtie_zipped, rewrite_target)
        self.compress_desc = self.flag(compress_desc, desc)
        self.extract_desc = self.flag(extract_desc, desc)

        # Связываем методы экземпляра с внутренними реализациями:
        self.compress = self.__compress
        self.extract = self.__extract

    @staticmethod
    def _source_to_list(source: str | Iterable[str]) -> list[str]:
        '''Преобразует различные форматы источников в список файлов'''
        # Обработка строки с возможными шаблонами (wildcards):
        if isinstance(source, str):
            if any(char in source for char in '*?['):
                paths = glob.glob(source, recursive=True)
                if not paths:
                    raise FileNotFoundError(f'Маска "{source}" не найдена!')
                return paths
            return [source]  # Одиночный файл/папка

        # Обработка итератора (списка, кортежа и т.д.):
        paths = list(source)
        if not paths:
            raise ValueError('Список файлов пуст!')
        return paths

    @staticmethod
    def _get_base_path(paths: list[str]) -> str:
        '''Определяет базовый путь для группы файлов'''

        abs_paths = list(map(os.path.abspath, paths))

        # Пытаемся найти общий путь для всех файлов:
        try:
            base = os.path.commonpath(abs_paths)

        # Если пути на разных дисках - берем первый путь:
        except ValueError:
            base = abs_paths[0]

        return base if os.path.isdir(base) else os.path.dirname(base)

    @staticmethod
    def _get_default_archive_path(source_list: list[str]) -> str:
        '''
        Определяет путь по умолчанию для архива:
        - Для одного файла: /path/to/file.txt → /path/to/file.txt.zip
        - Для нескольких файлов/папок: /path/to/common → /path/to/common.zip
        '''
        if len(source_list) == 1 and os.path.isfile(source_list[0]):
            return os.path.abspath(source_list[0]) + '.zip'
        base = Zipper._get_base_path(source_list)
        return base + '.zip'

    @classmethod
    def _compress(
        cls,
        source: str | Iterable[str],
        target: str = '',
        remove_source: bool = False,
        rewrite_target: bool = False,
    ) -> str | bool:
        '''
        Основной метод для сжатия файлов/папок в ZIP-архив

        Этапы работы:
        1. Преобразование входных данных в список файлов
        2. Проверка существования всех исходных файлов
        3. Определение пути для архива (если не задан)
        4. Проверка конфликтов существующего архива
        5. Создание архива с сохранением структуры папок
        6. Удаление исходных файлов (если требуется)
        7. Возврат пути к созданному архиву

        Логика упаковки директорий:
        - Если передан путь к одной директории: сохраняет директорию целиком
          с ее именем в корне архива
        - Если передана маска или список файлов: сохраняет файлы без
          родительской директории

        Возвращает:
        - Путь к созданному архиву при успехе
        - False при ошибке
        '''
        try:
            # Шаг 1 - Преобразование источника в список файлов:
            source_list = cls._source_to_list(source)

            # Шаг 2 - Проверка существования всех файлов в списке:
            for path in source_list:
                if not os.path.exists(path):
                    raise FileNotFoundError(f'"{path}" не найден!')

            # Шаг 3 - Определение пути для архива, если не задан:
            if not target:
                target = cls._get_default_archive_path(source_list)

            # Шаг 4 - Проверка существования архива и политики перезаписи:
            if os.path.exists(target) and not rewrite_target:
                raise FileExistsError(f'"{target}" уже существует!')

            # Шаг 5 - Создание ZIP-архива с разной логикой для директорий:
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Особый случай: одна директория - сохраняем ее целиком
                if len(source_list) == 1 and os.path.isdir(source_list[0]):
                    dir_path = os.path.abspath(source_list[0])
                    dir_name = os.path.basename(dir_path)

                    # Рекурсивный обход директории
                    for root, _, files in os.walk(dir_path):
                        for file in files:
                            full_path = os.path.join(root, file)

                            # Формируем путь в архиве:
                            arcname = os.path.join(
                                dir_name,
                                os.path.relpath(full_path, start=dir_path)
                            )
                            zipf.write(full_path, arcname)
                else:
                    # Для файлов/масок/списков - сохраняем без родительской
                    # директории:
                    base_path = cls._get_base_path(source_list)

                    for path in source_list:
                        abs_path = os.path.abspath(path)

                        # Обработка директорий (рекурсивный обход):
                        if os.path.isdir(abs_path):
                            for root, _, files in os.walk(abs_path):
                                for file in files:
                                    full_path = os.path.join(root, file)

                                    # Вычисление относительного пути для
                                    # архива:
                                    arcname = os.path.relpath(
                                        full_path,
                                        start=base_path
                                    )
                                    zipf.write(full_path, arcname)

                        # Обработка одиночных файлов:
                        else:
                            arcname = os.path.relpath(abs_path,
                                                      start=base_path)
                            zipf.write(abs_path, arcname)

            # Шаг 6 - Удаление исходных файлов при включенной опции:
            if remove_source:

                # Удаляем в обратном порядке (вложенные элементы сначала):
                for path in reversed(source_list):
                    abs_path = os.path.abspath(path)
                    if os.path.isdir(abs_path):
                        rmtree(abs_path)  # Рекурсивное удаление папки
                    else:
                        os.remove(abs_path)  # Удаление файла

            # Шаг 7 - Возвращаем путь к созданному архиву:
            return target

        except Exception as e:
            print(f'Ошибка сжатия: {e}')
            return False

    @classmethod
    def _extract(
        cls,
        source: str,
        target: str = '',
        remove_source: bool = False,
        rewrite_target: bool = False,
    ) -> str | bool:
        '''
        Основной метод для распаковки ZIP-архива

        Этапы работы:
        1. Проверка существования архива
        2. Анализ содержимого архива (один файл или несколько)
        3. Определение целевого пути (если не задан):
            - Для архива с одним файлом в корне: директория архива
            - Для остальных случаев: директория архива (без создания подпапки)
        4. Создание целевой директории (если не существует)
        5. Распаковка с учетом типа архива (один файл/несколько)
        6. Удаление исходного архива (если требуется)
        7. Возврат пути к распакованному файлу/папке

        Особый случай:
        - Для архивов с одним файлом возвращается путь к файлу
        - Для архивов с несколькими файлами возвращается путь к папке

        Возвращает:
        - Путь к распакованным данным при успехе
        - False при ошибке
        '''
        try:
            # Шаг 1 - Проверка существования архива:
            if not os.path.exists(source):
                raise FileNotFoundError(f'Архив "{source}" не найден!')

            # Шаг 2 - Анализ содержимого архива:
            single_file_in_root = False
            if isinstance(target, str) and \
                    any(char in target for char in '*?['):
                target = os.path.dirname(target)  # ".../*/" -> ".../"
            result_path = target  # Инициализация результирующего пути

            with zipfile.ZipFile(source, 'r') as zipf:
                namelist = zipf.namelist()  # Получаем список элементов

                # Проверка на архив с одним файлом в корне:
                if len(namelist) == 1 and not namelist[0].endswith('/'):
                    single_file_in_root = True

                    # Проверка отсутствия вложенных директорий:
                    if '/' in namelist[0] or '\\' in namelist[0]:
                        single_file_in_root = False

            # Шаг 3 - Определение целевого пути:
            if not target:
                # Всегда используем директорию архива как цель по умолчанию:
                target = os.path.dirname(os.path.abspath(source))

            # Шаг 4 - Создание целевой директории:
            os.makedirs(target, exist_ok=True)

            # Шаг 5 - Распаковка с учетом типа архива:
            with zipfile.ZipFile(source, 'r') as zipf:
                if single_file_in_root:
                    filename = namelist[0]  # Имя единственного файла

                    # Распаковка во временную папку:
                    zipf.extract(filename, target)

                    # Формирование путей:
                    extracted_path = os.path.join(target, filename)
                    final_path = os.path.join(
                        target, os.path.basename(filename)
                    )

                    # Перемещение файла в целевую директорию при
                    # необходимости:
                    if extracted_path != final_path:
                        move(extracted_path, final_path)
                        result_path = final_path
                    else:
                        result_path = extracted_path

                # Стандартная распаковка всего архива:
                else:
                    zipf.extractall(target)
                    result_path = target  # Возвращаем путь к папке

            # Шаг 6 - Удаление исходного архива при включенной опции:
            if remove_source:
                os.remove(source)

            # Шаг 7 - Возвращаем путь к результату:
            return result_path

        except Exception as e:
            print(f'Ошибка распаковки: {e}')
            return False

    # Статические методы-обертки для прямого вызова:
    @classmethod
    def compress(
        cls,
        unzipped: str | Iterable[str],
        zipped: str = '',
        remove_source: bool = False,
        rewrite_target: bool = False,
        desc: str = ''
    ) -> str | bool:
        with AnnotateIt(desc):
            return cls._compress(
                source=unzipped,
                target=zipped,
                remove_source=remove_source,
                rewrite_target=rewrite_target,
            )

    @classmethod
    def extract(
        cls,
        zipped: str,
        unzipped: str = '',
        remove_source: bool = False,
        rewrite_target: bool = False,
        desc: str = ''
    ) -> str | bool:
        with AnnotateIt(desc):
            return cls._extract(
                source=zipped,
                target=unzipped,
                remove_source=remove_source,
                rewrite_target=rewrite_target,
            )

    # Методы экземпляра класса:
    def __compress(self) -> str | bool:
        with AnnotateIt(self.compress_desc):
            return self._compress(
                source=self.unzipped,
                target=self.zipped,
                remove_source=self.remove_unzipped,
                rewrite_target=self.rewrtie_zipped,
            )

    def __extract(self) -> str | bool:
        with AnnotateIt(self.extract_desc):
            return self._extract(
                source=self.zipped,
                target=self.unzipped,
                remove_source=self.remove_zipped,
                rewrite_target=self.rewrtie_unzipped,
            )


class ImportVisitor(ast.NodeVisitor):
    """
    Анализатор AST для извлечения информации об импортах Python-кода.

    Атрибуты:
        unconditional_imports (set): Модули, импортированные на верхнем уровне
        conditional_imports (set): Модули, импортированные внутри блоков кода
        depth (int): Текущий уровень вложенности при обходе AST
    """

    def __init__(self):
        """Инициализирует анализатор, сбрасывая состояние."""
        self.reset()

    def reset(self):
        """
        Сбрасывает состояние анализатора перед новым анализом.

        Обнуляет:
        - Множества безусловных и условных импортов
        - Счетчик глубины вложенности
        """
        self.unconditional_imports = set()  # Импорты верхнего уровня
        self.conditional_imports = set()    # Импорты внутри блоков кода
        self.depth = 0                      # Уровень вложенности (0 = верхний уровень)

    def analyze_file(self, file_path):
        """
        Основной метод для анализа файла.

        Параметры:
            file_path (str): Путь к анализируемому .py файлу

        Возвращает:
            tuple: (unconditional_imports, conditional_imports) - два множества строк

        Исключения:
            FileNotFoundError: Файл не существует
            UnicodeDecodeError: Ошибка чтения файла
            SyntaxError: Синтаксическая ошибка в коде

        Процесс:
            1. Сбрасывает предыдущее состояние анализатора
            2. Читает содержимое файла
            3. Парсит AST
            4. Обходит AST для сбора информации об импортах
        """
        # Сброс предыдущего состояния
        self.reset()

        # Чтение файла с обработкой ошибок
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        except UnicodeDecodeError:
            raise UnicodeDecodeError(f"Ошибка декодирования файла: {file_path}")

        # Парсинг AST и обработка синтаксических ошибок
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SyntaxError(f"Синтаксическая ошибка в файле {file_path}: {e}")

        # Запуск обхода AST
        self.visit(tree)

        return self.unconditional_imports, self.conditional_imports

    def analyze_string(self, source_code):
        """
        Анализирует код из строки вместо файла.

        Параметры:
            source_code (str): Строка с Python-кодом для анализа

        Возвращает:
            tuple: (unconditional_imports, conditional_imports) - два множества строк

        Исключения:
            SyntaxError: Синтаксическая ошибка в коде

        Полезно для:
            - Анализа фрагментов кода
            - Юнит-тестирования
            - Интерактивного использования
        """
        self.reset()
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise SyntaxError(f"Синтаксическая ошибка в коде: {e}")
        self.visit(tree)
        return self.unconditional_imports, self.conditional_imports

    # --- Методы обработки узлов AST ---

    def visit_Import(self, node):
        """
        Обрабатывает обычные импорты (import ...).

        Примеры:
            import os
            import pandas as pd

        Логика:
            - Извлекает корневое имя модуля (первую часть)
            - Определяет контекст (верхний уровень или блок)
            - Добавляет в соответствующее множество
        """
        for alias in node.names:
            # Извлекаем корневое имя модуля (до первой точки)
            module_name = alias.name.split('.')[0]

            # Разделяем импорты по уровню вложенности
            if self.depth == 0:  # Верхний уровень
                self.unconditional_imports.add(module_name)
            else:  # Внутри блока кода
                self.conditional_imports.add(module_name)

    def visit_ImportFrom(self, node):
        """
        Обрабатывает импорты вида 'from ... import ...'.

        Примеры:
            from sys import exit
            from sklearn.model_selection import train_test_split

        Особенности:
            - Игнорирует относительные импорты (с точками)
            - Не обрабатывает импорты без указания модуля
        """
        # Пропускаем относительные импорты (level > 0)
        if node.level > 0:
            return

        # Проверяем наличие имени модуля
        if node.module:
            # Извлекаем корневое имя модуля
            module_name = node.module.split('.')[0]

            # Разделяем по уровню вложенности
            if self.depth == 0:
                self.unconditional_imports.add(module_name)
            else:
                self.conditional_imports.add(module_name)

    # --- Управление контекстом вложенности ---

    def _enter_context(self):
        """Увеличивает счетчик глубины при входе в новый контекст."""
        self.depth += 1

    def _exit_context(self):
        """Уменьшает счетчик глубины при выходе из контекста."""
        self.depth -= 1

    # --- Методы обработки контекстных узлов ---
    # Эти методы обеспечивают отслеживание вложенности кода
    def visit_FunctionDef(self, node):
        """Обрабатывает определение обычной функции (def)."""
        self._enter_context()     # Вход в контекст функции
        self.generic_visit(node)  # Продолжение обхода дочерних узлов
        self._exit_context()      # Выход из контекста
    # Остальные элементы обрабатываются аналогично:
    visit_AsyncFunctionDef = visit_ClassDef = visit_If = \
        visit_For = visit_AsyncFor = visit_While = visit_With = \
        visit_AsyncWith = visit_Try = visit_TryFinally = \
        visit_TryExcept = visit_AsyncFunctionDef = visit_FunctionDef

    # --- Дополнительные методы ---

    def get_combined_imports(self):
        """Возвращает объединенное множество всех импортов."""
        return self.unconditional_imports | self.conditional_imports

    def get_stats(self):
        """Возвращает статистику по найденным импортам."""
        return {
            'total_unconditional': len(self.unconditional_imports),
            'total_conditional': len(self.conditional_imports),
            'total_combined': len(self.get_combined_imports()),
            'all_imports': sorted(self.get_combined_imports())
        }

    def build_dependency_graph(self, files):
        """
        Строит направленный граф зависимостей между модулями.

        Параметры:
            files (list): Список путей к .py файлам

        Возвращает:
            dict: Граф в формате {
                module: {
                    'unconditional': set(модулей),
                    'conditional': set(модулей)
                }
            }

        Процесс:
            1. Извлекает имена модулей из путей файлов
            2. Для каждого файла анализирует импорты
            3. Строит граф, учитывая только модули из списка
            4. Различает условные и безусловные зависимости
        """
        # Получаем имена модулей (без расширения .py)
        module_names = set()
        for file_path in files:
            base = os.path.basename(file_path)
            if base.endswith('.py'):
                module_name = base[:-3]
            else:
                module_name = base
            module_names.add(module_name)

        # Инициализация графа
        graph = {module: {'unconditional': set(), 'conditional': set()}
                 for module in module_names}

        # Анализ каждого файла
        for file_path in files:
            base = os.path.basename(file_path)
            current_module = base[:-3] if base.endswith('.py') else base

            try:
                # Получаем импорты файла
                unconditional, conditional = self.analyze_file(file_path)
            except Exception:
                continue  # Пропустить файл при ошибке

            # Добавляем безусловные зависимости
            for imp in unconditional:
                if imp in module_names:  # Только модули из списка
                    graph[current_module]['unconditional'].add(imp)

            # Добавляем условные зависимости
            for imp in conditional:
                if imp in module_names:  # Только модули из списка
                    graph[current_module]['conditional'].add(imp)

        return graph

    @staticmethod
    def _filter_edges_by_longest_paths(edges):
        """
        Обрабатывает рёбра одного типа, оставляя только те, что принадлежат
        максимальным путям.

        Новая логика:
            - Удаляет короткие пути при наличии более длинных альтернативных
              путей между теми же узлами
            - Сохраняет независимые пути максимальной длины
            - Ребро удаляется, если существует более длинный путь между его
              началом и концом

        Параметры:
            edges (list): Список рёбер вида [(source, target)]

        Возвращает:
            set: Множество рёбер, принадлежащих максимальным путям
        """

        # Строим граф и определяем все узлы
        graph = defaultdict(list)
        all_nodes = set()
        for u, v in edges:
            graph[u].append(v)
            all_nodes.add(u)
            all_nodes.add(v)

        # Вычисляем максимальную длину пути между всеми парами узлов
        max_path_lengths = {}
        for node in sorted(all_nodes):  # Стабильный порядок
            queue = deque([(node, 0)])  # (current_node, path_length)
            visited = {node: 0}  # node: max_path_length

            while queue:
                current, length = queue.popleft()
                for neighbor in graph.get(current, []):
                    new_length = length + 1
                    # Обновляем только если нашли более длинный путь
                    if neighbor not in visited or new_length > visited[neighbor]:
                        visited[neighbor] = new_length
                        queue.append((neighbor, new_length))

            # Сохраняем результаты для стартового узла
            for target, path_len in visited.items():
                if node != target:  # Исключаем петли
                    max_path_lengths[(node, target)] = path_len

        # Фильтруем рёбра: оставляем только те, которые являются частью максимального пути
        kept_edges = set()
        for (u, v) in edges:
            # Проверяем, является ли это ребро частью максимального пути
            if (u, v) in max_path_lengths:
                current_length = 1  # Длина прямого ребра
                max_length = max_path_lengths[(u, v)]

                # Если существует более длинный путь между u и v, удаляем прямое ребро
                if max_length > current_length:
                    continue
                kept_edges.add((u, v))
            else:
                # Если нет другого пути, сохраняем ребро
                kept_edges.add((u, v))

        return kept_edges

    def filter_longest_paths(self, graph):
        """
        Оставляет только рёбра, входящие в самые длинные пути.

        Параметры:
            graph (dict): Граф зависимостей

        Возвращает:
            dict: Отфильтрованный граф

        Особенности:
            - Условные и безусловные рёбра обрабатываются раздельно
            - Для каждого типа рёбер вычисляются максимальные пути
            - Сохраняются только рёбра, принадлежащие этим путям
        """
        # Копируем граф для модификации
        new_graph = {module: {'unconditional': set(), 'conditional': set()}
                     for module in graph}

        # Обработка безусловных рёбер
        unconditional_edges = []
        for source, data in graph.items():
            for target in data['unconditional']:
                unconditional_edges.append((source, target))
        kept_unconditional = self._filter_edges_by_longest_paths(unconditional_edges)

        # Обработка условных рёбер
        conditional_edges = []
        for source, data in graph.items():
            for target in data['conditional']:
                conditional_edges.append((source, target))
        kept_conditional = self._filter_edges_by_longest_paths(conditional_edges)

        # Заполнение нового графа
        for (source, target) in kept_unconditional:
            new_graph[source]['unconditional'].add(target)
        for (source, target) in kept_conditional:
            new_graph[source]['conditional'].add(target)

        return new_graph

    def _sort_edges_for_minimal_crossings(self, edges, safe_names, sink_nodes, graph):
        """
        Простой алгоритм сортировки рёбер:
        1. Сначала рёбра от узлов с низкими индексами (центр) к узлам с низкими индексами
        2. Затем рёбра от центра к периферии
        3. Затем рёбра от периферии к центру
        4. Затем рёбра между периферийными узлами
        """
        # Получаем порядок узлов
        node_order = self._reorder_nodes_by_importance(graph, safe_names)

        # Определяем, какие узлы считаем центральными (топ-8 по важности)
        central_nodes = set(sorted(graph.keys(), key=lambda x: node_order[x])[:8])

        # Разделяем рёбра на 4 группы
        center_to_center = []  # Центр → Центр
        center_to_edge = []    # Центр → Периферия
        edge_to_center = []    # Периферия → Центр
        edge_to_edge = []      # Периферия → Периферия

        for edge in edges:
            edge_type, source, target = edge

            source_is_center = source in central_nodes
            target_is_center = target in central_nodes

            if source_is_center and target_is_center:
                center_to_center.append(edge)
            elif source_is_center:
                center_to_edge.append(edge)
            elif target_is_center:
                edge_to_center.append(edge)
            else:
                edge_to_edge.append(edge)

        # Функция для сортировки рёбер: сначала по источнику, потом по цели
        def sort_key(edge):
            return (node_order[edge[1]], node_order[edge[2]])

        # Сортируем каждую группу
        center_to_center.sort(key=sort_key)
        center_to_edge.sort(key=sort_key)
        edge_to_center.sort(key=sort_key)
        edge_to_edge.sort(key=sort_key)

        # Объединяем в порядке: центр→центр, центр→край, край→центр, край→край
        return center_to_center + center_to_edge + edge_to_center + edge_to_edge

    def _reorder_nodes_by_importance(self, graph, safe_names):
        """
        Смешанный подход: ручные приоритеты для известных узлов +
        автоматическая сортировка для остальных
        """
        # Ручные приоритеты (чем меньше число, тем выше приоритет)
        manual_priorities = {
            'cvat': 1,
            'utils': 2,
            'pt_utils': 3,
            'ml_utils': 4,
            'cv_utils': 5,
            'video_utils': 6,
            # Остальные получат приоритет 100 + алфавитный порядок
        }

        # Считаем indegree для автоматической сортировки
        indegree = defaultdict(int)
        for source, data in graph.items():
            for target in data['unconditional']:
                indegree[target] += 1

        # Определяем важность
        importance = {}
        for node in graph:
            if node in manual_priorities:
                # Используем ручной приоритет
                importance[node] = manual_priorities[node]
            else:
                # Автоматическая сортировка: сначала по indegree, потом по алфавиту
                # Инвертируем: чем больше indegree, тем меньше значение
                importance[node] = 100 - indegree.get(node, 0) * 10

        # Сортируем узлы по важности (возрастание)
        sorted_nodes = sorted(graph.keys(), key=lambda x: (importance[x], x))

        return {node: i for i, node in enumerate(sorted_nodes)}

    def to_mermaid(self, graph, filename=None, sort=False):
        """
        Конвертирует граф в формат Mermaid.

        Параметры:
            graph (dict): Граф зависимостей
            filename (str, optional): Файл для сохранения
            sort (bool | callable): 
                - False: не сортировать
                - True: использовать встроенную сортировку
                - callable: внешняя функция сортировки, которая принимает граф
                  и возвращает (ordered_nodes, ordered_edges)
        """
        # Обрабатываем разные варианты параметра sort
        if callable(sort):
            # Внешняя функция сортировки
            try:
                ordered_nodes, ordered_edges = sort(graph)
                # Используем упорядоченные узлы от внешнего алгоритма
                sorted_nodes = ordered_nodes
                edges = ordered_edges
                use_external_sort = True
            except Exception as e:
                print(f"Ошибка во внешнем алгоритме сортировки: {e}")
                # Возвращаемся к стандартному порядку
                sorted_nodes = sorted(graph.keys())
                edges = []
                use_external_sort = False
        elif sort:
            # Встроенная сортировка
            # Получаем порядок узлов по важности
            node_order = self._reorder_nodes_by_importance(graph, None)
            sorted_nodes = sorted(graph.keys(), key=lambda x: node_order[x])
            edges = []
            use_external_sort = False
        else:
            # Без сортировки
            sorted_nodes = sorted(graph.keys())
            edges = []
            use_external_sort = False

        # Генерация безопасных имён узлов
        safe_names = {}
        for i, node in enumerate(sorted_nodes):
            safe_names[node] = f"node_{i}"

        lines = ["graph RL;"]

        # Определение стоков (модулей без исходящих зависимостей)
        all_nodes = set(graph.keys())
        sink_nodes = set()
        for node in all_nodes:
            if not graph[node]['unconditional'] and not graph[node]['conditional']:
                sink_nodes.add(node)

        # Создаем множество безусловных рёбер для проверки
        unconditional_set = set()
        for source, data in graph.items():
            for target in data['unconditional']:
                unconditional_set.add((source, target))

        # Если не использовался внешний алгоритм, собираем рёбра стандартным способом
        if not use_external_sort:
            # Собираем все рёбра
            for source, data in graph.items():
                # Безусловные зависимости
                for target in data['unconditional']:
                    if target in safe_names:
                        edges.append(('-->', source, target))

                # Условные зависимости (только если нет безусловной)
                for target in data['conditional']:
                    if target in safe_names:
                        if (source, target) in unconditional_set:
                            continue
                        edges.append(('-.->', source, target))

            # Если включена встроенная сортировка рёбер
            if sort and callable(sort) is False:
                edges = self._sort_edges_for_minimal_crossings(
                    edges, safe_names, sink_nodes, graph
                )

        # Добавление всех узлов
        for node in sorted_nodes:
            lines.append(f"    {safe_names[node]}[{node}];")

        # Создание подграфа для стоков (горизонтальное выравнивание)
        if sink_nodes:
            lines.append("    %% Выравнивание стоков на одном уровне")
            lines.append("    subgraph SinkGroup [ ]")
            lines.append("        direction LR")
            for node in sink_nodes:
                lines.append(f"        {safe_names[node]}")
            lines.append("    end")
            lines.append("    style SinkGroup fill:none,stroke:none;")

        # Добавление рёбер
        for edge_type, source, target in edges:
            lines.append(f"    {safe_names[source]} {edge_type} {safe_names[target]};")

        mermaid_content = "\n".join(lines)

        # Сохранение в файл или возврат строки
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
        else:
            return mermaid_content

    def draw_dependency_in_mermaid(self, files, filter_short_paths=False,
                                   output_file=None, sort=True):
        """
        Строит граф зависимостей модулей в формате Mermaid.

        Параметры:
            files (list): Список .py файлов
            filter_short_paths (bool): Флаг фильтрации коротких путей
            output_file (str, optional): Файл для сохранения результата
            sort (bool | callable): Сортировать рёбра да / нет / внешная функция
        """
        graph = self.build_dependency_graph(files)

        if filter_short_paths:
            graph = self.filter_longest_paths(graph)

        return self.to_mermaid(graph, filename=output_file, sort=sort)


def draw_repo_dependency_graph(
    dirname: str = '',
    sort: bool | Callable = True,
) -> str:
    '''
    Строит графы зависимостей Python-модулей в заданной папке с проектом.

    Параметры:
        dirname (str): Путь к папке с проектом
        sort (bool | callable): Сортировать рёбра да / нет / внешная функция
    '''
    # Полный список модулей библиотеки:
    dirname = dirname or os.path.dirname(__file__)
    filename = os.path.join(dirname, 'dependency_graph.mmd')
    py_files = sorted(get_file_list(dirname, '.py', False))

    # Выкидываем все файлы, начинающиеся с '_':
    py_files = [py_file for py_file in py_files
                if os.path.basename(py_file)[0] != '_']

    iv = ImportVisitor()
    iv.draw_dependency_in_mermaid(py_files, True, filename, sort)

    return filename


def obj2yaml(obj, file='./cfg.yaml', encoding='utf-8', allow_unicode=True):
    '''
    Пишет словарь, множество или кортеж в yaml-файл.
    Параметры по умолчанию позволяют сохранять кириллицу.
    '''
    with open(file, 'w', encoding=encoding) as f:
        yaml.safe_dump(obj,
                       f,
                       allow_unicode=allow_unicode,
                       sort_keys=False)

    return file


def yaml2obj(file='./cfg.yaml', encoding='utf-8'):
    '''
    Читает yaml-файл.
    '''
    with open(file, 'r', encoding=encoding) as f:
        obj = yaml.safe_load(f)

    return obj


class NpEncoder(json.JSONEncoder):
    '''
    Поддержка numpy-типов в json.dumps в obj2json.

    Взято из:
    https://stackoverflow.com/a/57915246/14474616
    '''

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def obj2json(obj, file='./cfg.json', encoding='utf-8'):
    '''
    Пишет json- и jsonl-файлы.
    '''

    # Определяем тип файла:
    ext = os.path.splitext(file)[-1].lower()
    assert ext in {'.json', '.jsonl'}

    with open(file, 'w', encoding='utf-8') as f:

        # Если обычный json:
        if ext == '.json':
            json.dump(obj, f, cls=NpEncoder)

        # Если jsonl:
        else:
            assert isinstance(obj, (tuple, list, set))
            for line in obj:
                f.write(json.dumps(line, cls=NpEncoder) + '\n')

    return file


def json2obj(file='./cfg.json', encoding='utf-8'):
    '''
    Читае json- и jsonl-файлы.
    '''

    # Определяем тип файла:
    ext = os.path.splitext(file)[-1].lower()
    assert ext in {'.json', '.jsonl'}

    with open(file, 'r', encoding=encoding) as f:

        # Если обычный json:
        if ext == '.json':
            obj = json.load(f)

        # Если jsonl:
        else:
            obj = []
            for line in f:
                obj.append(json.loads(line.strip()))

    return obj


class ManyToOne(dict):
    """Словарь-классификатор.

    Словарь, поддерживающий двунаправленное отображение между ключами и значениями.

    Реализует преобразование между двумя представлениями данных:
    1. Many-to-One (многие к одному): множество ключей отображаются в одно значение;
    2. One-to-Many (один ко многим): один ключ отображается в коллекцию значений.

    Основные сценарии использования:
    - Классификация объектов (категоризация);
    - Создание обратных индексов (reverse lookup);
    - Группировка данных по общим признакам;
    - Работа со справочниками и кодами.

    Примеры:
        # Инициализация many-to-one словарём
        classifier = ManyToOne({'яблоко': 'фрукт', 'морковь': 'овощ', 'банан': 'фрукт'})
        print(classifier['яблоко'])  # 'фрукт'

        # Инициализация one-to-many словарём
        grouping = ManyToOne(
            {'фрукт': ['яблоко', 'банан'], 'овощ': ['морковь']}, type='one2many'
        )
        print(grouping['яблоко'])  # 'фрукт'

    Наследует все методы стандартного словаря Python (dict).
    Дополнительные возможности:
    - Поддержка обратного отображения (one2many);
    - Сохранение/загрузка в различных форматах;
    - Валидация целостности данных;
    - Экспорт в Mermaid.
    """
    acceptable_types = ['one2many', 'many2one', 'auto']

    def __init__(self, source: dict | str | Path,
                 type: str = 'auto') -> None:
        """
        Инициализирует словарь из источника данных с автоматическим определением
        или явным указанием типа отображения.

        Параметры:
        ----------
        source : dict | str | Path
            Источник данных. Может быть:
            - Словарём Python
            - Путь к файлу в формате JSON, JSONL, YAML или YML

            Для типа 'many2one' ожидается словарь вида {key: value}
            Для типа 'one2many' ожидается словарь вида {value: [key1, key2, ...]}

        type : str, default='auto'
            Тип отображения в источнике:
            - 'many2one': источник содержит прямое отображение (ключ -> значение)
            - 'one2many': источник содержит обратное отображение (значение -> список ключей)
            - 'auto': тип определяется автоматически на основе структуры значений.
              Если все значения - коллекции (list, tuple, set), то 'one2many',
              иначе - 'many2one'.

        Исключения:
        -----------
        ValueError
            - Если указан неподдерживаемый тип отображения
            - Если файл имеет неподдерживаемое расширение
            - Если после загрузки данные не являются словарём

        KeyError
            - Если в режиме 'one2many' один ключ относится к нескольким значениям

        Примечания:
        -----------
        1. При чтении из файла: автоматически определяется формат по расширению;
        2. При инициализации с type='auto': проверяется только первый уровень
        вложенности
           для определения типа словаря;
        3. В режиме 'one2many' происходит валидация на уникальность ключей во всех
        группах.

        Примеры:
        ---------
        # Прямая инициализация
        m2o = ManyToOne({'a': 1, 'b': 1, 'c': 2})

        # Загрузка из файла с автоматическим определением типа
        m2o = ManyToOne('data.yml')

        # Явное указание типа
        m2o = ManyToOne({'группа1': ['item1', 'item2']}, type='one2many')
        """
        if type not in self.acceptable_types:
            msg = f'Оижаемые типы - {self.acceptable_types}.\nПолучено: "{type}"!'
            raise ValueError(msg)
        # Читаем из файла, если надо:
        if isinstance(source, dict):
            source = dict(source)  # Создаём копию исходного словаря
        if isinstance(source, (str, Path)):
            source = Path(source)
            ext = source.suffix.lower()
            if ext in {'.json', '.jsonl'}:
                source = json2obj(source)
            elif ext in {'.yaml', '.yml'}:
                source = yaml2obj(source)
            else:
                msg = f'Неподдерживаемый тип файла "{ext}"!'
                raise ValueError(msg)

        # Убеждаемся, что получили словарь:
        if not isinstance(source, dict):
            msg = f'Неподдерживаемый тип данных "{source.__class__}"!'
            raise ValueError(msg)

        # Доопределяем тип словаря, если надо:
        if type == 'auto':
            type = 'one2many'
            for value in source.values():
                if not isinstance(value, (list, tuple, set)):
                    type = 'many2one'
                    break
        # Если не все значения являются списками, словарями или множествами,
        # то считаем, что он - 'many2one'.

        # Присваеваем данные:
        if type == 'many2one':
            self |= source
        else:
            self._one2many = source
            for one, many in source.items():
                for key in many:
                    if key in self:
                        msg = f'Ключ "{key}" относится одновременно к "{one}" и "{self[key]}"!'
                        raise KeyError(msg)
                    self[key] = one

    @property
    def one2many(self):
        """Обратный словарь: один -> список_многих.

        Пример:
        -------
        >>> m2o = ManyToOne({'a': 'X', 'b': 'X', 'c': 'Y'})
        >>> m2o.one2many
        {'X': ['a', 'b'], 'Y': ['c']}

        Примечания:
        -----------
        1. Результат кэшируется после первого вычисления;
        2. Порядок ключей в списках соответствует порядку их добавления;
        3. Для пустого словаря возвращается пустой словарь.

        Вычислительная сложность: O(n) при первом вызове, O(1) при последующих
        """
        # Создаём словарь one2many, если его не было:
        if not hasattr(self, '_one2many'):
            _one2many = defaultdict(list)
            for many, one in self.items():
                _one2many[one].append(many)
            self._one2many = dict(_one2many)

        return self._one2many

    def asdict(self):
        """Возвращает стандартное представление словаря Python."""
        return dict(self)

    def save(self, file_path: str | Path, type: str = 'one2many'):
        """Сохранение словаря в файл.

        Параметры:
        ----------
        file_path : str | Path
            Путь для сохранения JSON- или YAML-файла.

        type : str, default='one2many'
            Формат сохранения:
            - 'one2many': сохраняется в компактном виде {значение: [ключи]};
            - 'many2one': сохраняется в развёрнутом виде {ключ: значение}.

        Пример:
        -------
        >>> m2o = ManyToOne({'a': 'X', 'b': 'X', 'c': 'Y'})
        >>> m2o.save('data.yaml')  # Сохранит в компактном формате
        >>> m2o.save('data.json', type='many2one')  # Сохранит в развёрнутом формате
        """
        acceptable_types = self.acceptable_types[:2]
        if type not in acceptable_types:
            msg = f'Оижаемые типы - {acceptable_types}.\nПолучено: "{type}"!'
            raise ValueError(msg)

        # Получаем словарь в нужном виде:
        source = self.one2many if type == 'one2many' else self.asdict()

        # Пишем словарь в файл:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext in {'.json', '.jsonl'}:
            return obj2json(source, file_path)
        elif ext in {'.yaml', '.yml'}:
            return obj2yaml(source, file_path)
        else:
            msg = f'Неподдерживаемый тип файла "{ext}"!'
            raise ValueError(msg)

    def to_mermaid(self, file_path: str | Path | None = None) -> str:
        """
        Генерирует Mermaid-граф отношений many-to-one.

        Формат: graph LR (left-to-right)
        Слева: ключи, справа: значения

        Параметры:
            file_path: путь для сохранения .mmd файла (опционально).

        Возвращает:
            строка с кодом Mermaid-диаграммы.
        """
        lines = ["graph LR"]

        for key, value in self.items():
            # Экранируем спецсимволы для Mermaid
            key_esc = str(key).replace('"', "'").replace('\n', ' ')
            val_esc = str(value).replace('"', "'").replace('\n', ' ')

            lines.append(f'    "{key_esc}" --> "{val_esc}"')

        diagram = "\n".join(lines)

        if file_path is not None:
            Path(file_path).write_text(diagram, encoding='utf-8')

        return diagram


def get_file_list(path: str,
                  extentions: str | list[str] | tuple[str] | set[str] = [],
                  recurcive: bool = True):
    '''
    Возвращает список всех файлов, содержащихся по указанному пути,
    включая поддиректории.

    Если extentions = 'dir', то выводиться будут дирректории.
    '''
    # Переданный путь должен быть дирректорией:
    if not os.path.isdir(path):
        raise FileNotFoundError('Параметр `path` должен быть путём до '
                                f'существующей папки. Получено {path}')

    # Обработка параметра extentions:

    # Если вместо списка/множества/кортежа расширений
    # указана строка, то делаем из неё множество:
    if isinstance(extentions, str):
        extentions = {extentions}

    # Переводим все элементы списка в нижний регистр:
    extentions = {ext.lower() for ext in extentions}

    # Список расширений только для файлов:
    file_extentions = set(extentions)

    # Если нужно выводить и дирректории:
    if 'dir' in extentions:
        get_dirs = True
        file_extentions -= {'dir'}
    else:
        get_dirs = False

    # Если в списках расширений есть '.*' или 'all',
    # или исходный extentions оставлен по умолчанию,
    # то искать надо все файлы:
    get_all_files = (file_extentions & {'.*', 'all'}) or len(extentions) == 0

    # Рекурсивное заполнение списка найденных файлов:

    # Инициализация списка найденных файлов:
    file_list = [path] if get_dirs else []

    # Перебор всего содержимого заданной папки:
    for file in os.listdir(path):

        # Уточняем путь до текущего файла:
        file = os.path.join(path, file)

        # Если текущий файл - каталог:
        if os.path.isdir(file):

            # Если нужна рекурсия, то включаем всё его содержимое:
            if recurcive:
                file_list += get_file_list(file, extentions, recurcive)

            # Если рекурсия не нужна, но дирректории тоже надо добавлять,
            # то добавляем текущую:
            elif get_dirs:
                file_list.append(file)

        # Если это файл, и либо любой файл подходит, либо его расширение
        # подходит, то добавляем в список:
        elif get_all_files or \
                os.path.splitext(file)[1].lower() in file_extentions:
            file_list.append(file)

    return file_list


def split_dir_name_ext(file):
    '''
    Разделяет путь до файла на:
    dir - путь до папки,
    name - имя файла,
    ext - расширение файла.

    Т.е. функция комбинирует os.path.split и os.path.splitext.
    '''
    dir, name_ext = os.path.split(file)
    name, ext = os.path.splitext(name_ext)
    return dir, name, ext


########################
# Работа с процессами: #
########################


# Аналог Pool.starmap, совместимый с tqdm:
def istarmap(self, func, iterable, chunksize=1):
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = pool.Pool._get_tasks(func, iterable, chunksize)
    result = pool.IMapIterator(self)
    self._taskqueue.put((self._guarded_task_generation(result._job,
                                                       pool.starmapstar,
                                                       task_batches),
                         result._set_length))

    return (item for chunk in result for item in chunk)


# Вносим новый метод в старый класс:
pool.Pool.istarmap = istarmap
# Взято с https://stackoverflow.com/a/57364423


def batch_mpmap_func(func, *args):
    return [func_(*args_) for func_, *args_ in zip(func, *args)]


def batch_mpmap_args(func, args, batch_size=10):
    '''
    Группирует аргументы и создаёт соовтествующую функцию для mpmap.
    Используется для выполнения в mpmap нескольких задачь в одном процессе.
    '''
    # def batched_func(*agrs):
        # return mpmap(func, *args, num_procs=1)

    num_args  = len(args   )
    num_tasks = len(args[0])

    agrs_batches = [[] for _ in range(num_args + 1)]

    for start_ind in range(0, num_tasks, batch_size):

        end_ind = start_ind + batch_size
        if end_ind >= num_tasks:
            end_ind = None

        for arg_ind in range(num_args):
            agrs_batches[arg_ind + 1].append(args[arg_ind][start_ind:end_ind])

        agrs_batches[0].append([func] * len(agrs_batches[-1][-1]))

    return agrs_batches


def mpmap(func      : 'Функция, применяемая отдельно к каждому элементу списка аргументов'            ,
          *args     : 'Список аргументов'                                                             ,
          num_procs : 'Число одновременно запускаемых процессов. По умолчанию = числу ядер ЦПУ' = 0   ,
          batch_size: 'Группировать несколько элементов в одном процессе для мелких задач'      = 1   ,
          desc      : 'Текст статус-бара. По умолчанию статус-бар не отображается'              = None):
    '''
    Обрабатывает каждый элемент списка в отдельном процессе.
    Т.е. это некий аналог map-функции в параллельном режиме.
    '''
    # Размер группы не может быть нулевым:
    batch_size = batch_size or 1
    # if batch_size == 0:
    #     batch_size = 1

    if len(args) == 0:
        raise ValueError(
            'Должен быть задан хотя бы один список/кортеж аргументов!')

    # Если число процессов задано вещественным числом, то берём его как ...
    # ... коэффициент для общего числа ядер в системе:
    if isfloat(num_procs):
        num_procs = int(num_procs * os.cpu_count())

    # Если нужно запускать всего 1 процесс одновременно, то обработка будет в
    # текущем процессе:
    if num_procs == 1:
        return list(tqdm(map(func, *args),
                         total=reduce(min, map(len, args)),
                         desc=desc, disable=not desc))

    # Если в одном процессе должно быть сразу несколько задач:
    if batch_size > 1:
        # Формируем сгруппированные аргументы:
        batched_args = batch_mpmap_args(func, args, batch_size=batch_size)

        # Выполняем параллельную обработку групп:
        return flatten_list(mpmap(batch_mpmap_func, *batched_args,
                                  num_procs=num_procs, desc=desc))

    # Если нужен реальный параллелизм для каждой задачи:
    with Pool(num_procs if num_procs else None) as p:
        if len(args) > 1:
            total = reduce(min, map(len, args))
            args = zip(*args)
            pmap = p.istarmap

        else:
            args = args[0];
            total = len(args)
            pmap = p.imap

        return list(tqdm(pmap(func, args), total=total,
                         desc=desc, disable=desc is None))


def invzip(args_list):
    '''
    В каком-то смысле это преобразование обратно zip-функции:
        inp = [(1, 2, 3), (4, 5, 6)]
        assert list(zip(*invzip(inp))) == inp

    Пример переразбиения:
        [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]].

    В частности используется для предобработки аргументов
    перед использованием в mpmap. Пример использования:
        args = invzip(args_list)
        out = mpmap(func, *args)
    '''
    # Инициируем выходной список:
    num_args = len(args_list[0])
    out = [[] for _ in range(num_args)]

    for args in args_list:
        for ind, arg in enumerate(args):
            out[ind].append(arg)

    return out


def exec_function(function, *args, **kwargs):
    '''
    Функция-пустышка.
    Выполняет пееданную функцию с указанными аргументами.
    Используется, если нужно выполнить map/mpmap с разыми функциями.
    '''
    return function(*args, **kwargs)


def exec_Functor(Functor, *args, **kwargs):
    '''
    Функция-пустышка.
    Создаёт экземпляр переданного Функтера с указанными аргументами.
    Используется, если нужно выполнить map/mpmap с разыми функциями.
    '''
    return Functor()(*args, **kwargs)


###################
# Другие утилиты: #
###################


def a2hw(a, drop_tail=False):
    '''
    Принудительно дублирует входное значение, если оно одно.

    Т.е.:
        a2hf(a)      == (a, a)
        a2hf([a])    == (a, a)
        a2hf((a, b)) == (a, b)

        a2hf((a, b, c), drop_tail=True) == (a, b)
        a2hf((a, b, c)): Error
    '''
    if hasattr(a, '__len__'):
        if   len(a) == 1              : return (a[0], a[0])
        elif len(a) == 2              : return a
        elif len(a)  > 2 and drop_tail: return a[:2]
        else: raise ValueError(f'Ожидался объект, содержащий 1 или 2 значения. Получен {a}!')
    else:
        return (a, a)


# Словарь перевода римских цифр в арабские числа:
rim2arbic_dict = {'M': 1000,
                  'D': 500 ,
                  'C': 100 ,
                  'L': 50  ,
                  'X': 10  ,
                  'V': 5   ,
                  'I': 1   }


def rim2arabic(rim):
    '''
    Перевод римских чисел в арабские.
    Взято без вникания из https://otvet.mail.ru/question/222121139
    '''
    arabic, mx = 0, 0
    for cur in map(lambda c: rim2arbic_dict[c], rim.upper()[::-1]):
        if cur >= mx:
            mx = cur
            arabic += cur
        else:
            arabic -= cur

    return arabic


def invert_index_vector(vector: list[int] | tuple[int]):
    '''
    Инвертирует вектор индексов.
    Т.е.:
        [1, 3, 0, 2] -> [2, 0, 3, 1]
        [1, 5, 0, 4] -> [2, 0, None, None, 3, 1]
    Используется для перестановок элементов в списках.

    Если вектор a состоит из неповторяющихся натуральных чисел от 0 до len(a),
    то invert_index_vector(invert_index_vector(a)) == a.
    '''
    new_vector = [None] * (max(vector) + 1)
    for ind, val in enumerate(vector):
        new_vector[val] = ind

    return new_vector


def flatten_list(list_of_lists, depth=np.inf):
    '''
    Уменьшает глубину вложенности списков.
    Т.е.:
    flatten_list([[[], [1], 2], [3, [4, 5]]], 0) = [[[], [1], 2], [3, [4, 5]]]
    flatten_list([[[], [1], 2], [3, [4, 5]]], 1) = [[], [1], 2, 3, [4, 5]]
    flatten_list([[[], [1], 2], [3, [4, 5]]], 2) = [1, 2, 3, 4, 5]
    flatten_list([[[], [1], 2], [3, [4, 5]]]   ) = [1, 2, 3, 4, 5]
    '''
    if depth and isinstance(list_of_lists, list):
        new_list_of_lists = []
        for list_ in list_of_lists:
            list_ = flatten_list(list_, depth - 1)

            if not isinstance(list_, list):
                list_ = [list_]
            new_list_of_lists += list_
        return new_list_of_lists
    return list_of_lists


def unflatten_list(flatten_list, shape):
    '''
    Обращает ф-ию "flatten_list".
    Работает как reshape в numpy.
    '''
    # Доопределяем размер, если надо:
    if -1 in shape:
        shape = np.arange(len(flatten_list)).reshape(shape).shape
    else:
        assert np.prod(shape) == len(flatten_list)

    # Конец рекурсии:
    if len(shape) == 1:
        return list(flatten_list)

    # Если размер размерности ненулевой, то делаем рекурсию:
    if shape[0]:
        new_list_of_lists = []
        step = len(flatten_list) // shape[0]
        start = 0
        end = step
        sub_shape = shape[1:]
        for ind in range(shape[0]):
            new_list_of_lists.append(
                unflatten_list(flatten_list[start: end], sub_shape)
            )
            start, end = end, end + step

        return new_list_of_lists

    # Если текущая размерность имеет нулевой размер, возвращаем пустой список:
    return []


def restart_kernel_and_run_all_cells():
    '''
    Перезагрузка ядра ноутбука с последующим запуском всех ячеек
    restart_kernel_and_run_all_cells()
    '''
    display(HTML(
        '''
            <script>
                code_show = false;
                function restart_run_all(){
                    IPython.notebook.kernel.restart();
                    setTimeout(function(){
                        IPython.notebook.execute_all_cells();
                    }, 10000)
                }
                function code_toggle() {
                    if (code_show) {
                        $('div.input').hide(200);
                    } else {
                        $('div.input').show(200);
                    }
                    code_show = !code_show
                }
                code_toggle() 
                restart_run_all()
            </script>
        '''
    ))


ExceptoinOrType = Exception | type | str
RetryExceptions = list[ExceptoinOrType] | tuple[ExceptoinOrType] | set[ExceptoinOrType]


class Retry:
    """
    Декоратор для повторного запуска функций при возникновении указанных исключений.

    Класс позволяет перехватывать и обрабатывать исключения, повторяя выполнение функции
    до указанного максимального количества попыток. Поддерживает фильтрацию исключений
    по типу и/или по сообщению.

    Parameters
        max_attempts : int, optional
            Максимальное количество попыток выполнения функции. По умолчанию 3.
        exception : ExceptionOrType | RetryExceptions, optional
            Исключение или набор исключений для перехвата. Может быть:
            - экземпляром исключения
            - классом исключения (типом)
            - строкой (сообщением об ошибке)
            - коллекцией (список, кортеж, множество) из вышеперечисленных элементов.
            По умолчанию перехватываются все исключения (Exception).

    Attributes
        exceptions : set
            Множество исключений для перехвата.
        max_attempts : int
            Максимальное количество попыток выполнения.

        Examples
        >>> @Retry(max_attempts=3)
        ... def risky_function():
        ...     # код, который может вызвать исключение
        ...     pass

        >>> @Retry(max_attempts=5, exception=ValueError)
        ... def value_sensitive_function():
        ...     # код, чувствительный к значениям
        ...     pass

        >>> @Retry(max_attempts=3, exception=["connection failed", "timeout"])
        ... def network_operation():
        ...     # сетевая операция
        ...     pass

        >>> @Retry(exception=ZeroDivisionError('division by zero'))
        ... def ariph_operation():
        ...     # арифметическая операция
        ...     pass
    """

    def __init__(self,
                 max_attempts: int = 3,
                 exception: ExceptoinOrType | RetryExceptions = Exception) -> None:
        """
        Инициализирует декоратор с указанными параметрами.

        Parameters
            max_attempts : int
                Максимальное количество попыток выполнения функции.
            exception : ExceptionOrType | RetryExceptions
                Исключение или набор исключений для перехвата.

        Raises
            TypeError
                Если передан неподдерживаемый тип исключения.
        """
        # Если предъявлено само исключение или его тип, делаем из него множество:
        if isinstance(exception, ExceptoinOrType):
            exception = {exception}
        # Т.е. ExceptoinOrType -> RetryExceptions

        self.exceptions = exception
        self.max_attempts = max_attempts

    def __call__(self, function: Callable) -> Callable:
        """
        Инициализирует декоратор с указанными параметрами.

        Parameters
            max_attempts : int
                Максимальное количество попыток выполнения функции.
            exception : ExceptionOrType | RetryExceptions
                Исключение или набор исключений для перехвата.

        Raises
            TypeError
                Если передан неподдерживаемый тип исключения.
        """
        def retry(*args, **kwargs):

            # Список уникальных перехваченных исключений:
            cached_exceptions = []

            # Предпринимаем попытки:
            for attempt in range(self.max_attempts):

                try:
                    return function(*args, **kwargs)

                except Exception as e:

                    # Пополняем список перехваченных исключений, если такого ещё не было:
                    for cached_exception in cached_exceptions:
                        if str(cached_exception) == str(e) and type(cached_exception) == type(e):
                            break
                    else:
                        cached_exceptions.append(e)

                    # Обрабатываем исключение в зависимости от его допустимости:
                    if self.is_acceptable_exception(e):
                        print(f'Попытка №{attempt + 1}', end='\r')
                    else:
                        print('=========')
                        raise e

            # Если допустимое число попыток исчерпано:
            print('Превышено допустимое число попыток!')
            if len(cached_exceptions) > 1:
                exception = ExceptionGroup(
                    'Перехвачено несколько исключений!',
                    cached_exceptions
                )
            else:
                exception = cached_exceptions[0]
            raise exception

        return retry

    def is_acceptable_exception(self, exception: Exception) -> bool:

        is_acceptable = False
        for acceptable_exception in self.exceptions:

            # Если задана строка:
            if isinstance(acceptable_exception, str):
                is_acceptable = str(exception) == acceptable_exception

            # Если задано исключение:
            elif isinstance(acceptable_exception, Exception):
                is_acceptable = isinstance(exception, type(acceptable_exception)) and \
                    str(exception) == str(acceptable_exception)

            # Если задан тип исключения:
            elif issubclass(acceptable_exception, Exception):
                is_acceptable = isinstance(exception, acceptable_exception)

            else:
                raise TypeError(f'Неожиданное исключение: {acceptable_exception}!')

            # Если уже найдено совпадение - дальше не ищем:
            if is_acceptable:
                return True

        # Если не найдено ни одного совпадения:
        return False


def cls():
    '''
    Очистка консоли или ячейки
    '''
    os.system('cls' if os.name == 'nt' else 'clear')
    clear_output(wait=True)


class TimeIt():
    '''
    Контекст, засекающий время выполнения вложенного кода в секундах.

    Пример использования:
    with TimeIt('генерацию случайнных чисел') as t:
        np.random.rand(100000000)
    print(t())
    >>> На генерацию случайнных чисел потрачено 0.0363 секунд.
    >>> 0.03633379936218262
    '''

    def __init__(self, title=None):
        self.title = title

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.time_spent = time.time() - self.start
        if self.title:
            print(
                'На %s потрачено %.6s секунд.' % (self.title, self.time_spent))

    def __call__(self):
        return self.time_spent


class AnnotateIt():
    '''
    Контекст, выводящий одну строчку перед выполнением вложенного кода,
    и перезаписывающий его другой сточкой после окончания выполнения.
    Полезен при необходимости комментировать начало и конец какого-то процесса.

    Пример использования:
    with AnnotateIt('Обработка выполняется...',
                    'Обработка выполнена.'    ) as a:
        np.random.rand(1000000000)
    '''

    def __init__(self,
                 start_annotation: 'Предворяющий текст' = '',
                 end_annotation: 'Завершающий  текст' = None):
        """Инициализация."""

        # Если оба текста не указаны, то выводиться ничего не будет:
        self.enable = start_annotation or end_annotation

        if self.enable:
            # Фиксируем оба текста, если они заданы явно:
            if end_annotation:
                self.start_annotation = start_annotation
                self.end_annotation = end_annotation

                # Если заверщающий текст короче предворяющего,
                # то дополняем длину пробелами, чтобы затереть:
                self.end_annotation += ' ' * max(
                    0,
                    len(start_annotation) - len(end_annotation)
                )

            # Дополняем символами из юникода если дан лишь базовый текст:
            else:
                self.start_annotation = '⏳ ' + start_annotation
                self.end_annotation = '✅' + start_annotation

    def __enter__(self):
        if self.enable:
            print(self.start_annotation, end='')

    def __exit__(self, type, value, traceback):
        if self.enable:
            print('\r' + self.end_annotation)


class Beep:
    """Контекст для кода в Jupyter, издающий звук при завершении.

    Требуется установка модуля jupyter_beeper.
    """

    def __init__(self, *args):
        """Инициализация.

        В качестве агрументов должны перечисляться паузы в секундах для
        однократного, двукратного, трёхкратного и т.д. гудков соответственно. Список
        сортируется по возрастанию. По умолчанию будет одинарный при любой задержке.

        Примеры:
            > with Beep():
            >     pass
            > # Одинарный сигнал.

            > with Beep(60, 300):
            >     time.sleep(30)
            > # Сигнала не будет.

            > with Beep(60, 300):
            >     time.sleep(100)
            > # Одинарный сигнал.

            > with Beep(60, 300):
            >     time.sleep(500)
            > # Двойной сигнал.
        """

        # Убеждаемся, что это действительно Jupyter:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            # Проверяем различные признаки Jupyter:
            if ip is None or not hasattr(ip, 'kernel'):
                raise ImportError
        except ImportError:
            print(
                'Код запущен не в ноутбуке.',
                'Звукового уведомления не будет!',
            )
            return

        # Загрузка необходимого модуля:
        try:
            import jupyter_beeper
            self.beeper = jupyter_beeper.Beeper()
        except ImportError:
            print(
                'Модуль "jupyter_beeper" не установлен.',
                'Звукового уведомления не будет!',
            )
            return

        # По умолчанию - одинарный гудок для любой задержки:
        if len(args):
            self.pauses = sorted(args)
        else:
            self.pauses = [0]

    def __enter__(self):
        """Засекаем время в начале."""
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        """Издаём звук в конце с нужным числом повторений."""
        # Если звук всё равно издать не удастся - ничего не делаем:
        if not hasattr(self, 'beeper'):
            return

        # Определяем временной инрервал:
        time_spent = time.time() - self.start

        for pause in self.pauses:
            if time_spent < pause:
                break
            self.beeper.beep(frequency=440, secs=0.5, blocking=True)
            self.beeper.beep(frequency=530, secs=0.5, blocking=True)


class SuppressModuleLogs:
    """Контекст, подавляющий логи указанного модуля ниже заданного уровня.

    По умолчанию подавляются логи всех модулей в контексте.
    """

    def __init__(self, module_name: str = '', level: int = logging.WARNING) -> None:
        self.module_name = module_name
        self.level = level
        self._active = False
        # Сохраняем оригинальные функции:
        self._orig_getLogger = logging.getLogger
        self._orig_addHandler = logging.Logger.addHandler

    def _is_our_module(self, name: str) -> bool:
        """Проверяет, относится ли логгер к подавляемому модулю."""
        if not self.module_name:
            return True
        return name == self.module_name or name.startswith(f"{self.module_name}.")

    def _suppress_logger(self, logger: logging.Logger) -> None:
        """Подавляет уровень логгера и всех его обработчиков."""
        logger.setLevel(self.level)
        for handler in logger.handlers:
            handler.setLevel(self.level)

    def _patch_logging(self) -> None:
        """Подменяет функции logging для перехвата новых логгеров."""
        suppressor = self

        def getLogger(name=None):
            logger = suppressor._orig_getLogger(name)
            if suppressor._active and name and suppressor._is_our_module(name):
                suppressor._suppress_logger(logger)
            return logger

        def addHandler(self_logger, handler):
            if suppressor._active and suppressor._is_our_module(self_logger.name):
                handler.setLevel(suppressor.level)
            return suppressor._orig_addHandler(self_logger, handler)

        logging.getLogger = getLogger
        logging.Logger.addHandler = addHandler

    def _restore_logging(self) -> None:
        """Восстанавливает оригинальные функции logging."""
        logging.getLogger = self._orig_getLogger
        logging.Logger.addHandler = self._orig_addHandler

    def _suppress_existing(self) -> None:
        """Подавляет уже существующие логгеры."""
        for name in list(logging.root.manager.loggerDict.keys()):
            if self._is_our_module(name):
                logger = self._orig_getLogger(name)
                self._suppress_logger(logger)

    def __enter__(self) -> Self:
        self._active = True
        self._patch_logging()
        self._suppress_existing()
        return self

    def __exit__(self, *_) -> None:
        self._active = False
        self._restore_logging()


class DelayedInit:
    '''
    Откладывает создание экземпляра заданного класса до первого обращения к
    его атрибутам.

    Суть аргументов становится ясна, если понять, что отложенная инициализация
    объекта в самом общем виде выполняется следующим образом:
    ```
    getattr(DelayedClass, init_method)(
        *(args + args_func(*args_func_args, **args_func_kwargs),
        **(kwargs | kwargs_func(*kwargs_func_args, **kwargs_func_kwargs))
    )
    ```
    Т.е. откладывать можно и выполнение функций, генерирующих входные параметры
    конструктора.

    Пример работы.
    Для класса ...
    ```
    class MyClass:
    x = 'x'

    def __init__(self, y='y'):
        self.y = y
        print('\tСоздан!')

    def __call__(self):
        xy = self.x + self.y
        print('\tИспользован:', xy)
        return(xy)
    ```
    ... обычный жизненный цикл ...
    ```
    print('Создание:')
    mc = MyClass(y='y_')

    print('Использование:')
    mc()
    ```
    ... приводит к следующему выводу:
    ```
    Создание:
    	Создан!
    Использование:
    	Использован: xy_
    ```
    Но при оборачивании в DelayedInit ...
    ```
    print('Создание:')
    mc = DelayedInit(MyClass, kwargs={'y': 'y_'})

    print('Использование:')
    mc()
    ```
    ... изменяет порядок вывода:
    ```
    Создание:
    Использование:
    	Создан!
    	Использован: xy_
    ```
    '''

    def __init__(self, DelayedClass, init_method='__init__',
                 args=[], kwargs={},
                 args_func=None, kwargs_func=None,
                 args_func_args=[], args_func_kwargs={},
                 kwargs_func_args=[], kwargs_func_kwargs={}):
        self.DelayedClass = DelayedClass
        self.init_method = init_method
        self.args = args
        self.kwargs = kwargs
        self.args_func = args_func
        self.kwargs_func = kwargs_func
        self.args_func_args = args_func_args
        self.args_func_kwargs = args_func_kwargs
        self.kwargs_func_args = kwargs_func_args
        self.kwargs_func_kwargs = kwargs_func_kwargs

    # Выполняем отложенную инициализацию:
    def ExecInit(self):

        # Берём явно заданные аргументы конструктора:
        args = self.args
        kwargs = self.kwargs

        # Дополняем аргументы конструктора результатами выполнения
        # соотвествующих функций, если они были заданы:
        if self.args_func is not None:
            args = args + self.args_func(*self.args_func_args,
                                         **self.args_func_kwargs)
        if self.kwargs_func is not None:
            kwargs = kwargs | self.kwargs_func(*self.kwargs_func_args,
                                               **self.kwargs_func_kwargs)

        # Получаем экземпляр нужный класса:
        delayed_obj = getattr(self.DelayedClass,
                              self.init_method)(*args, **kwargs)

        # Выполняем подмену экземпляра класса со всеми его атрибутами:
        self.__class__ = self.DelayedClass
        # https://stackoverflow.com/a/18529310
        self.__dict__.update(delayed_obj.__dict__)
        # https://stackoverflow.com/a/37658673

    def __getattr__(self, attr):
        self.ExecInit()
        if hasattr(self, attr):
            return self.__getattribute__(attr)
        else:
            raise AttributeError(f'Атрибут {attr} не найден!')

    def __call__(self, *args, **kwargs):
        call_method = self.__getattr__('__call__')
        return call_method(*args, **kwargs)
    # Почему-то для delayed_obj() приходится прописывать __call__ явно.


def disable_methods(*method_names):
    """
    Декоратор для отключения унаследованных методов

    Пример использования:
        class Parent:
            def method1(self): print("Метод 1")
            def method2(self): print("Метод 2")
            def method3(self): print("Метод 3")

        @disable_methods('method1', 'method2', 'method3')
        class Child(Parent):
            pass
    """
    def decorator(cls):
        for name in method_names:
            def method_raiser(self, *args, __name=name, **kwargs):
                raise AttributeError(f"Метод '{__name}' отключен в классе {cls.__name__}")
            setattr(cls, name, method_raiser)
        return cls
    return decorator


def apply_on_cartesian_product(func     : 'Функция двух аргументов'        ,
                               values1  : 'Первый список объектов'         ,
                               values2  : 'Второй список объектов'  = None ,
                               symmetric: 'Функция симметрическая?' = False,
                               diag_val : 'Чему равно func(a, a)'   = None ,
                               **mpmap_kwargs):

    '''
    Формирует матрицу применения функции func(a, b) к
    декартовому произведению элементов из values1 и values2.

    Если values2 не задан, то values1 умножается сам на
    себя. Если при этом задаётся diag_val, то им заменяются
    все диагональные элементы квадратной матрицы.

    Если symmetric == True, то func считается симметрической
    (т.е. func(a, b) == func(b, a)), и итоговая матрица
    будет рассчитываться по упрощённой схеме для ускорения
    вычислений.
    '''
    # Для выполнения вычислений в параллельном режиме
    # сначала составляется список значений каждого
    # агрумента функции, а так же соответствующий ему
    # список индексов (в какую ячейку вносить результат).

    # Инициализация матрицы результатов:
    if values2 is not None:
        mat = np.zeros((len(values1), len(values2)), dtype=object)
    else:
        mat = np.zeros([len(values1)] * 2          , dtype=object)

    ###############################
    # Формируем спиок аргументов. #
    ###############################

    # Инициализация списков аргументов и индексов для задач:
    args1 = []
    args2 = []
    inds  = []

    # Если второй список объектов задан:
    if values2 is not None:

        # Заполняем списки аргументов и индексов задач:
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                args1.append(v1)
                args2.append(v2)
                inds .append([(i, j)])

    # Если второй список объектов не задан, то
    # строим связность первого списка с собой:
    else:

        # Заполнение таблицы:
        for i, v1 in enumerate(values1):

            # Если значение диагональных элементов не задано:
            if diag_val is None:

                # Добавляем очередную задачу для ячейки (i, i):
                args1.append(v1)
                args2.append(v1)
                inds .append([(i, i)])

            # Если значение диагональных элементов задано, то сразу прописываем
            # его в матрицу:
            else:
                mat[i, i] = diag_val

            # Последнюю строку пропускаем, т.к. все её элементы уже учтены:
            if i == len(values1) - 1:
                continue
            # Элемент диагонали был учтён в предыдущих строках, а ...
            # ... остальные в последующих благодаря работе не только ...
            # ... с (i, j), но и с (j, i)!

            # Все остальные элементы рассчитываем обычным образом:
            for j, v2 in enumerate(values1[i + 1:], i + 1):

                # Если функция симметрическая, то матрица будет симметрична
                # относительно главной диаганали:
                if symmetric:
                    args1.append(v1)
                    args2.append(v2)
                    inds.append([(i, j), (j, i)])

                # Если функция несимметрическая, то создаются две задачи
                # (для (i, j) и (j, i)):
                else:
                    args1.append(v1)
                    args1.append(v2)
                    args2.append(v2)
                    args2.append(v1)
                    inds.append([(i, j)])
                    inds.append([(j, i)])

    # Выполняем составленные задачи:
    rzlts = mpmap(func, args1, args2, **mpmap_kwargs)

    # Записываем результаты в соответствующие ячейки:
    for ijs, rzlt in zip(inds, rzlts):
        for i, j in ijs:
            mat[i, j] = rzlt

    return mat


def reorder_lists(ordered_inds, *args):
    '''
    Меняет очерёдность элементов нескольких списков по общему шаблону
    sorted_inds.
    '''
    # Инициализируем итоговый список:
    sorted_args = []

    # Перебираем все сортируемые списки:
    for arg in args:

        # Сортируем очередной список:
        sorted_arg = [arg[ordered_ind] for ordered_ind in ordered_inds]

        # Добавляем отсортированный список в итоговый:
        sorted_args.append(sorted_arg)

    return sorted_args


class CircleInd:
    """
    Целое число с замкнутым инкриментом/декриментом.
    Полезно для круговой адресации к элементам массива.
    """

    def __init__(self, circle, ind=0):
        assert 0 <= ind < circle
        self.circle = circle
        self.ind = ind

    def inc(self):
        self.ind += 1
        if self.ind == self.circle:
            self.ind = 0
        return self.ind

    def dec(self):
        self.ind -= 1
        if self.ind == -1:
            self.ind = self.circle - 1
        return self.ind

    def __int__(self):
        return self.ind

    __call__ = __int__

    def __eq__(self, other):
        return self.ind == int(other)

    def __ne__(self, other):
        return self.ind != int(other)


class InternalRandomSeed:
    '''
    Декоратор и контекст.
    Выполняет вложенный код с собственным random.seed(), сохраняя
    внешнее состояние генератора псевдослучайных чисел (ГПСЧ) неизменным.

    Позволяет отвязывать внутреннее и внешние состояния ГПСЧ, достигая,
    например, воспроисводимости результатов какого-то генератора.

    Пример работы:

        ```
        import random

        irs = InternalRandomSeed()

        # Декоратор:
        ri = irs(random.randint)
        print(ri(0, 100), random.randint(0, 100))
        print(ri(0, 100), random.randint(0, 100))
        ri.reset_seed()
        print(ri(0, 100), random.randint(0, 100))
        print(ri(0, 100), random.randint(0, 100))

        # Контекст:
        irs.reset()
        with irs:
            print(random.randint(0, 100))
            print(random.randint(0, 100))
        ```

        >>> 81 60
        >>> 14 32
        >>> 81 91
        >>> 14 48
        >>> 81
        >>> 14
    '''

    def __init__(self, start_seed=42):
        self.start_seed = start_seed
        self.reset(start_seed)

    # Установка стартового состояния генератора:
    def reset(self, start_seed=None):
        start_seed = start_seed or self.start_seed
        self.internal_seed = random.getstate()
        with self as s:
            random.seed(start_seed)

    # Установка нового состояния и возвращение старого:
    @staticmethod
    def swap_random_python(new_seed):
        old_seed = random.getstate()
        random.setstate(new_seed)
        return old_seed

    # Установка и снятие контекста:
    def __enter__(self):
        self.external_seed = self.swap_random_python(self.internal_seed)

    def __exit__(self, type, value, traceback):
        self.internal_seed = self.swap_random_python(self.external_seed)

    # Декорация через контекст:
    def __call__(self, func):

        # Создаём декорированную функцию:
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        # Включаем возможность сбрасывать внутреннее состояние в исходное:
        new_func.reset_seed = self.reset

        return new_func


def isint(a):
    '''
    Возвращает True, если число или numpy-массив является целочисленным.
    '''
    return isinstance(a, (int, np.integer)) or \
           isinstance(a, np.dtype) and np.issubdtype(a, np.integer) or \
           isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.integer)


def isfloat(a):
    '''
    Возвращает True, если число или numpy-массив является вещественным.
    '''
    return isinstance(a, (float, np.floating)) or \
           isinstance(a, np.dtype)   and np.issubdtype(a, np.floating) or \
           isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating)


def obj_diff(obj1, obj2, prefix=''):
    '''
    Выводит список найденных различий в двух структурах.
    '''
    if obj1 == obj2:
        return ''

    # Сравниваем типы:
    if type(obj1) is not type(obj2):
        return f'Несовпадение типов {prefix}: {type(obj1)} != {type(obj2)}!'

    # Сравниваем размеры:
    if hasattr(obj1, '__len__') and len(obj1) != len(obj2):
        return f'Несовпадение размеров {prefix}: {len(obj1)} != {len(obj2)}!'

    # Если объект итерируемый:
    if hasattr(obj1, '__iter__'):

        # Если объекты являются словарями:
        if isinstance(obj1, dict):

            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())

            dkeys = keys1 - keys2

            if dkeys:
                return f'Ненайденные ключи {prefix}: {dkeys}!'

            else:
                out = ''
                for key in keys1:
                    out_ = obj_diff(obj1[key], obj2[key], prefix + f'[{key}]')

                    if out_:
                        out = out + '\n' + out_

                if out:
                    return out[1:]
                else:
                    return ''

        # Если объекты являются множествами, кортежами или списками:
        if isinstance(obj1, (set, tuple, list)):

            # На всякий случай сортируем элементы для облегчения
            # поэлементного сопоставления, если возможно:
            try:
                obj1 = sorted(obj1)
                obj2 = sorted(obj2)
                if obj1 == obj2:
                    return ''
            except:
                pass

            out = ''
            for ind, (sub1, sub2) in enumerate(zip(obj1, obj2)):
                out_ = obj_diff(sub1, sub2, prefix + f'[{ind}]')

                if out_:
                    out = out + '\n' + out_

            if out:
                return out[1:]
            else:
                return ''

    return f'Несовпадение значений {prefix}: {obj1} != {obj2}!'


# При автономном запуске создаётся dependency_graph.mmd для dl_utils:
if __name__ == '__main__':
    draw_repo_dependency_graph()


__all__ = [
    # Работа с изображениями:
    'cv2_vid_exts', 'cv2_img_exts', 'autocrop', 'df2img', 'fig2nparray',
    'draw_contrast_text', 'resize_with_pad', 'img_dir2video', 'ImReadBuffer',

    # Работа с файлами:
    'mkdirs', 'rmpath', 'emptydir', 'first_existed_path', 'unzip_file',
    'unzip_dir', 'obj2yaml', 'yaml2obj',

    # Работа с итерируемыми объектами:
    'mpmap', 'invzip', 'flatten_list',

    # Украшательства:
    'cls', 'TimeIt', 'AnnotateIt',

    # Прочее:
    'rim2arabic', 'restart_kernel_and_run_all_cells', 'isint', 'isfloat'
]