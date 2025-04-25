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

import numpy as np

from typing import Union
# from inspect import isclass
from functools import reduce
from zipfile import ZipFile
from shutil import rmtree, copyfile
from tqdm import tqdm
from time import time
from multiprocessing import pool, Pool
from IPython.display import clear_output, HTML  # , Javascript, display
from matplotlib import pyplot as plt


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
    textsize = text_img.shape

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
        alpha=alpha,
    )

    return img


def color_float_hsv_to_uint8_rgb(h: float,
                                 s: float = 1.,
                                 v: float = 1.,
                                 a: Union[float, None] = None) -> tuple:
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
        # Если это папка:
        if os.path.isdir(path):
            with AnnotateIt(desc):
                rmtree(path)
            return True

        # Если это Файл:
        elif os.path.isfile(path):
            with AnnotateIt(desc):
                os.remove(path)
            return True

        else:
            raise ValueError(f'Не файл и не папка "{path}"!')

    except PermissionError:
        raise PermissionError(f'Недостаточно прав удалить "{path}"!')


def emptydir(path):
    '''
    Создаёт папку, если её не было или
    очищает всё её содержимое, если она была.
    '''
    rmpath(path)
    mkdirs(path)
    return


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
    with ZipFile(zip_file, 'r') as z:
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
        zip_files             .append(     zip_file        )
        unzipped_files_subdirs.append(unzipped_files_subdir)

    # Распаковываем во временную папку:
    unzipped_files_dirs = mpmap(unzip_file, zip_files, unzipped_files_subdirs,
                                num_procs=not use_multiprocessing, desc=desc)

    return unzipped_files_dirs


def obj2yaml(obj, file='./cfg.yaml', encoding='utf-8', allow_unicode=True):
    '''
    Пишет словарь, множество или кортеж в yaml-файл.
    Параметры по-умолчанию позволяют сохранять кириллицу.
    '''
    with open(file, 'w', encoding=encoding) as stream:
        yaml.safe_dump(obj, stream, allow_unicode=allow_unicode, sort_keys=False)

    return file


def yaml2obj(file='./cfg.yaml', encoding='utf-8'):
    '''
    Читает yaml-файл.
    '''
    with open(file, 'r', encoding=encoding) as stream:
        obj = yaml.safe_load(stream)

    return obj


def get_file_list(path, extentions=[]):
    '''
    Возвращает список всех файлов, содержащихся по указанному пути,
    включая поддиректории.
    '''
    # Обработка параметра extentions:

    # Если вместо списка/множества/кортежа расширений ...
    # ... указана строка, то делаем из неё множество:
    if isinstance(extentions, str):
        extentions = {extentions}

    # Если же это действительно список/множество/кортеж:
    elif isinstance(extentions, (list, tuple, set)):

        # Формируем список элементов, не являющихся строками:
        exts = [ext for ext in extentions if not isinstance(ext, str)]

        # Если список не пуст, выводим ошибку:
        if exts:
            raise ValueError('Найдены следующие некорректные объекты в ' +
                             f'списка/множества/кортежа расширений: {exts}')

    else:
        raise ValueError('extentions должен быть строкой, или ' +
                         'списком/кортежем/множеством строк. Получен ' +
                         str(extentions))

    # Переводим все элементы списка в нижний регистр:
    extentions = {ext.lower() for ext in extentions}

    # Рекурсивное заполнение списка найденных файлов:

    # Инициализация списка найденных файлов:
    file_list = []

    # Перебор всего содержимого заданной папки:
    for file in os.listdir(path):

        # Уточняем путь до текущего файла:
        file = os.path.join(path, file)

        # Если текущий файл - каталог, то добавляем всё его ...
        # ... содержимое в список через рекурсивный вызов:
        if os.path.isdir(file):
            file_list += get_file_list(file, extentions)

        # Если тип текущего файла соответствует искомому, либо ...
        # ... типы искомых файлов не заданы, то вносим файл в список:
        elif not len(extentions) or \
                os.path.splitext(file)[1].lower() in extentions:
            file_list.append(file)

    return file_list


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
    self._taskqueue.put((self._guarded_task_generation(result._job     ,
                                                       pool.starmapstar,
                                                       task_batches    ),
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
          desc      : 'Текст статус-бара. По-умолчанию статус-бар не отображается'              = None):
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
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        self.time_spent = time() - self.start
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
                self.start_annotation = start_annotation + ' ⏳'
                self.end_annotation = start_annotation + ' ✅'

    def __enter__(self):
        if self.enable:
            print(self.start_annotation, end='')

    def __exit__(self, type, value, traceback):
        if self.enable:
            print('\r' + self.end_annotation)


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


def extend_list_in_dict_value(d: dict,
                              key,
                              value: list,
                              filo=True) -> dict:
    '''
    Значения в словаре являются списками, которые можно дополнять.
    Это используется если нужно чтобы по одному ключу были доступны сразу
    несколько значений, которые могут добавляться постепенно.
    '''
    if key in d:
        d[key] = d[key] + value if filo else value + d[key]
    else:
        d[key] = list(value)
    return d


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
    def __exit__ (self, type, value, traceback):
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


# Работа с изображениями:
__all__  = ['cv2_vid_exts', 'cv2_img_exts']
__all__ += ['autocrop', 'df2img', 'fig2nparray', 'draw_contrast_text', 'resize_with_pad', 'img_dir2video', 'ImReadBuffer']

# Работа с файлами:
__all__ += ['mkdirs', 'rmpath', 'emptydir', 'first_existed_path']
__all__ += ['unzip_file', 'unzip_dir']
__all__ += ['obj2yaml', 'yaml2obj']


# Работа с итерируемыми объектами:
__all__ += ['mpmap', 'invzip', 'flatten_list']

# Украшательства:
__all__ += ['cls', 'TimeIt', 'AnnotateIt']

# Прочее:
__all__ += ['rim2arabic', 'restart_kernel_and_run_all_cells']
__all__ += ['isint', 'isfloat']