import os
import cv2

import scipy
import numpy as np
from time import time
from tqdm import tqdm
from skimage import feature

from matplotlib import pyplot as plt

from utils import (isint, isfloat, text2img, overlap_with_alpha, mpmap,
                   rmpath, mkdirs, ImReadBuffer, cv2_vid_exts, cv2_img_exts)


def recomp2mp4(source_file, target_file, rm_soruce=True, quiet=True):
    '''
    Пересжимает исходный файл в mp4. Полезно для достижения большей
    компактности за счёт использования внутрикадрового сжатия и лучшей
    аппаратной совместимости с различными устройствами воспроизведения.
    '''
    # Формируем команду для пересжатия:
    cmd = 'ffmpeg -y -hide_banner -loglevel quiet ' + \
          f'-i "{source_file}" -c:v libx264 -preset slow ' + \
          f'-crf 20 -tune animation "{target_file}"'

    # Прячем stderr и stdout, если надо:
    if quiet:
        cmd += '>/dev/null 2>&1'

    # Выполняем пересжатие:
    exit_code = os.system(cmd)

    # Если пересжатие прошло с ошибкой:
    if exit_code:
        # Удаляем неудачный файл:
        rmpath(target_file)

        raise OSError(f'Ошибка пересжатия "{source_file}"!')

    # Если целевой файл создан, а исходный надо удалить, то удаляем:
    elif os.path.isfile(target_file):
        if rm_soruce:
            rmpath(source_file)

    else:
        raise NotImplementedError('Файл не создан, но ошибка не обнаружена!')
    # Удаляем исходный файл, если надо, и если пересжатие завершено успешно:


##############################################
# Работа в конвейере (в т.ч. с видеопотоком) #
##############################################

def SkipNone(func):
    '''
    Декоратор для __call__-функйий в функторах-фильтрах.
    Позволяет пропускать None дальше в конвейере без применения
    самих функций. Такая возможность позволяет имитировать
    асинхронность даже при синхронной работе конвейера.
    '''
    def func_(inp):
        return None if inp is None else func(inp)

    return func_


class Pipeline():
    '''
    Конвейер фильтров обработки изображений.
    '''
    # Конкатенация списка функций:
    @staticmethod
    def concat(image_filter_list):
        functions, names = [], []

        for image_filter in image_filter_list:

            # Если функция/функтор:
            if hasattr(image_filter, '__call__'):
                functions.append(image_filter)
                if hasattr(image_filter, 'name'):
                    names.append(image_filter.name)
                else:
                    names.append('NonameFunction')

            # Если список/кортеж функций:
            elif isinstance(image_filter, (tuple, list)):
                functions_, names_ = concat(image_filter)
                functions.extend(functions_)
                names    .extend(names_)

            # Если строка:
            elif isinstance(image_filter, str):
                # Список синонимов:
                if   image_filter.lower() == 'rgb2gray' : functions.append(rgb2gray )
                elif image_filter.lower() == 'bgr2gray' : functions.append(bgr2gray )
                elif image_filter.lower() == 'rgb2yuv'  : functions.append(rgb2yuv  )
                elif image_filter.lower() == 'rgb2bgr'  : functions.append(rgb2bgr  )
                elif image_filter.lower() == 'bgr2rgb'  : functions.append(bgr2rgb  )
                elif image_filter.lower() == 'yuv2rgb'  : functions.append(yuv2rgb  )
                elif image_filter.lower() == 'yuv2bgr'  : functions.append(yuv2bgr  )
                elif image_filter.lower() == 'yuv2gray' : functions.append(lambda image: image[..., 0])
                elif image_filter.lower() == 'gray2rgb' : functions.append(gray2rgb )
                elif image_filter.lower() == 'gray2bgr' : functions.append(gray2bgr )
                elif image_filter.lower() == 'im2double': functions.append(im2double)
                elif image_filter.lower() == 'double2im': functions.append(double2im)
                else: raise ValueError('"%s" не входит в список синонимов.' % image_filter)
                names.append(image_filter)

            # Если число:
            elif isinstance(image_filter, int):
                thresh = float(image_filter)
                functions.append(lambda image: cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1])
                names.append('trheshold_%d' % image_filter)

            else:
                raise ValueError('%s не является подходящим элементом конвеера.' % image_filter)

        return functions, names

    def __init__(self, image_filter_list, name='Pipeline'):
        self.functions, self.names = self.concat(image_filter_list)
        self.name = name

    def __call__(self, image=None):
        if image is None:
            image = self.functions[0]()
        else:
            image = self.functions[0](image)

        for image_filter in self.functions[1:]:
            image = image_filter(image)

        return image

    # Применяет конвейер для конвертации файлов:
    def convert(self,
                inp_file,
                out_file,
                step=1,
                desc='auto',
                recompress=True,
                skip_existed=True):

        if skip_existed and os.path.isfile(out_file):
            return

        # Если тип конечного файла не поддерживается без пересжатия или
        # небоходимость пересжатия явно задана:
        out_file_name, out_file_ext = os.path.splitext(out_file)
        if recompress or out_file_ext.lower() not in {'.avi'}:

            # Определяем имя временного файла:
            tmp_file = out_file_name + '_tmp.avi'
            assert not os.path.exists(tmp_file)
            # Такой файл к этому моменту ещё не должен существовать!

        # Если пересжатие не требуется, то временный файл и будет
        # окончательным:
        else:
            tmp_file = out_file

        try:
            # Инициируем чтение и запись видео-файлов
            with ViRead(inp_file) as vr, ViSave(tmp_file) as vs:

                # Перебираем все кадры источника:
                for frame_ind in tqdm(range(vr.total_frames),
                                      desc=inp_file if hasattr(desc, 'lower')
                                      and desc.lower() == 'auto' else desc,
                                      disable=desc is None):

                    # Получаем очередной кадр:
                    frame = vr()

                    # Если номер кадра соответствует текущему шагу, то
                    # обрабатываем и записываем его:
                    if frame_ind % step == 0:
                        vs(self(frame))

        except Exception as e:
            raise RuntimeError('Обработка файла прервана из-за ошибки!') from e

        # Если файл требуется пересжать:
        if tmp_file != out_file:
            recomp2mp4(tmp_file, out_file)

    # Сброс всех использующихся фильтров:
    def reset(self, im_size=None):
        for image_filter in self.functions:
            if hasattr(image_filter, 'reset'):
                image_filter.reset()


class ViRead:
    '''
    Возвращает посделовательность кадров видео из файла.
    Работает и как функция и как генератор.
    '''
    def __init__(self, path, start_frame=0, colorspace='rgb', on_end='close'):
        self.path = path
        self.start_frame = start_frame
        self.colorspace = colorspace.lower()
        self.on_end = on_end.lower()
        self.reset()

    def reset(self):
        if hasattr(self, 'cap'):
            self.close()

        # Определение конечного цветового пространства
        if self.colorspace == 'yuv':
            self.colorspace_converter = bgr2yuv
        elif self.colorspace == 'gray':
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.colorspace == 'rgb':
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB )
        elif self.colorspace == 'bgr':
            self.colorspace_converter = lambda image: image
        else:
            raise ValueError('"%s" не входит в список доступных цветовых схем.' % self.colorspace) 

        # Определение файла-источника
        if isinstance(self.path, str):
            path = self.path
        elif isinstance(self.path, (list, tuple, set)):
            path = np.random.choice(self.path)
        else:
            raise ValueError('"%s" должен быть строкой//списком//множеством строк путей до файла.' % path)

        self.cap = cv2.VideoCapture(path)
        if (self.cap.isOpened() == False):
            raise ValueError('Ошибка открытия файла "%s"' % path)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if isinstance(self.start_frame, int):
            start_frame = self.start_frame if self.start_frame >= 0 else self.total_frames + self.start_frame
        elif isinstance(self.start_frame, (list, tuple)):
            start = self.start_frame[0] if self.start_frame[0] >= 0 else self.total_frames + self.start_frame[0]
            end   = self.start_frame[1] if self.start_frame[1] >= 0 else self.total_frames + self.start_frame[1]
            if start >= end:
                self.close()
                raise ValueError('В "%s" всего %d кадров.' % (path, self.total_frames))
            start_frame = np.random.randint(start, end + 1)
        if start_frame < 0 or start_frame >= self.total_frames:
            self.close()
            raise ValueError('В "%s" всего %d кадров.' % (path, self.total_frames))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def close(self):
        self.cap.release()

    def __call__(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = self.colorspace_converter(frame)
            return self.last_frame
        else:
            if self.on_end == 'reset':
                self.reset()
                return self()
            elif self.on_end == 'close':
                self.close()
                return None
            elif self.on_end == 'repeat_last_frame':
                return self.last_frame
            else:
                raise ValueError('Параметр "%s" не может быть равен "%s".' % ('on_end', on_end))

    def __next__(self):
        return self()

    def __enter__(self):
        return self 

    def __exit__(self, type, value, traceback):
        self.close()


class ViSave:
    '''
    Покадрово записывает видеофайл.
    '''
    def __init__(self, path, colorspace='rgb', fps=30.):
        self.path = path
        self.fps = fps

        colorspace = colorspace.lower()
        if colorspace == 'yuv':
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_YUV2BGR )
        elif colorspace == 'gray':
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif colorspace == 'rgb':
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2BGR )
        elif colorspace == 'bgr':
            self.colorspace_converter = lambda image: image
        else:
            raise ValueError('"%s" не входит в список доступных цветовых схем.' % colorspace)

        # Создаём путь до файла, если надо:
        path_dir = os.path.abspath(os.path.dirname(path))
        if not os.path.isdir(path_dir):
            mkdirs(path_dir)

    def close(self):
        if hasattr(self, 'wrt'):
            self.wrt.release()

    def __call__(self, frame):
        if not hasattr(self, 'wrt'):
            self.wrt = cv2.VideoWriter(self.path,
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                       self.fps,
                                       (frame.shape[1], frame.shape[0]))
        self.wrt.write(self.colorspace_converter(frame))

    def __enter__(self):
        return self 

    def __exit__(self, type, value, traceback):
        self.close()


class AsType:
    '''
    Меняет тип тензора.
    '''

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, img):
        return img.astype(self.dtype)


class Resize:
    '''
    Изменение размера изображения:
    '''
    # Изменяет целевой размер изображения:
    def reset_im_size(self, im_size):
        # Если задан не кортеж/список/numpy-массив, дублируем его:
        if not isinstance(im_size, (tuple, list, np.ndarray)):
            im_size = (im_size, im_size)

        self.im_size = tuple(im_size)

    def __init__(self, im_size=512):
        self.reset_im_size(im_size)

    # Перерасчитывает размер оси изображения:
    @staticmethod
    def adopt_axis_size(target_size, source_size):
        if isint(target_size):
            return target_size

        elif isfloat(target_size):
            return int(target_size * source_size)

        else:
            raise ValueError('Параметр "im_size" должен быть задан целыми ' + \
                             'или вещественными числами. Получено ' + \
                             target_size, '!')
    # Нужно, например, для перехода от относительного размера к абсолютному.

    def __call__(self, img):
        # Получаем размер входного изображения:
        im_size = img.shape[:2]

        # Перерассчитываем итоговый размер изображения:
        target_size = [self.adopt_axis_size(self.im_size[i], im_size[i])
                       for i in range(2)]
        # target_size = list(map(, zip(self.im_size, im_size)))
        # Нужно, если размеры указаны вещественными ...
        # ... числами, т.е. являются коэффициентами.

        return cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_AREA)


class ResizeAndRestore:
    '''
    Изменение размера входного изображения перед применением заданного
    обработчика с последующим приведением результата к исходному размеру.
    '''

    def __init__(self, im_size, processor):
        self.preprocessor = Resize(im_size)
        self.processor = processor
        self.postprocessor = Resize()

    def __call__(self, img):
        # Обновляем итоговый размер, если входной размер изменился
        im_size = img.shape[:2]
        if tuple(im_size) != tuple(self.postprocessor.im_size):
            self.postprocessor.reset_im_size(im_size)

        # Применяем всю цепь преобразований:
        out = self.preprocessor(img)
        out = self.processor(out)
        return self.postprocessor(out)

    def reset(self):
        if hasattr(self.processor, 'reset'):
            self.processor.reset()


class AddCaption:
    '''
    Наносит контрастный текст на изображение.
    '''
    def __init__(self, text, scale=0.6):
        self.text = text
        self.scale = scale
        self.reset()

    @staticmethod
    def new_mask(text, im_size, scale):

        # Растеризируем текст:
        mask = text2img(text, im_size[:2], scale)

        # Строим альфаканал как дилатированную версию текста:
        mask_alpha = mask.copy()

        # Размазываем по вертикали через сдвиги:
        mask_alpha_1 = np.roll(mask_alpha, -1, 0)
        mask_alpha_2 = np.roll(mask_alpha,  1, 0)
        mask_alpha = np.dstack([mask_alpha, mask_alpha_1, mask_alpha_2])
        mask_alpha = mask_alpha.max(-1)

        # Размазываем по горизонтали через сдвиги:
        mask_alpha_1 = np.roll(mask_alpha, -1, 1)
        mask_alpha_2 = np.roll(mask_alpha,  1, 1)
        mask_alpha = np.dstack([mask_alpha, mask_alpha_1, mask_alpha_2])
        mask_alpha = mask_alpha.max(-1)

        # В результате получится белый текст с чёрной обводкой.

        return np.dstack([mask, mask_alpha])

    def __call__(self, img):
        # Если размер изображения изменился, или это первый запуст после сброса:
        if self.mask is None or self.mask.shape[:2] != img.shape[:2]:

            # Сбрасываем маску ещё раз:
            self.reset()

            # Векторизируем текст (создаём предварительную маску):
            mask = self.new_mask(self.text, img.shape[:2], self.scale)

            # Накладываем текст на изображение и фиксируем итоговую маску:
            img, self.mask = overlap_with_alpha(img, mask, True)

        # Если маска уже есть, и она адекватна, то сразу используем её:
        else:
            img = overlap_with_alpha(img, self.mask)

        return img

    def reset(self):
        self.mask = None


class PutTime:
    '''
    Наносит на изображение время, либо (если fps не задан) номер кадра.
    '''

    def __init__(self, fps=None, *args, **kwargs):
        self.fps = fps
        self.args = args
        self.kwargs = kwargs
        self.reset()

    # Превращает секунды в строку в человеческом формате времени:
    @staticmethod
    def seconds2sexagesimal (seconds):
        minutes = int(seconds // 60)
        seconds -= minutes * 60
        text = f'{seconds:06.3f}'
        if not minutes:
            return text

        hours = minutes // 60
        minutes -= hours * 60
        text = f'{minutes:02d}:' + text
        if not hours:
            return text

        days = hours // 24
        hours -= days * 24
        text = f'{hours:02d}:' + text
        if not days:
            return text
        else:
            return f'{days} {text}'

    def __call__(self, img):
        self.frame += 1  # Прирост счётчика кадров

        # Определяем содержимое строки вывода:
        if self.fps:
            text = self.seconds2sexagesimal(self.frame / self.fps)
        else:
            text = str(self.frame)

        # Наносим текст на изображение и возвращаем результат:
        return text2img(text, img, *self.args, **self.kwargs)

    # Сбрасываем счётчик кадров:
    def reset(self):
        self.frame = 0


class Res:
    '''
    Объединяет входное и выходное изображения для заданного вложенного конвейера.
    '''
    def __init__(self, InnerPipeline, mode='h'):
        self.pl = InnerPipeline
        self.mode = mode.lower()

    def __call__(self, image):

        # Получаем входное и выходное изображения:
        inp = image
        out = self.pl(image)

        # Вход и выход должны быть одинакового типа:
        assert inp.dtype == out.dtype

        # Определяем способ объединения изображений в зависимости от параметра mode:
        concat = np.hstack if self.mode=='h' else np.vstack

        # Возвращаем результат объединения:
        return concat([inp, out])


class Mix:
    '''
    Накладывает выходное изображение на входное для заданного вложенного конвейера.
    '''

    def __init__(self, InnerPipeLine, alpha=0.5):
        self.pl = InnerPipeLine
        self.alpha = alpha

    def __call__(self, image):
        # Получаем входное и выходное изображения:
        inp = image
        out = self.pl(image)

        # Вход и выход должны быть одинакового типа и размера
        if inp.dtype != out.dtype:
            raise TypeError(f'{inp.dtype} != {out.dtype}')
        if inp.shape != out.shape:
            raise ValueError(f'{inp.shape} != {out.shape}')

        # Накладываем полупрозрачный выход на вход:
        mix = inp * (1. - self.alpha) + out * self.alpha

        # Возвращаем приведённый к нужному типу результат:
        return mix.astype(inp.dtype)


class Concat:
    '''
    Объединяет входное и выходное изображения для заданного вложенного конвейера.
    '''

    def __init__(self, Pipelines, mode='h'):

        # Pipelines должен быть списком или кортежем:
        assert isinstance(Pipelines, (list, tuple))

        self.pls = Pipelines
        self.mode = mode.lower()

    def __call__(self, image):
        # Применяем все фильтры к исходному изображению:
        outs = [image if f is None else f(image) for f in self.pls]
        # Если вместо фильтра в списке стоит None, ...
        # ... то просто повторяем входное изображение.

        '''
        # Вход и выход должны быть одинакового типа:
        for out in outs[1:]:
            assert outs[0].dtype == out.dtype
        '''

        # Определяем способ объединения изображений в зависимости от параметра mode:
        concat = np.hstack if self.mode=='h' else np.vstack

        # Возвращаем результат объединения:
        return concat(outs)


class CompareTwoFilters:
    '''
    Выводит демо-коллаж из четырёх изображений в след. виде:
    <Входное изображение             > <Выходи из Filter1>
    <diff_func от выходов Filter1 и 2> <Выходи из Filter2>

    Полезно для отладки и сравнения фильтров.
    '''

    def __init__(self, filter1, filter2, diff_func=cv2.absdiff, name='Comparator'):
        self.f1 = filter1
        self.f2 = filter2
        self.diff = diff_func
        self.name = name

    def __call__(self, img):
        out1 = self.f1(img)
        out2 = self.f2(img)
        diff = self.diff(out1, out2)

        return np.vstack([np.hstack([img , out1]),
                          np.hstack([diff, out2])])

    # Сброс внутренних состояний:
    def reset(self):
        for f in [self.f1, self.f2]:
            if hasattr(f, 'reset'):
                f.reset()


class CompareFiltersWithTarget:
    '''
    Выводит демо-коллаж из изображений в след. виде:
    <Входное изображение  > <Выходи из Filter1                     > ... <Выходи из FilterN                     >
    <Эталонное изображение> <diff_func от выходов Filter1 и элалона> ... <diff_func от выходов FilterN и элалона>

    На вход должен подаваться коллаж в след. виде:
    <Входное изображение  >
    <Эталонное изображение>
    (изображения должны распологаться одно под
    другим и иметь одинаковый размер по высоте).

    Полезно для отладки и сравнения фильтров.
    '''

    def __init__(self, filters, diff_func=cv2.absdiff, name='TargetComparator'):
        self.filters = filters
        self.diff = diff_func
        self.name = name

    def __call__(self, img):

        # Разделяем изображение на входное и эталонное:
        inp, out = np.vsplit(img, 2)

        # Выполняем фильтрацию:
        preds = [f(inp) for f in self.filters]

        # Вычисляем отличия от эталона:
        diffs = [self.diff(out, pred) for pred in preds]

        # Собираем в коллаж и возвращаем:
        return np.vstack([np.hstack([inp, *preds]),
                          np.hstack([out, *diffs])])

    # Сброс внутренних состояний:
    def reset(self):
        for f in [self.f1, self.f2]:
            if hasattr(f, 'reset'):
                f.reset()


class KerasModel:
    '''
    # Использование keras-модели как фильтра.
    '''

    def __init__(self, model, name='KerasModel'):
        self.model = model
        self.name = name
        self.reset()
        self.stateful = model.stateful if hasattr(model, 'stateful') else None

    def __call__(self, image):
        return self.model.predict(np.expand_dims(image, 0), verbose=0)[0, ...]

    def reset(self):
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
        # В Keras3 нет reset_states для моделей
        # https://github.com/keras-team/keras/issues/18467#issuecomment-2096448735

    def close(self):
        if hasattr(self.model, 'stateful'):
            self.model.stateful = self.stateful
        self.reset()


class StoreLastFrames:
    '''
    # Возвращает не только текущий кадр, но и n-1 предыдущих.
    '''
    def __init__(self,
                 n                 : 'Число хранимых кадров'=3,
                 fill_empty_frames : 'Заполнять нулями отсутствующие кадры'=None,
                 name              : 'Имя фильтра'=None):
        self.n = n

        if fill_empty_frames and not fill_empty_frames.lower() == 'none':
            self.fill_empty_frames = fill_empty_frames.lower()
        else:
            self.fill_empty_frames = None

        self.name = name if name else 'StoreLast%sFrames' % n
        self.reset()

    def __call__(self, image):
        self.frames.insert(0, image)
        if len(self.frames) == 1 and self.n > 1 and self.fill_empty_frames:
            if self.fill_empty_frames == 'zeros':
                self.frames += [np.zeros_like(image)] * (self.n - 1)
            elif self.fill_empty_frames == 'copy':
                self.frames += [image] * (self.n - 1)

        while len(self.frames) > self.n:
            self.frames.pop()

        return self.frames

    def reset(self):
        self.frames = []


class DrawSemanticSegments:
    '''
    Рисует цветные сегменты для семантической сегментации.

    Если сегментаторм задан, то исходное изображение раскрашивается цветами
    сегментов. Если сегментатор не указан, то входящее изображение принимается
    за результат сегментации и сегменты отрисовываются без фона.

    Если результат сегментации уже пропущен через argmax, то требуется задать
    число классов (num_classes). Если число классов не задано, то к результату
    сегментации будет применён argmax для последнего измерения.

    Если num_classes не задан, а число каналов тензора сегментации = 1, то вся
    постобработка будет исходить из задачи бинарной классификации с порогом =
    0.5.
    '''

    def __init__(self,
                 num_classes=None,
                 segmentator=None):
        self.num_classes = num_classes
        self.segmentator = segmentator

    def __call__(self, img):

        # Фиксируем размер изображения:
        im_size = img.shape[:2]

        # Инициируем выходное изображение:
        target_shape = list(im_size) + [3]
        out = np.zeros(target_shape, np.uint8)

        # Если входнй тензор является исходным изображением:
        if self.num_classes:

            # Тип выходного изображения берём по аналогии со входным:
            dtype = img.dtype

            # Яркость берём из исходного изображения:
            if img.ndim == 3:
                if img.shape[2] == 3:
                    V = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 1:
                    V = img[..., 0]
                else:
                    raise ValueError(
                        f'Неожиданный размер входного изображения: {img.shape}'
                    )
            elif img.ndim == 2:
                V = img
            else:
                raise ValueError(
                    f'Неожиданный размер входного изображения: {img.shape}')

        else:
            dtype = np.uint8  # Тип выходного изображения
            V = 255           # Используем максимальную яркость

        # Применяем сегментатор или берём готовый результат из входа:
        segments = img if self.segmentator is None else self.segmentator(img)

        # Применяем к тензору сегментов простобработку, если она нужна:
        if not self.num_classes:

            # Если используется бинарная сегментация:
            if segments.ndim == 2 or segments.shape[2] == 1:
                segments = segments > 0.5             # Пороговая обработка
                segments = segments.reshape(im_size)  # shape: HW1 -> HW
                segments = segments.astype(np.int8)   # Bool -> Int
                num_classes = 2

            # Если классов больше двух:
            elif segments.ndim == 3 and segments.shape[2] > 2:
                num_classes = segments.shape[2]
                segments = np.argmax(segments, axis=-1)

            else:
                raise ValueError('Неожиданный размер тензора сегментов: ' + \
                                 segments.shape)

        else:
            num_classes = self.num_classes

        # Определяем оттенок (H) по результатам сегментации:
        H = segments / num_classes
        if dtype == np.uint8:
            H = (H * 180).astype(np.uint8)
        else:
            H = H * 360

        # Для выходного изображения используется цветовое пространство HSV:
        out[..., 0] = H
        out[..., 2] = V
        out[..., 1] = 255 if dtype == np.uint8 else 1.

        # Возвращаем результат, сконвертированный из HSV в RGB:
        return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)

    def reset(self):
        if hasattr(self.segmentator, 'reset'):
            self.segmentator.reset()


class StreamRandomCrop:
    '''
    # Вырезает случайный фрагмент из целой видеопоследовательности.
    # Т.е. рамка меняет своё положение только при вызове reset или изменении размера входного изображения.
    '''
    def reset(self):
        self.im_size = None
    
    def __init__(self, size = 256):
        size = np.array(size)
        if not hasattr(size, 'size') or size.size == 1:
            size = np.array([size] * 2)[:]
        self.size = size
        self.reset()
    
    def __call__(self, image):
        im_size = np.array(image.shape[:2])
        if not np.all(np.equal(self.im_size, im_size)):
            self.im_size = im_size
            
            isint = np.issubdtype(self.size.dtype, np.uint64) or np.issubdtype(self.size.dtype, np.int64)
            isfloat = np.issubdtype(self.size.dtype, np.float64)
            if isint and not isfloat:
                if np.any(self.im_size < self.size):
                    raise ValueError('Размер изображения должен быть больше-равен %dx%d.' % tuple(self.size))
                self.true_size = self.size.copy()
            elif not isint and isfloat:
                if np.any(self.size > 1.) or np.any(self.size <= 0.):
                    raise ValueError('Размер обрезки должен быть в интервале (0, 1] от размера исходного изображения.')
                self.true_size = (self.im_size * self.size).astype(self.im_size.dtype)
            else:
                raise ValueError('Параметр size должен быть вещественного или целочисленного типа')
            
            self.di = np.random.randint(self.im_size[0] - self.true_size[0])
            self.dj = np.random.randint(self.im_size[1] - self.true_size[1])
        
        return image[self.di:self.di + self.true_size[0], self.dj:self.dj + self.true_size[1], ...]


class StreamRandomFlip:
    '''
    # Отражает и/или поворачивает изображение (применяет обратимую аугментацию).
    # Т.е. параметры преобразования меняются только при вызове reset или изменении размера входного изображения.
    '''
    def reset(self):
        self.im_size = None
        self._ud  = np.random.choice([True, False]) if self.ud  is None else self.ud
        self._lr  = np.random.choice([True, False]) if self.lr  is None else self.lr
        self._rot = np.random.randint(4)            if self.rot is None else self.rot
    
    def __init__(self, ud=None, lr=None, rot=None):
        '''
        Отражение и поворот. Если параметр == None, то 
        преобразование включается случайно для каждой посделовательности.
        '''
        self.ud = ud
        self.lr = lr
        self.rot = rot
        self.reset()
    
    def __call__(self, image):
        im_size = np.array(image.shape[:2])
        if not np.all(np.equal(self.im_size, im_size)):
            self.reset()
            self.im_size = im_size
        
        if self._ud:
            image = np.flipud(image)
        if self._lr:
            image = np.fliplr(image)
        if self._rot:
            image = np.rot90(image, self._rot)
        
        return image


# RGB/BGR <-> YUV:
def rgb2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
def yuv2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
def yuv2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
bgr2rgb = rgb2bgr

# RGB/BGR <-> Gray:
def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def gray2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
def gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# uint8 <-> float32:
def im2double(img):
    return img.astype(np.float32) / 255.
def double2im(double):
    return (double * 255).astype(np.uint8)


def color_canny(image):
    '''
    Трёхканальный canny
    '''
    frame0 = feature.canny(image[:, :, 0])
    frame1 = feature.canny(image[:, :, 1])
    frame2 = feature.canny(image[:, :, 2])
    return 1 - np.stack([frame0, frame1, frame2], -1)


def apply2video(im_filter      : 'Фильтр, применяемый к цветному изображению'                             ,
                inp_file       : 'Обрабатываемый файл'                                             = None ,
                out_file       : 'Файл для записи'                                                 = None ,
                save2subfolder : 'Имя файла для записи = имя обробатываемого файла/имя модели.avi' = False,
                verbose        : 'Вывод статусбара'                                                = True ,
                rescale        : 'Масштабировать изображение до HD'                                = True ,
                step           : 'Шаг с которым прореживается видео'                               = 1    ,
                skip_existed   : 'Пропуск уже существующих файлов'                                 = True ):
    '''
    Применяет какую-то функцию обработки изображений к видео
    '''
    default_inp_dir = '/home/user/work/shubin/Test/Video'
    default_out_dir = '/home/user/work/shubin/Results/Video'
    
    # Файл для чтения
    if not inp_file:
        inp_file = default_inp_dir
    if os.path.isdir(inp_file): # Если это папка - обрабатываем все файлы в ней
        for file in os.listdir(inp_file):
            inp_file_ = os.path.join(inp_file, file)
            if out_file:
                out_file_ = os.path.join(out_file, file)
            else:
                out_file_ = os.path.join(default_out_dir, file)
            apply2video(im_filter,
                        inp_file_,
                        out_file_,
                        save2subfolder,
                        verbose,
                        rescale,
                        step,
                        skip_existed)
        return
    
    if save2subfolder:
        # Файл для записи
        out_flie_basename = im_filter.name if hasattr(im_filter, 'name') else 'NoName_Filter'
        out_file_basename = out_flie_basename + '.avi'
        if out_file:
            if os.path.isdir(out_file) or out_file[-4:] in ['.mp4', '.avi']:
                out_file = os.path.join(out_file, out_file_basename)
        else:
            out_file = os.path.join(default_out_dir, out_file_basename)
    else:
        inp_file_ext = os.path.splitext(inp_file)[1]
        if inp_file_ext.lower() not in ['.avi', '.mp4', '.mpeg', '.mkv', '.mov']:
            return
        im_filter_name = im_filter.name if hasattr(im_filter, 'name') else 'NoName_Filter'
        if out_file: # Если выходной файл(папка) задан(а) явно
            if out_file[-4:] in ['.mp4', '.avi', '.mkv', '.mov'] and not os.path.isdir(out_file): # Если это, видимо, файл
                #out_file = out_file[:-4] + '.avi' # Заменяем разрешение на доступное для записи
                out_path = os.path.dirname(out_file)
            elif os.path.isdir(out_file): # Если это точно директория
                out_path = os.path.dirname(out_file)
            else:
                raise ValueError('Непредвиденная ошибка параметра out_file="%s".' % out_file)
        
        else:        # Если выходной файл не задан
            out_file = os.path.dirname(default_out_dir)
    
    out_file = out_file[:-3] + 'avi'
    
    # Пропуск существующих файлов, если надо
    if skip_existed and os.path.exists(out_file):
        return
    
    # Создаём необходимые вложенные папки
    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        #print(out_path)
        mkdirs(out_path)
    
    # Файлы чтения и записи
    imp = cv2.VideoCapture(inp_file)
    if rescale:
        if rescale == True:
            target_size = (1280, 720) # Размер кадра
        elif isinstance(rescale, int):
            target_size = (round(imp.get(cv2.CAP_PROP_FRAME_WIDTH ) / rescale) * rescale,
                           round(imp.get(cv2.CAP_PROP_FRAME_HEIGHT) / rescale) * rescale)
        elif isinstance(rescale, (list, tuple)):
            target_size = tuple(rescale)
        else:
            raise ValueError('Параметр rescalse задан неверно.')
    else:
        target_size = (int(imp.get(cv2.CAP_PROP_FRAME_WIDTH )),
                       int(imp.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(out_file,
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          #cv2.VideoWriter_fourcc(*'RGBA'),
                          imp.get(cv2.CAP_PROP_FPS) / step,
                          target_size)
    
    if (imp.isOpened() == False):
        print('Ошибка открытия файла "%s"' % inp_file)
    
    # Сброс состояния детектора границы
    if hasattr(im_filter, 'reset'):
        im_filter.reset()
    
    #while(imp.isOpened()):
    total_frames = int(imp.get(cv2.CAP_PROP_FRAME_COUNT)) // step
    for frame_ind in tqdm(range(total_frames), inp_file, disable=not verbose):
        
        # Пропуск ряда кадров
        if step != 1:
            imp.set(cv2.CAP_PROP_POS_FRAMES, frame_ind * step)
        
        ret, frame = imp.read()
        
        if ret == True:
            if rescale:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            
            #frame = bgr2yuv(frame)
            frame = bgr2rgb(frame)
            
            frame = im_filter(frame)
            
            if frame.dtype != np.uint8:
                frame[frame < 0] = 0
                frame[frame > 1] = 1
                frame = (frame * 255).astype(np.uint8)
            
            # Работа с каналами
            if len(frame.shape) == 2:                 # Если матрица двумерная...
                frame = np.stack([frame] * 3, -1)     #     ... то дублируем её на 3 канала
            elif len(frame.shape) == 3:               # Если измерений 3 ... 
                if frame.shape[2] == 1:               #     ... но канал 1 ...
                    frame = np.repeat(frame, 3, -1)   #         ... то дублируем его 3 раза
                elif frame.shape[2] == 3:             #     ... и каналов 3 ...
                    frame = rgb2bgr(frame)            #         ... то RGB -> BGR
                    #pass
                else:
                    raise ValueError('Формат кадра (%s) не соответствует ожидаемому.' % (frame.shape))
            else:
                raise ValueError('Формат кадра (%s) не соответствует ожидаемому.' % (frame.shape))
            
            # Запись кадра
            out.write(frame)
    
    # Закрытие файлов
    imp.release()
    out.release()
    
    # Сброс состояния детектора границы
    if hasattr(im_filter, 'reset'):
        im_filter.reset()


class VideoGenerator:
    '''
    Класс, облегчающий чтение кадра из последовательности видео- и/или фото-
    файлов. Может собирать в одну последовательность разные кадры из разных
    файлов. Работает как генератор.
    '''
    # Определяет число кадров в видео:
    @staticmethod
    def get_file_total_frames(file):

        # Если имеется список изображений:
        if isinstance(file, (list, tuple)):
            for file_ in file:

                # Все изображения должны иметь поддерживаемый формат:
                file_ext = os.path.splitext(file_)[-1].lower()
                if file_ext not in cv2_img_exts:
                    raise ValueError('Вложенный список файлов должен ' +
                                     'содержать только изображения. ' +
                                     'Получен файл неподдерживаемого ' +
                                     f'расширения: {file_ext}!')

            # Число кадров для изображений равно числу файлов:
            return len(file)

        # Определяем тип файла:
        file_ext = os.path.splitext(file)[-1].lower()

        # Если файл является изображением, то в нём может быть всего 1 кадр:
        if file_ext in cv2_img_exts:
            return 1

        # Если это видео:
        elif file_ext in cv2_vid_exts:
            vcap = cv2.VideoCapture(file)
            if not vcap.isOpened():
                raise RuntimeError('Ошибка открытия файла "%s"!' % file)
            total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            vcap.release()
            return total_frames

        else:
            raise ValueError('Ожидалось получить путь к видео, фото или ' +
                             f'списку/кортежу фото. Получен: {file}!')

    def __init__(self, files=[], frame_ranges=[]):
        # Инициируем внутренние состояния:
        self.files = []
        self.frame_ranges = []

        # Абстракция для удобного чтения очередного кадра:
        self.im_read_buffer = ImReadBuffer()

        # Наполняем списки переданными значениями:
        self.extend(files, frame_ranges)

    # Добавляет несколько файлов и их диапазоны в общий список:
    def extend(self, files, frame_ranges=[]):
        # Если передан список/кортеж, то кортеж превращаем в список, а
        # список копируем:
        if isinstance(files, (list, tuple)):
            files = list(files)
        # Если передан не список/кортеж, то считаем, что это один файл:
        else:
            files = [files]
            frame_ranges = frame_ranges or [None]

        # Заполняем frame_ranges пустышками, если не задан:
        frame_ranges = frame_ranges or [None] * len(files)

        # frame_ranges и files должны иметь равную длину:
        assert len(frame_ranges) == len(files)

        # Поочерёдно вносим каждый файл с его диапазоном в общий список:
        for file, frame_range in zip(files, frame_ranges):
            self.append(file, frame_range)

    # Добавляет очередной файл и его диапазон в общий список:
    def append(self, file, frame_range=None):
        '''
        frame_ranges для изображений не используется.
        Для видео и списка кадров frame_ranges может быть задан несколькими
        способами:
            1) Объектом класса range. Тогда он сохраняется без изменений
            2) Целое число. Оно принимается как аргумент объекта range.
                Если оно отрицательное, производится предварительный
                перерасчёт относительно длины последовательности.
            3) Список или кортеж. Его элементы принимаются как аргументы
                объекта range, если их 2 или 3, и второй из них является
                отрицательным (указывает на номер с конца, как в случае
                с п.2). Если первый элемент неотрицательный, или элементов
                больше 3х, то сам список/кортеж принимается как перечисление
                номеров кадров (порядок может быть любым).
            4) None или пусткой список/кортеж. В этом случае берутся все
                кадры последовательности.
        '''
        # Если передан путь к папке, то подменяем его списком вложенных в неё
        # изображений:
        if isinstance(file, str) and os.path.isdir(file):
            file = sorted([os.path.join(file, _)
                           for _ in os.listdir(file)
                           if os.path.splitext(_)[1].lower() in cv2_img_exts])

        # Вносим имя очередного файла в список:
        self.files.append(file)

        # Если в качестве файла передан список изображений:
        if isinstance(file, (list, tuple)):

            # Проверяем корректность списка:
            total_frames = self.get_file_total_frames(file)
            is_im_seq = True  # Флаг последовательности изображений

        # Если указан всего один файл:
        else:

            # Определяем тип файла:
            file_ext = os.path.splitext(file)[-1].lower()

            # Если файл является изображением, то в нём лишь один кадр:
            if file_ext in cv2_img_exts:
                self.frame_ranges.append(range(1))
                return

            is_im_seq = False  # Флаг последовательности изображений

        # Если файл является набором изображений или видеофайлом:
        if is_im_seq or (file_ext in cv2_vid_exts):
            # Если frame_range действительно является диапазоном, то вносим
            # без изменений:
            if isinstance(frame_range, range):
                self.frame_ranges.append(frame_range)

            # Если frame_range не указан, берём все кадры исходной
            # последовательности:
            elif frame_range is None:
                total_frames = self.get_file_total_frames(file)
                self.frame_ranges.append(range(total_frames))

            # Если frame_range является списокм или кортежем:
            elif isinstance(frame_range, (list, tuple)):
                # Если frame_range состоит из 2х или 3х элементов, и второй
                # является отрицательным:
                if len(frame_range) in {2, 3} and frame_range[1] < 0:
                    # Собираем диапазон с отсчётом от конца файла:
                    total_frames = self.get_file_total_frames(file)
                    frame_range = range(frame_range[0],
                                        total_frames + frame_range[1],
                                        *frame_range[2:])
                    self.frame_ranges.append(frame_range)

                # Если frame_range не состоит из 2х или 3х элементов, или
                # второй из них не является отрицательным:
                else:
                    # Фоспринимаем список/кортеж как набор номеров кадров:
                    self.frame_ranges.append(frame_range)
            else:
                raise ValueError('Неверное значение параметра ' +
                                 f'"frame_range": {frame_range}!')
        else:
            raise TypeError(f'Неподдерживаемый тип файла: {file_ext}!')
        return

    # Подсчитывает общее число кадров в итоговой последовательности
    def __len__(self):
        return sum(map(len, self.frame_ranges))

    def __iter__(self):
        self.files_ind = 0
        self.frames_iter = iter(self.frame_ranges[self.files_ind])
        return self

    def __next__(self):
        # Определяем номер файла и номер следующего кадра в нём:
        while True:
            try:
                frame = next(self.frames_iter)
                break
            except StopIteration:
                self.files_ind += 1

                # Выходим, если дошли до конца:
                if self.files_ind == len(self.files):
                    self.im_read_buffer.close()
                    raise StopIteration

                self.frames_iter = iter(self.frame_ranges[self.files_ind])

        # Читаем и возвращаем следующий кадр:
        file = self.files[self.files_ind]
        return self.im_read_buffer(file, frame)

    # Для чтения произвольного кадра
    def __getitem__(self, index):

        # Создаём список буферов кадров, если это первый вызов функции:
        if not hasattr(self, 'im_read_buffers'):
            self.im_read_buffers = [ImReadBuffer() for _ in self.files]

        # Определяем номер файла, номер его кадра:
        frame_range_ind = int(index)
        for file, frame_range, im_read_buffer in zip(self.files,
                                                     self.frame_ranges,
                                                     self.im_read_buffers):
            if frame_range_ind < len(frame_range):
                break
            frame_range_ind -= len(frame_range)
        else:
            raise IndexError(f'{index} >= {len(self)}')

        # Читаем сам кадр:
        return im_read_buffer(file, frame_range_ind)

    # Деструктор:
    def __del__(self):
        self.im_read_buffer.close()
        if hasattr(self, 'im_read_buffers'):
            for im_read_buffer in self.im_read_buffers:
                im_read_buffer.close()


def apply2image(im_filter      : 'Фильтр, применяемый к цветному изображению'                             ,
                inp_file       : 'Обрабатываемый файл'                                             = None ,
                out_file       : 'Файл для записи'                                                 = None ,
                save2subfolder : 'Имя файла для записи = имя обробатываемого файла/имя модели.png' = False,
                rescale        : 'Масштабировать изображение до HD'                                = True ,
                skip_existed   : 'Пропуск уже существующих файлов'                                 = True):

    default_inp_dir = '/home/user/work/shubin/Test/Image'
    default_out_dir = '/home/user/work/shubin/Results/Image'

    # Файл для чтения:
    if not inp_file:
        inp_file = default_inp_dir
    if os.path.isdir(inp_file):  # Если это папка - обрабатываем все файлы в ней
        for file in os.listdir(inp_file):
            inp_file_ = os.path.join(inp_file, file)
            if out_file:
                out_file_ = os.path.join(out_file, file)
            else:
                out_file_ = os.path.join(default_out_dir, file)
            apply2image(im_filter,
                        inp_file_,
                        out_file_,
                        save2subfolder,
                        rescale,
                        skip_existed)
        return

    # Файл для записи
    im_filter_name = im_filter.name if hasattr(edge_detector, 'name') else 'NoName Filter'
    if save2subfolder:
        out_file_basename = edge_detector_name + '.png'
        if out_file:
            if os.path.isdir(out_file):
                out_file = os.path.join(out_file, out_file_basename)
        else:
            out_file = os.path.join(default_out_dir, out_file_basename)
    else:
        if out_file:
            if os.path.isdir(out_file):
                out_file = out_file + '.png'
        else:
            out_file = default_out_dir + out_file_basename

    # Пропуск существующих файлов, если надо
    if skip_existed and os.path.exists(out_file):
        return

    inp = io.imread(inp_file)

    while True:
        if rescale:
            if isinstance(rescale, int):
                target_size = (round(inp.shape[1] / rescale) * rescale,
                               round(inp.shape[0] / rescale) * rescale)
            else:
                target_size = (1280, 720)
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)

        if len(inp.shape) == 2:
            inp = np.stack([inp] * 3, -1)
        elif inp.shape[2] != 3:
            inp = np.stack([inp[..., 0]] * 3, -1)

        try:
            # Сброс состояния детектора границы
            if hasattr(edge_detector, 'reset'):
                edge_detector.reset()

            out = edge_detector(inp)
            break
        except RuntimeError:  # Если имеем OOM, то понижаем разрешение изображения в 2 раза по каждой стороне
            target_size = (round(inp.shape[1] / 2),
                           round(inp.shape[0] / 2))
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)

    if out.dtype != np.uint8:
        out = (out * 255).astype(np.uint8)

    # Работа с каналами
    if len(out.shape) == 2:               # Если матрица двумерная...
        out = np.stack([out] * 3, -1)     #     ... то дублируем её на 3 канала
    elif len(out.shape) == 3:             # Если измерений 3 ...
        if out.shape[2] == 1:             #     ... но канал 1 ...
            out = np.repeat(out, 3, -1)   #         ... то дублируем его 3 раза
        elif out.shape[2] == 3:           #     ... и каналов 3 ...
            out = out[:, :, ::-1]         #         ... то RGB -> BGR (так надо для OpenCV)
        else:
            raise ValueError('Формат кадра (%s) не соответствует ожидаемому.' % (out.shape))
    else:
        raise ValueError('Формат кадра (%s) не соответствует ожидаемому.' % (out.shape))

    # Создаём необходимые вложенные папки
    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        mkdirs(out_path)

    io.imsave(out_file, out)


def get_file_list(path, extentions=[]):
    '''
    Возвращает список всех файлов, содержащихся по указанному пути
    (включая поддиректории).
    '''
    # Обработка параметра extentions:
    if isinstance(extentions, str):
        if len(extentions) > 0:
            extentions = {extentions.lower()}
        else:
            extentions = []
    elif isinstance(extentions, (list, tuple, set)):
        for ext in extentions:
            if not isinstance(ext, str):
                raise ValueError('extentions должен быть строкой, или ' +
                                 'списком/кортежем/множеством строк. ' +
                                 f'Получен элемент {ext}')
    else:
        raise ValueError('extentions должен быть строкой, или эсписком/' +
                         'кортежем/множеством строк. Получен %s' % extentions)
    extentions = [ext.lower() for ext in extentions]

    # Составление списка файлов:
    file_list = []
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isdir(file):
            file_list += get_file_list(file, extentions)
        elif not len(extentions) or \
                os.path.splitext(file)[1][1:].lower() in extentions:
            file_list.append(file)

    return file_list


def convert_videos(inp_path, out_path, skip_existed=True):
    '''
    Пересжатие всех видеофайлов заданной директории в другую директорию.
    Используется для подготовки видеорезультатов к демонстрации.
    '''

    file_list = get_file_list(inp_path)
    num_files = len(file_list)
    for ind, inp_file in enumerate(file_list, 1):
        inp_file_path_name, inp_file_ext = os.path.splitext(inp_file)

        file_status_template = '%03d/%03d) ' % (ind, num_files) + \
            '%15s: ' % inp_file[len(inp_path):]
        if inp_file_ext.lower() in ['.avi', '.mpg', '.mpeg', '.mp4', 'mkv']:
            inp_file_rel_path = inp_file_path_name[len(inp_path):]
            while inp_file_rel_path[0] == '/':
                inp_file_rel_path = inp_file_rel_path[1:]
            out_file = os.path.join(out_path, inp_file_rel_path + '.mp4')

            if skip_existed and os.path.exists(out_file):
                print(file_status_template % 'Уже существует')
                continue

            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            print(file_status_template % 'Обрабатывается', end='\r')
            recomp2mp4(inp_file, out_file)
            print(file_status_template % 'Обработан')
        else:
            print(file_status_template % 'Пропущен')
    return


class OptFlow:
    '''
    Обвязка вокруг cv2.calcOpticalFlowFarneback.
    Позволяет выполнять различные операции на базе оптического потока.
    '''
    def __init__(self                                       ,
                 pyr_scale  = 0.5                           ,
                 levels     = 1                             ,
                 winsize    = 40                            ,
                 poly_n     = 5                             ,
                 poly_sigma = 1.1                           ,
                 iterations = 10                            ,
                 flags      = cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        self.pyr_scale  = pyr_scale
        self.levels     = levels
        self.winsize    = winsize
        self.poly_n     = poly_n
        self.poly_sigma = poly_sigma
        self.iterations = iterations
        self.flags      = flags

    def __call__(self, img1, img2, flow=None, flags=None):
        '''
        Оптический поток для двух изображений.
        '''
        # Переводим цветные изображения в оттенки серого, если надо:
        if img1.ndim > 2 and img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        if img2.ndim > 2 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # Берём флаги из входного параметра, если он задан:
        flags = self.flags if flags is None else flags

        # Если на вход передана начальная аппроксимация потока, то
        # указываем во флагах необходимость её использовать:
        if flow is not None:
            flags = flags | cv2.OPTFLOW_USE_INITIAL_FLOW

            # Чтобы не писать разультат во входную переменную:
            flow = flow.copy()

        # Вычисляем сам опт. поток:
        return cv2.calcOpticalFlowFarneback(img1                        ,
                                            img2                        ,
                                            flow                        ,
                                            pyr_scale  = self.pyr_scale ,
                                            levels     = self.levels    ,
                                            winsize    = self.winsize   ,
                                            poly_n     = self.poly_n    ,
                                            poly_sigma = self.poly_sigma,
                                            iterations = self.iterations,
                                            flags      = flags          )

    # Строит координатную сетку для деформации изображения:
    @staticmethod
    def create_meshgrid(shape):
        y = np.arange(shape[0], dtype=float)
        x = np.arange(shape[1], dtype=float)
        xv, yv = np.meshgrid(x, y)
        return np.dstack([xv, yv])

    # Деформирует или создаём координатную сетку по опт. потоку:
    @classmethod
    def apply_flow2meshgrid(cls, flow, meshgrid=None):

        # Копируем или создаём исходную сетку:
        if meshgrid is None:
            meshgrid = cls.create_meshgrid(flow.shape)
        assert flow.shape == meshgrid.shape

        # Деформируем сетку:
        return meshgrid - flow

    # Интерполируем каждый канал изображения, согласно деформированной
    # координатной сетке (деформируем изображение по координатной сетке):
    @staticmethod
    def apply_meshgrid2img(meshgrid, img,
                           nearest_interpolation=False,
                           mode='mirror', cval=0.0):
        # Готовим сетку для использования:
        if nearest_interpolation:  # Округляем координаты, если требуется
            meshgrid = np.round(meshgrid)
        meshgrid = [meshgrid[..., 1],
                    meshgrid[..., 0]]

        # Если изображенине одноканально:
        if img.ndim == 2:
            return scipy.ndimage.map_coordinates(img, meshgrid,
                                                 mode=mode, cval=cval)
        # Если изображенине многоканально:
        elif img.ndim == 3:

            # Уточняем значение cval:
            if hasattr(cval, '__len__'):
                if len(cval) == img.shape[2]:
                    cvals = cval
                else:
                    raise ValueError('Параметр "cval" должен быть ' +
                                     'скаляром, либо списком/кортежем ' +
                                     'длиной, соответсвующей числу каналов ' +
                                     f'изображения. Получено: {cval}!')
            else:
                cvals = [cval] * img.shape[2]
                out = []
                for ch, cval in enumerate(cvals):
                    out.append(scipy.ndimage.map_coordinates(
                        img[..., ch], meshgrid, mode=mode, cval=cval))
            return np.dstack(out)
        else:
            raise ValueError('Изображение должно иметь 2 или 3 измерения!')

    @classmethod
    def apply_flow2img(cls, flow, img,
                       nearest_interpolation=False,
                       mode='mirror', cval=0.0):
        '''
        "Восстановление" второго изображения по первому и их опт.потоку.
        '''
        # Строим деформированную координатную сетку:
        meshgrid = cls.apply_flow2meshgrid(flow)

        # Деформируем изображение по координатной сетке:
        return cls.apply_meshgrid2img(meshgrid, img,
                                      nearest_interpolation,
                                      mode, cval)

    def seq_flows(self, imgs, cum_sum=True, **mpmap_kwargs):
        '''
        Вычисляем потоки между соседними кадрами видеопоследовательности.
        '''
        # Рассчёт потока между каждой парой соседних кадров:
        flows = mpmap(self.__call__, imgs[:-1], imgs[1:], **mpmap_kwargs)

        # Если поток отстраивается от первого изображения:
        if cum_sum:

            # Инициируем нулями поток для первого кадра с самим собой:
            flow = np.zeros_like(flows[0])
            cum_flows = [flow]

            # Накапливаем сдвиги для следующих кадров:
            for dflow in flows:
                flow = flow + dflow
                cum_flows.append(flow)

            # Заменяем исходные потоки, потокам с накоплением:
            flows = cum_flows

        return flows

    def apply_flows2img(self, flows, first_img, desc=None, *args, **kwargs):
        '''
        Восстанавливает последовательность кадров, используя
        лишь первый кадр и последовательность опт. потоков.
        '''
        imgs = [first_img.copy()]
        for flow in tqdm(flows, desc=desc, disable=not desc):
            imgs.append(self.apply_flow2img(flow, imgs[-1]))
        return imgs