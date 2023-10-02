import os
import cv2
import numpy as np
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
#from skimage import io, transform, feature, filters, draw

import torch, torchvision

#from tensorflow import keras
#from tensorflow.keras import callbacks, models
#from tensorflow.keras.preprocessing.image import ImageDataGenerator


##############################################
# Работа в конвейере (в т.ч. с видеопотоком) #
##############################################

class Pipeline():
    '''
    # Конвейер фильтров обработки изображений
    '''
    
    # Конкатенация списка функций
    @staticmethod
    def concat(image_filter_list):
        functions, names = [], []
        
        for image_filter in image_filter_list:
            if hasattr(image_filter, '__call__'):         # Если функция
                functions.append(image_filter)
                if hasattr(image_filter, 'name'):
                    names.append(image_filter.name)
                else:
                    names.append('NonameFunction')
            
            elif isinstance(image_filter, (tuple, list)): # Если список/кортеж функций
                functions_, names_ = concat(image_filter)
                functions.extend(functions_)
                names    .extend(names_)
            
            elif isinstance(image_filter, str):           # Если строка
                # Список синонимов
                if image_filter.lower() == 'rgb2gray':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                elif image_filter.lower() == 'rgb2yuv':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2YUV ))
                elif image_filter.lower() in ['rgb2bgr', 'bgr2rgb']:
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2BGR ))
                elif image_filter.lower() == 'yuv2rgb':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_YUV2RGB ))
                elif image_filter.lower() == 'yuv2bgr':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_YUV2BGR ))
                elif image_filter.lower() == 'yuv2gray':
                    functions.append(lambda image: image[..., 0])
                elif image_filter.lower() == 'gray2rgb':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
                elif image_filter.lower() == 'gray2bgr':
                    functions.append(lambda image: cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
                else:
                    raise ValueError('"%s" не входит в список синонимов.' % image_filter) 
                names.append(image_filter)
            
            elif isinstance(image_filter, int):           # Если число
                thresh = float(image_filter)
                functions.append(lambda image: cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1])
                names.append('trheshold_%d' % image_filter)
            
            else:
                raise ValueError('%s не является подходящим элементом конвеера.' % image_filter)
        
        return functions, names
    
    def __init__(self, image_filter_list, name='Pipeline'):
        self.functions, self.names = self.concat(image_filter_list)
        self.name = name
    
    def __call__(self, image):
        for image_filter in self.functions:
            image = image_filter(image)
        return image
    
    def reset(self, im_size=None):
        for image_filter in self.functions:
            if hasattr(image_filter, 'reset'):
                image_filter.reset()


class ViRead:
    '''
    Возвращает посделовательность кадров видео из файла.
    Работает и как функция и как генератор.
    '''
    def __init__(self, path, start_frame=0, colorspace='rgb', on_end='reset'):
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
            self.colorspace_converter = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2YUV )
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


class KerasNN:
    '''
    # Использование keras-модели как фильтра.
    '''
    def __init__(self, model, name='KerasNeuralNetwork'):
        self.model = model
        self.name = name
        self.reset()
        self.stateful = model.stateful
    
    def __call__(self, image):
        out = self.model.predict(np.expand_dims(image, 0))[0, ...]
        
        #out[out < 0] = 0
        #out[out > 1] = 1
        
        out = (out * 255).astype(np.uint8)
        
        return out
    
    def reset(self):
        self.model.reset_states()
        
    def close(self):
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


# RGB/BGR <-> YUV черес cv2
def rgb2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
def yuv2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
def bgr2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
def rgb2yuv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
bgr2rgb = rgb2bgr


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
        os.makedirs(out_path)
    
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
    if os.path.isdir(inp_file): # Если это папка - обрабатываем все файлы в ней
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
        except RuntimeError: # Если имеем OOM, то понижаем разрешение изображения в 2 раза по каждой стороне
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
        os.makedirs(out_path)
    
    io.imsave(out_file, out)


def get_file_list(path, extentions=[]):
    '''
    Возвращает список всех файлов, содержащихся по указанному пути (включая поддиректории).
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
                raise ValueError('extentions должен быть строкой, или списком/кортежем/множеством строк. Получен элемент %s' % ext)
    else:
        raise ValueError('extentions должен быть строкой, или списком/кортежем/множеством строк. Получен %s' % extentions)
    extentions = [ext.lower() for ext in extentions]    
    
    # Составление списка файлов:
    file_list = []
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if os.path.isdir(file):
            file_list += get_file_list(file, extentions)
        elif not len(extentions) or os.path.splitext(file)[1][1:].lower() in extentions:
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
        
        file_status_template = '%03d/%03d)' % (ind, num_files) + ' %15s: ' + inp_file[len(inp_path):]
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
            if 'gray' in inp_file_path_name:
                crf = 40
            else:
                crf = 20
            
            if os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel quiet -i "%s" -c:v libx264 -preset slow -crf %d -tune animation "%s"' % (inp_file, crf, out_file)):
                if os.path.exists(out_file):
                    os.remove(out_file)
                OSError(file_status_template % 'Ошибка обработки')
            else:
                print(file_status_template % 'Обработан')
        else:
            print(file_status_template % 'Пропущен')
    return