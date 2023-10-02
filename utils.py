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
import tempfile

import numpy as np

from functools import reduce
from zipfile import ZipFile
from shutil import rmtree
from tqdm import tqdm
from time import time
from multiprocessing import pool, Pool
from IPython.display import clear_output, HTML, Javascript, display
from sklearn.model_selection import train_test_split
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


def df2img(df, file=None, title='index'):
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
    table = ax.table(df.values, rowLabels=df.index, colLabels=df.columns, cellLoc='center', loc='upper left')
    if title:
        ax.set_title(df.index.name if title=='index' else title, fontweight="bold", loc='left')
    
    # Первичное созранение изображения таблицы:
    plt.savefig(file,
                bbox_inches='tight',
                transparent=True,
                dpi=200)
    
    img = plt.imread(file) # Чтение первичного изображения
    img = autocrop(img)    # Обрезка полей
    
    # Удаляем временный файл или сохраняем окончательный вариант:
    if rm_file:
        rmpath(file)
    else:
        plt.imsave(file, img)
    
    plt.close()
    
    return img


def fig2nparray(fig=None):
    '''
    Возвращает содержимое фигуры из Matplotlib в виде numpy-массива RGB-изображения.
    Бездумно взято отсюда: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
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


def draw_contrast_text(image, text):
    '''
    Делает многострочные контрастные надписи на заданном изображении.
    '''
    # Разбиваем текст на строки и отрисовываем каждую в отдельности:
    for line_ind, line in enumerate(text.split('\n'), 1):
        
        # Рисуем тёмную обводку вокруг строки:
        for i in [-1, 1]:
            for j in [-1, 1]:
                image = cv2.putText(image, line, (i, j + 20 * line_ind), cv2.FONT_HERSHEY_COMPLEX, 0.6, (  0,   0,   0), 1, cv2.LINE_AA)

        # Рисуем саму белую строку:
        image         = cv2.putText(image, line, (i, j + 20 * line_ind), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
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
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)


def img_dir2video(img_dir, video_file='preview.avi', tmp_file=None, desc=None, imsize=(1080, 1920), fps=5, rm_after_add=False):
    '''
    Сборка видео из всек изображений в папке.
    Может объединять изображения разных размеров,
    сохраняя соотношение сторон за счёт паддинга.
    Используется для превью.
    '''
    if tmp_file is None:
        video_file_name, video_file_ext = os.path.splitext(video_file)
        tmp_file = video_file_name + '_tmp' + video_file_ext
    
    # Сортированный по имени список изображений:
    images = sorted(os.listdir(img_dir))
    
    # Отбрасываем все файлы, не являющиеся изображениями:
    images = [image for image in images if os.path.splitext(image)[-1].lower() in cv2_img_exts]
    
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
            
            # Уточняем путь до файла:
            file = os.path.join(img_dir, file)
            
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
        except:
            print(f'Пропущена запись кадра "{file}" в видео "{video_file}"!')
            continue
    
    # Закрываем записанный видеофайл:
    out.release()
    
    # Формируем команду конвертации:
    cmd_line = f'ffmpeg -i "{tmp_file}" -y -hide_banner -c:v libx264 -crf 32 -preset slow "{video_file}"'
    
    # Отключаем вывод, если нужно:
    if desc is None: cmd_line += '>/dev/null 2>&1'
    
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
        self.file      = None # Текущий открытый файл
        self.vcap      = None # Объект открытого видеофайла
        self.img       = None # Текущий загруженный кадр
        self.frame_num = None # Номер текущего загруженного кадра (для видео)
    
    def __init__(self):
        # Задаём внутенние состояния по умолчанию:
        self.reset_state()
    
    def __call__(self, file, frame=0):

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
                self.close() # Сбрасываем внутренние состояния
                self.img = cv2.imread(file)
                self.file = file
        
        # Если файл является видеопоследовательностью:
        elif file_ext in cv2_vid_exts:
            
            # Если текущее видео ещё не загружалось, или номер ...
            # ... загруженного кадра больше номера текущего кадра:
            if file != self.file or frame < self.frame:
                self.close() # Сбрасываем внутренние состояния
                
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
        
        # Если файл не является изображением или видео:
        else:
            raise TypeError(f'Файл "{file}" не является ни изображением, ни видео!')
        
        # Возвращаем загруженное изображение:
        return self.img
    
    # Освобождает ресурсы:
    def close(self):
        if self.vcap:
            self.vcap.release() # Закрываем открытый видеофайл
            self.reset_state()  # Сбрасываем внутенние состояния
    
    def __enter__(self):
        return self 
    
    def __exit__(self, type, value, tb):
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


def rmpath(path):
    '''
    Удаляет файл или папку вместе с её содержимым.
    Возвращает False, если путь не существовал.
    '''
    try:
        if os.path.isdir(path):
            rmtree(path)
            return True

        elif os.path.isfile(path):
            os.remove(path)
            return True

        else:
            return False
    
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
        yaml.safe_dump(obj, stream, allow_unicode=allow_unicode)
    
    return file


def yaml2obj(file='./cfg.yaml', encoding='utf-8'):
    '''
    Читает yaml-файл.
    '''
    with open(file, 'r', encoding=encoding) as stream:
        obj = yaml.safe_load(stream)
    
    return obj


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
                                                       task_batches    ), result._set_length))
    
    return (item for chunk in result for item in chunk)

# Вносим новый метод в старый класс:
pool.Pool.istarmap = istarmap
# Взято с https://stackoverflow.com/a/57364423


def mpmap(func      : 'Функция, применяемая отдельно к каждому элементу списка аргументов'            ,
          *args     : 'Список аргументов'                                                             ,
          num_procs : 'Число одновременно запускаемых процессов. По умолчанию = числу ядер ЦПУ' = 0   ,
          desc      : 'Текст статус-бара. По-умолчанию статус-бар не отображается'              = None):
    '''
    Обрабатывает каждый элемент списка в отдельном процессе.
    Т.е. это некий аналог map-функции в параллельном режиме.
    '''
    
    if len(args) == 0:
        raise ValueError('Должен быть задан хотя бы один список/кортеж аргументов!')
    
    # Если нужно запускать всего 1 процесс одновременно, то обработка будет в текущем процессе:
    if num_procs == 1:
        return list(tqdm(map(func, *args), total=reduce(min, map(len, args)), desc=desc, disable=not desc))
    
    # Если нужен реальный параллелизм:
    with Pool(num_procs if num_procs else None) as p:
        if len(args) > 1:
            total = reduce(min, map(len, args))
            args = zip(*args)
            pmap = p.istarmap
        
        else:
            args = args[0];
            total = len(args)
            pmap = p.imap
        
        return list(tqdm(pmap(func, args), total=total, desc=desc, disable=desc is None))


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


###############
# ML-утилиты: #
###############


def soft_train_test_split(*args, test_size, random_state=0):
    '''
    Разделение данных на две подвыборки. Аналогичен train_test_split,
    но работает и при экстремальных случаях вроде нулевой длины
    выборки или нулевого размера одного из итоговых подвыборок.
    '''
    
    try:
        return train_test_split(*args, test_size=test_size, shuffle=True, random_state=random_state)
    
    # Если случай экстремальный:
    except ValueError:
        
        # Рассчитываем относительный размер тестовой выборки:
        if isfloat(test_size):
            test_size_ = test_size
        elif len(args[0]) > 0:
            test_size_ = test_size / len(args[0])
        else:              # Если выборка нулевой длины, то ...
            test_size_ = 1 # ... избегаем деления на ноль.
        
        # Если тестовая выборка должна быть больше проверочной - отдаём всё ей:
        if test_size_ > 0.5:
            return flatten_list(zip([type(arg)() for arg in args], args))
        
        # Иначе всё обучающей:
        else:
            return flatten_list(zip(args, [type(arg)() for arg in args]))


def train_val_test_split(*args, val_size=0.2, test_size=0.1, random_state=0):
    '''
    Режет выборку на обучающую, проверочную и тестовую.
    '''
    
    # Если на входе пустой список, возвращаем 3 пустых списка:
    if len(args[0]) == 0:
        return [], [], []
    
    # Величины val_size и test_size должны адекватно соотноситься с размером выборки:
    if isint(val_size) and isint(test_size):
        assert val_size + test_size <= len(args[0])
    if isfloat(val_size) and isfloat(test_size):
        assert val_size + test_size <= 1.
    
    # Получаем тестовую выборку:
    trainval_test = soft_train_test_split(*args, test_size=test_size, random_state=random_state)
    train_val = trainval_test[ ::2]
    test      = trainval_test[1::2]
    
    # Если val_size задан целым числом, то используем его как есть:
    if isint(val_size):
        val_size_ = val_size
    
    # Если val_size - вещественное число, то долю надо перерасчитать:
    elif isfloat(val_size):
        
        # Если при этом test_size целочисленного типа, то переводим его в дроби:
        if isint(test_size):
            test_size = test_size / len(args[0])
        
        # Перерасчитываем val_size с учётом уменьшения ...
        # ... выборки после отделения тестовой стоставляющей:
        val_size_ = val_size / (1. - test_size) if test_size < 1. else 0
        
    else:
        raise ValueError(f'Неподходящий тип "{type(val_size)}" переменной val_size!')
    
    # Разделяем оставшуюся часть выборки на обучающую и проверочную:
    
    train_val = soft_train_test_split(*train_val, test_size=val_size_, random_state=random_state)
    train     = train_val[ ::2]
    val       = train_val[1::2]
    
    return flatten_list(zip(train, val, test))


###################
# Другие утилиты: #
###################


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


def flatten_list(list_of_lists, depth=1):
    '''
    Уменьшает глубину вложенности списков.
    Т.е.:
    list_sum([[1, 2], [3, 4]]) = [1, 2, 3, 4]
    '''
    new_list_of_lists = []
    for list_ in list_of_lists:
        new_list_of_lists += list_
    
    if depth > 1:
        new_list_of_lists = flatten_list(new_list_of_lists, depth=depth - 1)
    
    return new_list_of_lists


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
    os.system('cls' if os.name=='nt' else 'clear')
    clear_output(wait=True)


class TimeIt():
    '''
    Контекст, засекающий время выполнения вложенного кода в секундах.
    
    Пример использования:
    with TimeIt('генерацию случайнных чисел') as t:
        np.random.rand(10000000)
    print(t())
    '''
    def __init__(self, title=None):
        self.title=title
    
    def __enter__(self):
        self.start = time()
        return self 
    
    def __exit__(self, type, value, tb):
        self.time_spent = time() - self.start
        if self.title:
            print('На %s потрачено %.6s секунд.' % (self.title, self.time_spent))
    
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
    def __init__(self                                          ,
                 start_annotation : 'Предворяющий текст' = None,
                 end_annotation   : 'Завершающий  текст' = None):
        
        self.start_annotation = start_annotation # Предворяющий текст
        self.  end_annotation =   end_annotation # Завершающий  текст
        
        # Если заверщающий текст короче предворяющего,
        # то дополняем длину пробелами, чтобы затереть:
        self.end_annotation += ' ' * max(0, len(start_annotation) - len(end_annotation))
    
    def __enter__(self):
        print(self.start_annotation, end='')
        return
    
    def __exit__(self, type, value, tb):
        print('\r' + self.end_annotation)


def isint(a):
    '''
    Возвращает True, если число или numpy-массив является целочисленным.
    '''
    return isinstance(a        , int       ) or  \
           isinstance(a        , np.ndarray) and \
           isinstance(a.flat[0], np.integer)


def isfloat(a):
    '''
    Возвращает True, если число или numpy-массив является вещественным.
    '''
    return isinstance(a        , float      ) or  \
           isinstance(a        , np.ndarray ) and \
           isinstance(a.flat[0], np.floating)


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
__all__ += ['train_val_test_split', 'rim2arabic', 'restart_kernel_and_run_all_cells']
__all__ += ['isint', 'isfloat']

