'''
********************************************
*         Работа с CVAT-датасетами.        *
*                                          *
* Из всех используемых сейчас датасетов    *
* (CVAT, GG и CG) формат, используемый в   *
* CVAT, является наиболее общим, в связи с *
* чем описанные здесь функции и классы     *
* активно используются в других парсерах.  *
*                                          *
* Сам CVAT-формат задач, который создают   *
* сейчас все используемые в проекте        *
* парсеры датасетов представляет собой     *
* список задач. Каждая задача так же       *
* представляет собой список уже подзачач,  *
* каждая из которых в свою очередь         * 
* является списком или кортежем из трёх    *
* элементов:                               *
*                                          *
*   df - датафремй данных разметки;        *
*   file - путь к изображению/видео;       *
*   true_frames - словарь перехода от      *
*     номеров кадров в CVAT к номерам      *
*     кадров в видео (содержит только те   *
*     кадры, которые используются в данной *
*     подзадаче).                          *
*                                          *
* Задача разбивается на подзадачи,         *
* например, когда подзадачи являются       *
* фрагментами одного видеофайла, либо      *
* серия фотографий одного проекта/полёта.  *
* Т.е. подзадачи каким-либо образом имеют  *
* высокую общность фоноцелевых обстановок, *
* и не должны быть разбиты на обучающую и  *
* проверочную выборки. Для этого их и      *
* группируют в список подзадач, а на       *
* подвыборки разбиваются уже только сами   *
* задачи.                                  *
*                                          *
*                                          *
* Основные функции и классы:               *
*   type(self) - класс, инкапсулирующий    *
*       разметку одного объекта. Позволяет *
*       выполнять над разметкой ряд        *
*       преобразований, включя конвертацию *
*       в YOLO-формат и обратно, поворот,  *
*       масштабирование, интерполяцию и    *
*       т.п.                               *
*                                          *
*   cvat_backups2tasks - сам парсер,       *
*       читающий распакованные             *
*       CVAT-бекапы, превращая их в список *
*       задач CVAT-формата.                *
*                                          *
*   drop_bbox_labled_cvat_tasks - ф-ия,    *
*       выбрасывающиая из списка задач те  *
*       из них, что размечены описывающими *
*       прямоугольниками, а не сегментами. *
*                                          *
*   sort_tasks - ф-ия сортировки списка    *
*       задач по имени используемого файла *
*       (нужна для достижения              *
*       воспроизводимости результатов      *
*       конвертации)                       *
*                                          *
********************************************
'''


import os
import cv2
import json
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils import mpmap, ImReadBuffer
from cv_utils import Mask


# Словарь имён и типов полей, которые надо считать из разметки:
df_columns_type = {'track_id'  : object, # 
                   'label'     : str   , # 
                   'frame'     : int   , # 
                   'true_frame': int   , # 
                   'type'      : str   , # 
                   'points'    : object, # 
                   'occluded'  : bool  , # 
                   'outside'   : bool  , # 
                   'z_order'   : int   , # 
                   'rotation'  : float , # 
                   'attributes': object, # 
                   'group'     : int   , # 
                   'source'    : str   , # 
                   'elements'  : object} # 

# Словарь имён полей из разметки и их значений по умолчанию:
df_default_vals = {'track_id'  : None     , # 
                   'label'     : ''       , # 
                   'frame'     : 0        , # 
                   'true_frame': 0        , # 
                   'type'      : 'polygon', # 
                   'points'    : []       , # 
                   'occluded'  : False    , # 
                   'outside'   : False    , # 
                   'z_order'   : 0        , # 
                   'rotation'  : 0.       , # 
                   'attributes': []       , # 
                   'group'     : 0        , # 
                   'source'    : 'manual' , # 
                   'elements'  : []       } # 


def new_df():
    '''
    Создаёт новый датафрейм для видео/изображения:
    '''
    return pd.DataFrame(columns=df_columns_type.keys()).astype(df_columns_type)
    # Задавать тип столбцов нужно, чтобы при конкатенации строк не вылезали предупреждения.


def add_row2df(df=None, **kwargs):
    '''
    Добавляет новую строку с заданнымии параметрами в датафрейм.
    Значения незаданных параметров берутся из df_default_vals.
    '''
    # Создаём датафремй, если он не задан:
    df = df or new_df()
    
    # Создаём новую строку с параметрами по умолчанию:
    row = pd.Series(df_default_vals)
    
    # Заменяем дефолтные значения на входные параметы:
    for key, val in kwargs.items():
        row[key] = val
    
    # Превращаем строку в датафрейм с заданными типами столбцов
    row = pd.DataFrame(row).T.astype(df_columns_type)

    # Возвращаем объединённый (или только новый, если исходный не задан) датафрейм:
    return row if df is None else pd.concat([df, row])


def shape2df(shape    : 'Объект, из которого считываются данные в  первую очередь'       ,
             parent   : 'Объект, из которого считываются данные во вторую очередь' = {}  ,
             track_id : 'ID объекта'                                               = None,
             df       : 'Датафрейм, в который данные нужно добавить'               = None):
    '''
    Вносит в датафрейм инфу о новом объекте.
    '''
    '''
    # Инициируем датафрейм, если это ещё не сделано:
    if df is None:
        df = new_df()
    '''
    # Список извлекаемых значений:
    columns = set(df_columns_type.keys()) - {'true_frame'}
    # "true_frame" исключаем, т.к. он не считывается а вычисляется потом
    
    # Формируем словарь с извлекаемыми значениями:
    row = {column : [shape.get(column, parent.get(column, df_default_vals[column]))] for column in columns}
    row['track_id'] = track_id
    
    # Добавляем строку к датафрейму, если он был задан:
    df = pd.DataFrame(row) if df is None else pd.concat([df, pd.DataFrame(row)])
    
    # Если остались неиспользованные поля (кроме 'shapes'), то выводим ошибку, т.к. это надо проверить вручную:
    unused_params = set(shape.keys()).union(parent.keys()) - set(df_columns_type.keys()) - {'shapes'}
    if len(unused_params):
        raise KeyError('Остались следующие неиспользованные поля: %s' % unused_params)
    
    return df


def cvat_backup_task_dir2task(task_dir):
    '''
    Извлекает данные из подпапки с задачей в бекапе CVAT.
    Возвращает распарсенную задачу в виде списка из
    одного или более кортежей из трёх объектов:
        * DataFrame с разметкой,
        * адрес видео/фотофайла и вектора,
        * вектор номеров кадров в непрореженной последовательности.
    '''
    # Путь к папке 'data' в текущей подпапке:
    data_dir = os.path.join(task_dir, 'data')
    
    # Пропускаем, эсли это не папка:
    if not os.path.isdir(task_dir):
        #print(f'Пропущен "{task_dir}"!')
        return
    
    # Парсим нужные json-ы task и annotations:
    with open(os.path.join(task_dir,        'task.json'), 'r', encoding='utf-8') as f: task_desc   = json.load (f) # Загружаем основную инфу о видео
    with open(os.path.join(task_dir, 'annotations.json'), 'r', encoding='utf-8') as f: annotations = json.load (f) # Загружаем файл разметки
    
    # Загружаем данные об исходных файлах:

    # Если есть manifest.jsonl, то список файлов читаем из него:
    if os.path.isfile(manifest_file := os.path.join(task_dir, 'data', 'manifest.jsonl')):

        # Читаем json-файл:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = [json.loads(l) for l in f]
        
        # Формируем кортеж имён файлов:
        file = tuple([os.path.join(task_dir, 'data', d['name'] + d['extension']) for d in manifest if 'name' in d])

    # Если manifest.jsonl не существует:
    else:

        # Формируем множество файлов в папке data, исключая task.json и annotations.json:
        file = set(os.listdir(os.path.join(task_dir, 'data'))) - {'task.json'} - {'annotations.json'}

        # Множество должно содержать лишь один элемент:
        assert len(file) == 1

        # Его и берём, формируя полный путь к файлу:
        file = os.path.join(task_dir, 'data', file.pop())
    
    task_name = task_desc['name']                # Имя задачи
    jobs      = task_desc['jobs']                # Список подзадач
    data      = task_desc['data']                # Данные текущей задачи
    start = int(data['start_frame' ]           ) # Номер    первого кадра
    stop  = int(data[ 'stop_frame' ]           ) # Номер последнего кадра
    step  = data.get('frame_filter', 'filter=1') # Шаг прореживания кадров
    step  = int(step.split('=')[-1]            ) # Берём число после знака "="
    
    # Номера используемых кадров:
    true_frames = np.arange(start, stop + 1, step)
    true_frames = {key:val for key, val in enumerate(true_frames)}
    
    # Подготавливаем список размеченных фрагментов последовательности для заполнения:
    task = []
    
    # Если список лишь из одного элемента, то берём его вместо списка:
    if len(file) == 1: file = file[0]
    
    # Перебор описаний:
    for job, annotation in zip(jobs, annotations):
        
        # Инициируем список датафреймов для каждой метки перед цилками чтения данных:
        dfs = [new_df()]
        
        # Пополняем список всеми формами текущего описания:
        dfs += [shape2df(shape) for shape in annotation['shapes']]
        
        # Перебор объектов в текущем описании:
        for track_id, track in enumerate(annotation['tracks']):
            
            # Пополняем список сегментами текущего объекта для разных кадров:
            dfs += [shape2df(shape, track, track_id) for shape in track['shapes']]
            
        # Объединяем все датафреймы в один:
        df = pd.concat(dfs)
        # Процесс объединения вынесен из цикла, т.к. он очень затратен.
        
        # Добавление столбца с номерами кадров полного видео (а не прореженного):
        df['true_frame'] = df['frame'].apply(lambda x: true_frames[x])
        
        # Формируем словарь кадров для текущего фрагментов:
        start_frame = job['start_frame'] # Номер  первого   кадра текущего фрагмента
        stop_frame  = job['stop_frame']  # Номер последнего кадра текущего фрагмента
        status      = job['status']      # Статус                 текущего фрагмента
        cur_true_frames = {frame : true_frames[frame] for frame in range(start_frame, stop_frame + 1)}
        
        # Дополняем списки новым фрагментом данных:
        task.append((df, file, cur_true_frames))
    
    return task


def cvat_backups2raw_tasks(unzipped_cvat_backups_dir, desc=None):
    '''
    Формирует список из распарсенных задач из папки с распакованными версиями CVAT-бекапов.
    Постаброботка вроде интерполяции контуров и разбиения на сцены не включена, т.е. задачи сырые.
    '''
    # Список всех дирректорий для парсинга:
    task_dirs = []
    
    # Список папок с распакованными датасетами:
    unzipped_cvat_backups_dirs = os.listdir(unzipped_cvat_backups_dir)
    
    # Перебор по всем распакованным датасетам:
    for ds_ind, cvat_ds_dir in enumerate(unzipped_cvat_backups_dirs):
        
        # Уточняем путь до распакованных датасетов:
        cvat_ds_dir = os.path.join(unzipped_cvat_backups_dir, cvat_ds_dir)

        # Пропускаем, если это не папка:
        if not os.path.isdir(cvat_ds_dir):
            continue
        
        # Название датасета:
        cvat_ds_name = os.path.basename(cvat_ds_dir)
        
        # Перебор всех подпапок внутри датасета:
        for task_dir in os.listdir(cvat_ds_dir):
            
            # Путь к текущей подпапке:
            task_dir = os.path.join(cvat_ds_dir, task_dir)
            
            # Добавляем в список папок на обработку:
            task_dirs.append(task_dir)
    
    # Параллельная обработка данных:
    tasks = mpmap(cvat_backup_task_dir2task, task_dirs,
                  #num_procs=1,
                  desc=desc)
    
    # Выбрасываем пустые задачи:
    tasks = [task for task in tasks if task]
    
    return tasks


def df_list2tuple(df):
    '''
    Переводит все ячейки датафрейма со списками в ячейки с кортежами.
    Используется для хеширования данных.
    '''
    for column in ['points', 'attributes', 'elements']:
        df[column] = df[column].apply(tuple)
    
    return df


def df_tuple2list(df):
    '''
    Переводит все ячейки датафрейма со кортежами в ячейки с списками.
    Используется для восстановления данных после хеширования.
    '''
    for column in ['points', 'attributes', 'elements']:
        df[column] = df[column].apply(list)
    return df


def drop_label_duplicates_in_task(task):
    '''
    Исключает все повторы в разметке текущей задачи.
    '''
    # Итоговый список:
    task_ = []
    
    # Перебор всех подзадач:
    for df, file, true_frames in task:
        
        # Подготовка датафрейма для хеширования данных:
        df_ = df_list2tuple(df)
        
        # Исключение дубликатов строк:
        df_ = df_.drop_duplicates()
        
        # Если дублей не было, то оставляем подзадачу без изменений:
        if len(df_) == len(df):
            task_.append((df, file, true_frames))
        
        # Если дубли были, то возвращаем изменённую подзадачу:
        else:
            df = df_tuple2list(df) # Восстановление датафрейма после хеширования
            task_.append((df, file, true_frames))
    
    return task_


def drop_label_duplicates_in_tasks(tasks, desc=None):
    '''
    Удаляет повторов в разметке всего списка задач.
    '''
    # Параллельная обработка данных:
    tasks = mpmap(drop_label_duplicates_in_task, tasks,
                  #num_procs=1,
                  desc=desc)
    
    return tasks


class CVATPoints:
    '''
    Класс контуров в CVAT.
    Предоставляет набор позезных методов
    для работы с точками контура.
    '''
    
    def __init__(self, points, type_='polygon', rotation=0., imsize=None, rotate_immediately=True):
        
        # Переводим точки в numpy-массив:
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        # Если параметр points является вектором:
        if points.ndim == 1:
            
            # Сохраняем вектор в виде матрицы, где строки соответстуют точкам, а столбцы - коодринатам x и y:
            self.points = points.reshape(-1, 2)
        
        # Если параметр points уже является матрицей:
        elif points.ndim == 2:
            
            # Если столбцов всего 2:
            if points.shape[1] == 2:
                
                # Сохраняем без изменений:
                self.points = points
            
            # Выводим ошибку, если столбцов не 2:
            else:
                raise ValueError('points.shape[1] != 2!')
        
        # Выводим ошибку, если points не матрица и не вектор:
        else:
            raise ValueError(f'Параметр points должен быть либо вектором, либо матрицей (n, 2), а передан: {points}!')
        
        self.type     = type_
        self.rotation = rotation # Градусы
        self.imsize   = imsize   # width, height
        
        # Если есть поворот и его надо применить сразу, то применяем:
        if self.rotation and rotate_immediately:
            self.points = self.apply_rot().points
            self.type = 'polygon'
            self.rotation = 0
    
    # Возвращает центр контура:
    def center(self):
        
        # Отбрасываем повторяющиеся точки, если это многоугольник:
        xy = self.fuse_multipoly() if self.type == 'polygon' else self
        
        # Возвращаем усреднённые значения каждой координаты:
        return np.array([xy.x().mean(), \
                         xy.y().mean()])
    
    # Применяет вращение к конуру на предварительно заданный угол:
    def apply_rot(self):

        if self.rotation % 360 == 0.:
            return type(self)(self.aspolygon(False).points, type='polygon', imsize=self.imsize, rotate_immediately=False)
        
        # Получаем координаты центра контура:
        pvot = self.center()
        
        # Строим матрицу поворота:
        rot = np.deg2rad(self.rotation)
        sin = np.sin(rot)
        cos = np.cos(rot)
        rot_mat = np.array([[cos, sin], [-sin, cos]])
        
        # Принудительно переводим контур в многоугольник:
        points = self if self.type == 'polygon' else self.aspolygon(False)
        
        # Возвращаем контур, повёрнутый в локальной системе координат:
        return type(self)(np.matmul((points.points - pvot), rot_mat) + pvot, imsize=self.imsize, rotate_immediately=False)
    
    # Переводит относительные YOLO-координаты в абсолютные CVAT-координаты:
    def yolo2cvat(self, height=None, width=None):
        
        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:
            
            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')
            
            height, width = self.imsize
        
        # Если объект - прямоугольник:
        if self.type == 'rectangle':
            xc, yc, w, h = self.flatten()
            
            xmin = (xc - w / 2) * width
            ymin = (yc - h / 2) * height
            xmax = (xc + w / 2) * width
            ymax = (yc + h / 2) * height
            
            return type(self)([xmin, ymin, xmax, ymax], 'rectangle', rotation=self.rotation, imsize=(height, width))
        
        # Если объект - многоугольник:
        elif self.type == 'polygon':
            x = self.x() * width
            y = self.y() * height
            
            return type(self)(np.vstack([x, y]).T, 'polygon', rotation=self.rotation, imsize=(height, width))
        
        # Если не прямоугольник, и не многоугольник:
        else:
            raise ValueError(f'Способ конвертации объекта типа "{self.type}" неизвестен!')
    
    # Создаёт полную копию текущего экземлпяра класса:
    def copy(self):
        return type(self)(self.points, self.type, self.rotation, self.imsize)
    
    # Возвращает число вершин контура:
    def __len__(self):
        return len(self.points)
    
    # Превращает матрицу обратно в вектор точек в формате CVAT:
    def flatten(self):
        return self.points.flatten()
    
    # Возвращает сдвинутый список точек:
    def shift_list(self, shift=1):
        return np.roll(self.points, shift, 0)
    
    # Выполняет сдвиг точек:
    def shift(self, shift):
        x0, y0 = shift
        x = self.x() + x0
        y = self.y() + y0
        return type(self)(np.vstack([x, y]).T, self.type, rotation=self.rotation, imsize=self.imsize)
    
    # Выполняет масштабирование точек:
    def scale(self, scale):
        sx, sy = scale
        x = self.x() * sx
        y = self.y() * sy
        
        # Размеры изображения также меняем:
        imsize = None if self.imsize is None else (self.imsize[0] * sy, self.imsize[1] * sx)
        
        return type(self)(np.vstack([x, y]).T, self.type, rotation=self.rotation, imsize=imsize)
    
    # Возвращает координаты векторов последовательного
    # перехода от предыдущей точки контура к следующей:
    def segments_vectors(self):
        return self.shift_list(-1) - self.points
    
    # Возвращает длины сегментов:
    def segments_len(self):
        return np.sqrt((self.segments_vectors() ** 2).sum(1))
    
    # Возвращает точки в обратном порядке:
    def reverse(self):
        return self.points[::-1, :]
    
    # Разворачивает контур в другую сторону:
    def switch_orientation(self):
        return self.shift_list(-1)[::-1, :]
    
    # Возвращает абсциссы всех точек:
    def x(self):
        return self.points[:, 0]
    
    # Возвращает ординаты всех точек:
    def y(self):
        return self.points[:, 1]
    
    # Поэлементная сумма коодринат вершин двух контуров с равным количеством вершин:
    def __add__(self, cvat_points):
        return type(self)(self.points + cvat_points.points, self.type, rotation=self.rotation, imsize=self.imsize)
    
    # Масштабирование величин контура:
    def __mul__(self, alpha):
        return type(self)(self.points * alpha, self.type, rotation=self.rotation,
                          imsize=None if self.imsize is None else tuple(np.array(self.imsize) * alpha))
    
    # Масштабирование величин контура:
    def __rmul__(self, alpha):
        return self * alpha
    
    # Пересечение контуров:
    def __and__(self, cvat_points):
        # Пока действует только для прямоугольников:
        assert self.type == cvat_points.type == 'rectangle'
        
        xmin1, ymin1, xmax1, ymax1 =        self.asrectangle()
        xmin2, ymin2, xmax2, ymax2 = cvat_points.asrectangle()
        
        # Упорядочиваем координаты по возрастанию, если очерёдность перепутана:
        if xmin1 > xmax1: xmin1, xmax1 = xmax1, xmin1
        if ymin1 > ymax1: ymin1, ymax1 = ymax1, ymin1
        if xmin2 > xmax2: xmin2, xmax2 = xmax2, xmin2
        if ymin2 > ymax2: ymin2, ymax2 = ymax2, ymin2
        
        # Определяем пересечение по абсциссе:
        xmin = max(xmin1, xmin2)
        xmax = min(xmax1, xmax2)
        
        # Если пересечения нет, возвращаем None:
        if xmin > xmax:
            return None
        
        # Определяем пересечение по ардинате:
        ymin = max(ymin1, ymin2)
        ymax = min(ymax1, ymax2)
        
        # Если пересечения нет, возвращаем None:
        if ymin > ymax:
            return None
        
        return type(self)([xmin, ymin, xmax, ymax], 'rectangle', imsize=self.imsize)
    
    # Объединение контуров:
    def __or__(self, cvat_points):
        
        # Пока действует только для прямоугольников:
        assert self.type == cvat_points.type == 'rectangle'
        
        xmin1, ymin1, xmax1, ymax1 =        self.asrectangle()
        xmin2, ymin2, xmax2, ymax2 = cvat_points.asrectangle()
        
        # Определяем объединение по абсциссе:
        xmin = min(xmin1, xmin2)
        xmax = max(xmax1, xmax2)
        
        # Определяем пересечение по ардинате:
        ymin = min(ymin1, ymin2)
        ymax = max(ymax1, ymax2)
        
        return type(self)([xmin, ymin, xmax, ymax], 'rectangle', imsize=self.imsize)
    
    # Перерассчитывает контур с учётом вырезания изображения:
    def crop(self, crop_bbox):
        
        # Создаём из параметров кропа новую рамку:
        crop_bbox = type(self)(crop_bbox, 'rectangle', imsize=(crop_bbox[-1], crop_bbox[-2]))
        
        # Ищем пересечение двух прямоугольников:
        intersection = crop_bbox & self
        
        # Возвращаем None, если пересечений нет:
        if intersection is None:
            return None
        
        # Если пересечение есть, то приводим к локальной системе координат и возвращаем:
        else:
            xmin, ymin, xmax, ymax = crop_bbox.flatten()
            return intersection.shift((-xmin, -ymin))
    
    # Возвращает площадь фигуры (величина может быть отрицательной):
    def area(self):
        
        # Пока работает только для прямоугольников:
        assert self.type == 'rectangle'
        
        # Получаем границы прямоугольника:
        xmin, ymin, xmax, ymax = self.flatten()
        
        return (xmax - xmin) * (ymax - ymin)
    
    # Неоднородное масштабирование:
    def rescale(self, k_height, k_width):
        return type(self)(self.points * np.array([k_width, k_height]),
                          imsize = None if self.imsize else (self.imsize[0] * k_height,
                                                             self.imsize[1] * k_width))
    
    # Интерполяция между контурами self.points и cvat_points.points с весом alpha для второго контура:
    def morph(self, cvat_points, alpha):
        
        # Принудительно делаем из веса итерируемый numpy-вектор:
        a = np.array(alpha).flatten()
        
        # Переходим к абстрактным контурам:
        p1 = self
        p2 = cvat_points
        
        # Число вершин в каждом контуре
        l1 = len(p1)
        l2 = len(p2)
        
        # Второй коэффициент взвешинвания
        b = 1. - a
        
        # Делаем так, чтобы первый контур всегда был не меньше второго по числу точек:
        if l2 > l1:
            p1, p2 = p2, p1
            l1, l2 = l2, l1
            a , b  = b , a
        
        dl = l1 - l2 # Разница числа вершин двух контуров
        # Если контуры действительно имеют разное число вершин:
        if dl:
            # Список длин каждого сегменда бОльшего контура:
            segments_len = p1.segments_len()
            
            # Список индексов сегментов, подлежащих схлопыванию в точку:
            short_inds = np.argsort(segments_len)[:dl]
            # Схлопываются самые короткие сегменты.
            
            # Список точек второго контура с повторением первой точки в конце:
            p2_circle_points = np.vstack([p2.points, p2.points[:1, :]])
            # Дублирование первой точки в конце нужно на случай, если схлопывать надо ...
            # ... последний отрезок контура, связанный с первой и последней точками.
            
            # Переопределяем список точек второго контура, ...
            # дублируя те точки, что должны разойтись при морфинге, ...
            # образуя кратчайшие сегменты другого контура:
            points = [] # Список точек
            delay = 0   # Задержка индекса (нужен для дублирования точек)
            for ind in range(l1): # Перебор по всем индексам бОльшего контура
                points.append(p2_circle_points[ind - delay, :]) # Вносим очередную точку малого контура в список
                if ind in short_inds: # Если текущую точку надо продублировать, то ,,,
                    delay += 1        # ... увеличиваем задержку индекса
            p2 = type(self)(np.vstack(points), imsize=self.imsize) # Собираем точки в контур
        # В соответствии с вышереализованным алгоритмом в контуре с бОльшим количеством точек ...
        # ... выбираются наикратчайщие отрезки, которые будут объеденины в точки при переходе ...
        # ... в более простой многоугольник. Т.о. геометрия более простого многоугольника не ...
        # ... влияет на то, какие отрезки будут вырождены в точки. При этом первые точки обоих ...
        # ... контуров обязаны переходить одна в другую. Остальные связываются в зависимости ...
        # ... от того, какие пары точек одного контура переходят в единственную точку другого.
        # ... Алгоритм не самый продвинутый, но кажется приемлемым в данном случае.
        
        # Производим линейную интерполяцию:
        points = []
        for a_, b_ in zip(a, b):
            points.append(p1 * b_ + p2 * a_)
        
        return points[0] if len(points) == 1 else points
    
    # Обрамляющий прямоугольник (левый верхний угол, правый нижний угол):
    def asrectangle(self, apply_rot=True):
        
        # Применяем поворот:
        if apply_rot and self.rotation % 360:
            self = self.apply_rot()
        
        # Получаем x- и y-координаты:
        x = self.x()
        y = self.y()
        
        # Вычисляем координаты рамки в зависимости от типа разметки:
        if self.type == 'ellipse':
            ax = abs(x[1] - x[0])
            ay = abs(y[1] - y[0])
            xmin = x[0] - ax
            ymin = y[0] - ay
            xmax = x[0] + ax
            ymax = y[0] + ay
        
        elif self.type == 'rectangle':
            xmin = x[0]
            ymin = y[0]
            xmax = x[1]
            ymax = y[1]
        
        elif self.type == 'polygon':
            xmin = x.min()
            ymin = y.min()
            xmax = x.max()
            ymax = y.max()
        
        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)
        
        return xmin, ymin, xmax, ymax
    # Параметр apply_rot пришлось ввести во избежание рекурсии при вызове метода apply_rot().
    
    # Обрамляющий прямоугольник (левый верхний угол, размеры):
    def asbbox(self):
        xmin, ymin, xmax, ymax = self.asrectangle()
        return min(xmax , xmin), min(ymax , ymin), abs(xmax - xmin), abs(ymax - ymin)
        
    # Обрамляющий прямоугольник в формате YOLO (центр, размер):
    def yolobbox(self, height=None, width=None):
        
        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:
            
            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')
            
            height, width = self.imsize
        
        # Интерпретируем параметры описанного прямоугольника как координаты крайних точек:
        xmin, ymin, xmax, ymax = self.asrectangle()
        
        cx = (xmin + xmax) / 2 / width  # Относительные ...
        cy = (ymin + ymax) / 2 / height # ... координаты центра
        w  = (xmax - xmin)     / width  # Относительные ...
        h  = (ymax - ymin)     / height # ... размеры
        
        return cx, cy, w, h
    
    # Представляет любой контур многоугольником:
    def aspolygon(self, apply_rot=True):
        
        if self.type == 'ellipse':
            # Параметры эллипса:
            cx, cy, rx, ry = self.flatten()
            ax = abs(rx - cx)
            ay = abs(ry - cy)
            
            # Точки эллипса:
            n = 30                                  # Число точек эллипса
            a = np.linspace(0, 2 * np.pi, n, False) # Углы в радианах от 0 до 2pi
            x = ax * np.cos(a) + cx
            y = ay * np.sin(a) + cy

            return type(self)(np.vstack([x, y]).T, 'polygon', self.rotation * apply_rot, imsize=self.imsize, rotate_immediately=apply_rot)
            
        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.asrectangle(apply_rot)
            x = np.array([xmin, xmax, xmax, xmin])
            y = np.array([ymin, ymin, ymax, ymax])

            return type(self)(np.vstack([x, y]).T, 'polygon', self.rotation * apply_rot, imsize=self.imsize, rotate_immediately=apply_rot)
        
        elif self.type == 'polygon':
            return type(self)(self.points, 'polygon', self.rotation * apply_rot, imsize=self.imsize, rotate_immediately=apply_rot)
            
        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)
    # Параметр apply_rot пришлось ввести во избежание рекурсии при вызове метода apply_rot().
    
    # Создаёт однострочный датафрейм, или добавляет новую строку к старому с даннымии о контуре:
    def to_dfraw(self, df=None, **kwargs):
        return add_row2df(type=self.type, points=self.flatten(), rotation=self.rotation, **kwargs)
    
    # Получить параметры для формирования cvat-разметки annotation.xml:
    def xmlparams(self):

        # Инициализация списка позиционных параметров:
        args = []
        # В настоящий момент должен содержать только тип метки.
        
        # Инициализация словаря именованных параметров:
        kwargs = {'rotation':str(self.rotation)}

        # Тип эллипса:
        if self.type == 'ellipse':
            args.append(self.type)
            
            # Параметры эллипса:
            kwargs['cx'], kwargs['cy'], kwargs['rx'], kwargs['ry'] = map(str, self.flatten())

        # Тип прямоугольника:
        elif self.type == 'rectangle':
            args.append('box')
            
            # Параметры прямоугольника:
            kwargs['xtl'], kwargs['ytl'], kwargs['xbr'], kwargs['ybr'] = map(str, self.asrectangle(False))
        
        # Тип многоугольника:
        elif self.type in {'polygon', 'polyline', 'points'}:
            args.append(self.type)
            
            # Параметры многоугольника:
            kwargs = {}
            kwargs['points'] = ';'.join(['%f,%f' % tuple(point) for point in self.points])
        
        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)
        
        return args, kwargs
    
    # Выполняет отражение по вертикали или горизонтали:
    def flip(self, axis={0,1}, height=None, width=None):
        
        # Проверка на корректность значения axis:
        if not isinstance(axis, (list, tuple, set)):
            assert axis in [0, 1]
            axis = set((axis,))
        assert axis <= {0, 1}
        
        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:
            
            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')
            
            height, width = self.imsize
        
        # Вычисляем новый угол поворота:
        rotation = self.rotation * (-1) ** len(axis)
        # Знак меняется на противоположный столько раз, сколько происходит отражений.
        
        # Вычисляем новые координаты в зависимости от типа разметки:
        
        if self.type == 'ellipse':
            cx, cy, rx, ry = self.flatten()
            
            if 0 in axis:
                cy, ry = height - cy, height - ry
            if 1 in axis:
                pass
                cx, rx = width  - cx, width  - rx
            
            return type(self)([cx, cy, rx, ry], self.type, rotation=rotation, imsize=(height, width), rotate_immediately=False)
        
        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.flatten()
            
            if 0 in axis:
                ymin, ymax = height - ymax, height - ymin
            if 1 in axis:
                xmin, xmax = width  - xmax, width  - xmin
            
            return type(self)([xmin, ymin, xmax, ymax], self.type, rotation=rotation, imsize=(height, width), rotate_immediately=False)
        
        elif self.type == 'polygon':
            points = self.points.copy()
            if 0 in axis:
                points[:, 1] = height - points[:, 1]
            if 1 in axis:
                points[:, 0] = width - points[:, 0]
            
            return type(self)(points, self.type, rotation=rotation, imsize=(height, width), rotate_immediately=False)
        
        else:
            raise ValueError('Неизвестный тип сегмента: %s' % self.type)
    
    # Выполняет отражение по горизонтали:
    def fliplr(self, height=None, width=None):
        return self.flip(1, height, width)
    
    # Выполняет отражение по вертикали:
    def flipud(self, height=None, width=None):
        return self.flip(0, height, width)
    
    # Поворот k-раз на 90 градусов по часовой стрелке:
    def rot90(self, k=1, height=None, width=None):
        
        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:
            
            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')
            
            height, width = self.imsize
        
        # Приводим к диапазону [0, 4) (т.е. поворот от 0 до 270 градусов):
        k = k - int(np.ceil((k + 1) / 4) - 1) * 4
        assert k in {0, 1, 2, 3}
        
        # Если 0 градусов, то просто повторяем контур:
        if k == 0:
            return type(self)(self.points, self.type, rotation=self.rotation, imsize=(height, width), rotate_immediately=False)
        
        # Если 180 градусов, то вместо поворота выполняем отражения по горизонтали и вертикали:
        elif k == 2:
            return self.flip(height=height, width=width)
        
        # Если 90 или 270 градусов:
        else: # (k in {1, 3})
            
            # Меняем местами ширину и высоту итогового изображнеия:
            imsize = (width, height)
            
            if self.type == 'ellipse':
                cx, cy, rx, ry = self.flatten()
                
                if k == 1:
                    cx, cy, rx, ry = cy, width - cx, ry, width - rx
                else:
                    cx, cy, rx, ry = height - cy, cx, height - ry, rx
                
                return type(self)([cx, cy, rx, ry], self.type, rotation=self.rotation, imsize=imsize, rotate_immediately=False)
            
            elif self.type == 'rectangle':
                xmin, ymin, xmax, ymax = self.flatten()
                
                if k == 1:
                    xmin, ymin, xmax, ymax = ymax, width - xmin, ymin, width - xmax
                else:
                    xmin, ymin, xmax, ymax = height - ymax, xmin, height - ymin, xmax
                
                return type(self)([xmin, ymin, xmax, ymax], self.type, rotation=self.rotation, imsize=imsize, rotate_immediately=False)
            
            elif self.type == 'polygon':
                points = self.points[:, ::-1].copy()
                
                if k == 1:
                    points[:, 1] = width  - points[:, 1]
                else:
                    points[:, 0] = height - points[:, 0]
                
                return type(self)(points, self.type, rotation=self.rotation, imsize=imsize, rotate_immediately=False)
    
    # Номера точек, где контур надо резать на два:
    def get_split_inds(self):

        # Если фигура - многоугольник:
        if self.type == 'polygon':
            
            # Формируем список индексов двух одинаковых точек, идущих в контуре подрят:
            split_inds = np.where((self.points == self.shift_list()).all(1))[0]
            # Ищем пересечение множеств индексов, от которых идёт совпадение значений со сдвигом на 1
            
            # Определяем, есть ли в контуре три одинаковые точки, идущие подрят:
            inds2del = set(split_inds) & set(np.where((self.points == self.shift_list(2)).all(1))[0])
            # Ищем пересечение множеств индексов, от которых идёт совпадение значений со сдвигом на 1 и 2

            # Если есть:
            if inds2del:
                
                # Если в контуре есть 4 и более одинаковых точек, идущих подрят, то у нас проблемы:
                if inds2del & set(np.where((self.points == self.shift_list(3)).all(1))[0]):
                    raise IndexError('В контуре совпадают более 3х точек подряд!\nЭтот контур уже нельзя разбить автоматически!')
                
                # Отбрасываем ненужные индексы:
                split_inds = np.array(sorted(list(set(split_inds) - set(inds2del))))
            
            return split_inds

        # Если фигура - не мгогоугольник, то возвращаем пустой список:
        else:
            return []
    
    # Расщепляет контур, если в нём на самом деле сохранены несколько контуров:
    def split_multipoly(self):
        
        # Расщепляем только если это многоугольник:
        if self.type != 'polygon':
            return [self.copy()]
        
        # Получаем точки расщепления:
        split_inds = self.get_split_inds()
        
        # Если точки вообще имеются, то:
        if len(split_inds):
            
            cvat_points_list = []
            for start, end in zip(np.roll(split_inds, 1) + 1, split_inds):
                lenght = end - start
                if lenght < 0:
                    lenght += len(self)
                
                sub_points = self.shift_list(-start)[:lenght, :]
                sub_points = type(self)(sub_points, rotation=self.rotation, imsize=self.imsize, rotate_immediately=False)
                cvat_points_list.append(sub_points)
            
            return cvat_points_list
        
        else:
            return [self.copy()]
    
    # Собирает один контур из нескольких:
    @classmethod
    def unite_multipoly(cls, poly_list, rotation=0., imsize=None):
        
        # Если контуров реально несколько, то действительно объединяем:
        if len(poly_list) > 1:
            
            # Создаём копию точек первого контура:
            points = cls(poly_list[0]).points

            # Берём последную точку этого контура:
            last_point = points[-1:, :]
            
            # Последнюю точку дублируем в самом контуре, чтобы отметить место начала следующего контура:
            points = np.vstack([points, last_point])

            # Наращиваем контур, дублируя в каждой из составляющих последние точки:
            for new_poly in poly_list[1:]:
                new_points = cls(new_poly).points
                points = np.vstack([points, new_points, new_points[-1:, :]])
            
            return cls(points, rotation=rotation, imsize=imsize)
        
        # Если контур всего один, то просто делаем его копию:
        elif len(poly_list) == 1:
            return cls(poly_list[0], rotation=rotation, imsize=imsize)
        
        # Если контуров вообще нет, то возвращаем None:
        else:
            return
    # Все контуры объединяются в один, а места сращивания отмечаются 
    # дублированием координат последней точки предыдущего контура, чтобы
    # потом можно было без проблем восстановить исходный набор конуров.
    
    # Отбрасывает метки мультиконтурности (убирает повторяющиеся точки):
    def fuse_multipoly(self):
        if self.type == 'polygon':
            points = list(self.points)  # Формируем список точек
            points.append(points[0])    # Добавляем в конец первую точку для проверки ...
                                        # ... на повторение первого элемента с последним.
            
            # Инициируем новый список:
            points_ = []
            
            # Заполняем его неповторяющимися элементами:
            for ind in range(len(self.points)):
                if (points[ind] != points[ind + 1]).any():
                    points_.append(points[ind])
            
            # Собираем из списка новый массив точек:
            return type(self)(np.array(points_), self.type, rotation=self.rotation, imsize=self.imsize, rotate_immediately=False)
        
        else:
            return self.copy()
    # Применяется если повторяющиеся точки появились случайно, например, ...
    # ... при разметке, и не являются признаком многоконтурности.
    
    # Векторизирует маску:
    @classmethod
    def from_mask(cls, mask, findContours_mode=cv2.CHAIN_APPROX_SIMPLE):

        # Бинарную маску надо перевести в полутоновую:
        if str(mask.dtype) == 'bool':
            mask = mask.astype(np.uint8)
        
        # Векторизируем:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, findContours_mode)
        
        # Убираем лишнее измерение в каждом контуре и оставляем лишь те контуры, что вообще содержат точки:
        contours = [contour.squeeze() for contour in contours if len(contour)]
        
        # Собираем контуры в один:
        points = cls.unite_multipoly(contours, imsize=tuple(mask.shape[:2]))
        
        return points
    
    # Создаёт прямоугольный контур из обрамляющего прямоугольника:
    @classmethod
    def from_bbox(cls, xmin, ymin, dx, dy, imsize=None):
        
        xmax = xmin + dx
        ymax = ymin + dy
        
        return cls([xmin, ymin, xmax, ymax], 'rectangle', imsize=imsize)
    
    # Создаёт прямоугольный контур из обрамляющего прямоугольника YOLO-формата:
    @classmethod
    def from_yolobbox(cls, cx, cy, w, h, imsize):
        
        imheight, imwidth = imsize
        
        cx = cx * imwidth
        cy = cy * imheight
        w  = w  * imwidth
        h  = h  * imheight
        
        xmin = cx - w / 2
        ymin = cy - h / 2
        
        xmax = xmin + w
        ymax = ymin + h
        
        return cls([xmin, ymin, xmax, ymax], 'rectangle', imsize=imsize)
    
    # Создаёт контур из строки в датафрейме подзадачи:
    @classmethod
    def from_dfraw(cls, raw, imsize=None, rotate_immediately=True):
        return cls(raw['points'], raw['type'], raw['rotation'], imsize=imsize, rotate_immediately=rotate_immediately)
    
    # Возвращает многоугольный контур с уменьшенным числом вершин:
    def reducepoly(self, epsilon=1.5):
        assert self.type == 'polygon'
        
        # Возвращаем список без изменений, если он пустой:
        if len(self.points) == 0:
            return self.copy()
        
        # Получаем Расщеплённые контуры:
        multipoly = self.split_multipoly()
        
        # Если контуров больше одного:
        if len(multipoly) > 1:
            
            # Обрабатываем каждый по тдельности:
            multipoly = [poly.reducepoly(epsilon).points for poly in multipoly]
            
            # Объединяем их обратно в один контур и возвращаем:
            return self.unite_multipoly(multipoly, rotation=self.rotation, imsize=self.imsize)
        
        # Если контур всего один, то обрабатываем его:
        reduced_poly = cv2.approxPolyDP(self.points, epsilon, True).squeeze()

        # Если в упрощённом контуре меньше 3-х точек, то возвращаем исходный контур:
        if reduced_poly.size < 6:
            return self.copy()

        # Возвращаем упрощённый контур:
        return type(self)(reduced_poly, rotation=self.rotation, imsize=self.imsize, rotate_immediately=False)
    
    # Многоугольник в формате YOLO:
    def yoloseg(self, height=None, width=None):
        
        # Доопределяем высоту и ширину изображения, если не заданы:
        if height is None and width is None:
            
            # Если размер изображения не был задан и изначально, выводим ошибку:
            if self.imsize is None:
                raise ValueError('Должен быть задан imsize либо (height, width)!')
            
            height, width = self.imsize
        
        # Конвертируем точки в многоугольник, если нужно:
        points = self if self.type == 'polygon' else self.aspolygon()
        
        # Интерпретируем параметры описанного прямоугольника как координаты крайних точек:
        x = points.x() / width  # Относительная абсцисса
        y = points.y() / height # Относительная ордината
        
        return np.vstack([x, y]).T.flatten()
    
    # Рисует многоугольник на бинарном изображении:
    def draw(self, img=None, caption=None, color=(255, 255, 255), thickness=1, show_start_point=False):
        
        # Поворачиваем контур, если объект не эллипс, а угол кратен 360 градусам:
        if self.type != 'ellipse' and self.rotation % 360:
            self = self.apply_rot()
        
        # Если изображение не задано, то:
        if img is None:
            
            # Если размер изображения не записан и в самом контуре, то используем размер самого многугольника:
            if self.imsize is None:
                shift = self.points.min(0)                              # Определяем координаты левого верхнего края обрамляющего прямоугольника
                points = self.points - shift                            # Прижимаем многоугольник к левому верхнему углу
                points = points.astype(int)                             # Округляем координаты вершин до целых
                img = np.zeros(points.max(0)[::-1] + 1, dtype=np.uint8) # Размер изображения = размеру многоугольника
            else:
                img = np.zeros(self.imsize, dtype=np.uint8)
                shift = np.zeros(2)
        
        # Если изображение задано, то контур берём как есть и округляем коодринаты до целых:
        else:
            img = img.copy()
            shift = np.zeros(2)
        
        # Отрисовываем многоугольник:
        if self.type == 'polygon':

            # Расщепляем составные контуры:
            multipoly = self.shift(-shift).split_multipoly()
            
            # Формируем списки списокв точек:
            pts = [p.points.astype(int) for p in multipoly]
            
            # Рисуем залитый или полый контур:
            if thickness == -1:
                img = cv2.fillPoly (img, pts,       color                           )
            else:
                img = cv2.polylines(img, pts, True, color=color, thickness=thickness)
            
            # Определение координат центра надписи:
            if caption:
                cx = int(self.x().mean())
                cy = int(self.y().mean())
            
            # Обводим кружком первую вершину контура, если надо:
            if show_start_point:
                for p in multipoly:
                    img = cv2.circle(img, p.points[0, :].astype(int), 20, color=color, thickness=thickness)
        
        # Отрисовываем прямоугольник:
        elif self.type == 'rectangle':
            xmin, ymin, xmax, ymax = self.flatten().astype(int)
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
            if caption:
                cx = int(self.x().mean())
                cy = int(self.y().mean())
        
        # Отрисовываем эллипс:
        elif self.type == 'ellipse':
            cx, cy, rx, ry = self.flatten().astype(int)
            ax = abs(rx - cx)
            ay = abs(ry - cy)
            img = cv2.ellipse(img, (cx, cy), (ax, ay), self.rotation, 0, 360, color=color, thickness=thickness)
        
        else:
            raise ValueError(f'Неизвестный тип контура "{self.type}"!')
        
        # Надпись в центре, если надо:
        if caption:
            img = cv2.putText(img, str(caption), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return img
    
    # Рисует многоугольник на изображении и выводит последнее на экран:
    def show(self, *args, **kwargs):
        #plt.figure(figsize=(5, 5))
        plt.imshow(self.draw(*args, **kwargs))
        plt.axis(False)
        #plt.show()


def interpolate_df(df, true_frames):
    '''
    Дополняет датафрейм интерполированными контурами.
    '''
    # Номер последнего кадра прореженной последовательности:
    last_frame = np.max(list(true_frames.keys()))
    
    # Список датафреймов с интерполированными кадрами:
    interp_frame_dfs = []
    
    # Перебор всех объектов:
    for track_id in df['track_id'].unique():
        
        # Пропускаем объекты без идентификатора, ...
        # ... т.к. обычно это означает, что объект  ...
        # ... появляется лишь в одном кадре:
        if track_id is None:
            continue
        
        # Берём ту часть датафрейма, в которой содержится инфомрация о текущем объекте:
        object_df = df[df['track_id'] == track_id]
        
        # Определяем последний кадр и его :
        last_key_frame =           object_df['frame'].max()              # Номер последнего ключевого кадра
        last_key       = object_df[object_df['frame'] == last_key_frame] # Датафрейм с последним ключём
        assert len(last_key) == 1                                        # Ключ должен быть только один
        #last_key_row = last_key.iloc[0]                                  # Для более лёгкой адресации к полям
        
        # Если последний кадр не закрывает анимацию, то копируем ключ на последний кадр в оба датафрейма:
        if last_key_frame < last_frame and not last_key['outside'].iloc[0]:
            row = last_key.copy()
            row[     'frame'] =             last_frame
            row['true_frame'] = true_frames[last_frame]
            df        = pd.concat([       df, row])
            object_df = pd.concat([object_df, row])
        
        # Инициируем первый кадр новой последовательности с объектом:
        start_frame = None
        
        # Перебор по всем кадрам, где отмечен объект:
        for frame in sorted(object_df['frame'].unique()):
            
            # Строка, описывающая текущий объект в текущем кадре:
            frame_row = object_df[object_df['frame'] == frame].iloc[0]
            
            # Если предыдущий ключевой кадр был, и между ним и текущим есть ещё хоть один кадр:
            if (start_frame is not None) and (frame > start_frame + 1):
                
                # Строка, описывающая текущий объект в предыдущем ключевом кадре:
                sart_frame = object_df[object_df['frame'] == start_frame]
                assert len(sart_frame) == 1
                sart_frame_row = sart_frame.iloc[0]
                
                # Контуры предыдущего и текущего ключевых кадров
                p1 = CVATPoints(sart_frame_row['points'], sart_frame_row['type'])
                p2 = CVATPoints(     frame_row['points'],      frame_row['type'])
                
                # Весовые коэффициенты для морфинга промежуточных кдаров:
                alphas = np.linspace(0, 1, frame - start_frame, endpoint=False)[1:]
                
                # Контуры для промежуточных кадров:
                points_list = p1.morph(p2, alphas)
                if isinstance(points_list, CVATPoints):
                    points_list = [points_list]
                
                # Добавляем промежуточные кадры:
                for interp_frame, points in enumerate(points_list, start_frame + 1):
                    row = sart_frame.copy()
                    row[    'points'] = [list(points.flatten())]
                    row[     'frame'] =             interp_frame
                    try:
                        row['true_frame'] = true_frames[interp_frame]
                    except:
                        print(interp_frame, true_frames)
                        raise
                    interp_frame_dfs.append(row)
            
            # Если на этом кадре объект скрывается, то сбрасываем start_frame, иначе обновляем:
            start_frame = None if frame_row['outside'] else frame
    
    # Добавляем интерполированные кадры к датафрейму:
    df = pd.concat([df] + interp_frame_dfs)
    # Операция вынесена за пределы вложенных циклов ради производительности.
    
    # Возвращаем отсортированный по кадрам проинтерполированный список:
    return df.sort_values('frame')


def interpolate_task(task):
    '''
    Дополняет все датафреймы текущей задачи интерполированными контурами.
    '''
    return [(interpolate_df(df, true_frames), file, true_frames) for df, file, true_frames in task]
    

def interpolate_tasks_df(tasks, desc=None):
    '''
    Интерполирует контуры всех датафреймов в датасете:
    '''
    # Параллельная обработка данных:
    return mpmap(interpolate_task, tasks, desc=desc)


def split_subtask_by_labels(subtask                           ,
                            debug_mode: 'Режим отладки' = True):
    '''
    Дробит подзадачу (df, file, true_frames) на более мелкие подзадачи,
    согласно меткам "scene-start" и "scene-finish" в датафрейме.
    Применять надо строго после интерполяции контуров.
    '''
    # Расщипляем задачу на составляющие:
    full_df, file_path, true_frames = subtask
    
    # Проверяем заполненность датафрейма:
    if full_df is None:
        debug_str = 'Не заполнен "%s"\n' % file_path
        
        if debug_mode:
            print(debug_str)
            
            # Возвращаем задачу без изменений, но обёрнутую в список:
            return [subtask]
        
        else:
            raise ValueError(debug_str)
    
    # Формируем датафрейм, по которому будем проводить расщепление:
    sf_df = full_df[full_df['outside'] == False]       # Убераем все скрытые сегменты
    sf_df = sf_df[(sf_df['label'] == 'scene-start' ) | \
                  (sf_df['label'] == 'scene-finish')]    # Оставляем только метки начала/конца сцены
    # Этот датафрейм содержит только действующие метки начала/конца сцены.
    
    # Если меток начала/конца сцены нет, то возвращаем подзадачу, обёрнув её в список:
    if len(sf_df) == 0:
        return [subtask]
    
    # Формируем датафрейм, подлежащий расщеплению:
    df = full_df[(full_df['label'] != 'scene-start' ) & \
                 (full_df['label'] != 'scene-finish')]
    # Из него выброшены все строки, связанные с метками начала и конца сцены.
    
    # Инициируем нужные переменные перед циклом:
    debug_str   = file_path + ':\n' # Строка с отладочными данными
    start_frame = None              # Номер первого кадра последовательности
    subtasks    = []                # Список разбитых подзадач
    
    # Проход по всем номерам прореженных кадров, где есть метки начала/конца сцены:
    for frame in sorted(sf_df['frame'].unique()):
        
        # Метки начала/конца сцены, касающиеся только текущего кадра:
        frame_data = sf_df[sf_df['frame'] == frame]
        
        # Номер текущего кадра в непрореженной последовательности:
        true_frame = frame_data['true_frame'].values[0]
        
        # Если в кадре есть метка начала последовательности:
        if (frame_data['label'] == 'scene-start').any():
            debug_str += '(%d ' % frame
            
            # Если при этом последовательность уже открыта, то выводим ошибку:
            if start_frame is not None:
                debug_str += ' <- Два начала последовательности в метках подряд'
                debug_str += ' в кадре №%d(%d)!\n' % (frame, true_frame)
                if debug_mode:
                    print(debug_str)
                    break
                else:
                    raise ValueError(debug_str)
            
            # Если при этом последовательность ещё не открыта, то открываем:
            else:
                start_frame = frame
        
        # Если в кадре есть метка конца последовательности:
        if (frame_data['label'] == 'scene-finish').any():
            debug_str += ')%d ' % frame
            
            # Если при этом последовательность пока открыта:
            if start_frame is not None:
                
                # Формируем соответствующий этому интервалу датафрейм в список:
                sub_df = df[(df['frame'] >= start_frame ) & \
                            (df['frame'] <=       frame )]
                
                # Формируем соответствующий этому интервалу словарь номеров кадров:
                sub_true_frames = {key:val for key,val in true_frames.items() if (key >= start_frame) and (key <= frame)}
                
                # Добавляем новую подзадачу в итоговый список:
                subtasks.append((sub_df, file_path, sub_true_frames))
                
                start_frame = None # Закрываем последовательность
            
            # Если при этом последовательность даже не открывалась, то выводим ошибку:
            else:
                debug_str += ' <- Конец последовательности в метках без начала'
                debug_str += ' в кадре №%d(%d)!\n' % (frame, true_frame)
                if debug_mode:
                    print(debug_str)
                    break
                else:
                    raise ValueError(debug_str)
    
    # После конца цикла проверяем, закрыта ли последняя последовательность:
    else:
        if start_frame is not None:
            debug_str += ' <- Ненайден конец последовательности в метках!\n'
            if debug_mode:
                print(debug_str)
            else:
                raise ValueError(debug_str)
    
    return subtasks


def split_task_by_labels(task, debug_mode: 'Режим отладки' = True):
    '''
    Дробит все подзадачи текущей задачи, согласно меткам
    "scene-start" и "scene-finish" в датафрейме.
    Применять надо строго после интерполяции контуров.
    '''
    splitted_task = []
    for subtask in task:
        splitted_task += split_subtask_by_labels(subtask, debug_mode)
    
    return splitted_task


def split_true_frames_in_tasks(tasks                                                                                    ,
                               debug_mode: 'Режим отладки выводит отладочную информацию, не прерывая выполнение' = True ,
                               desc      : 'Название статусбара'                                                 = None ):
    '''
    Распиливает df на части, согласно меткам 'scene-start', 'scene-finish'.
    '''
    # Параллельная обработка данных:
    debug_modes = [debug_mode] * len(tasks)
    tasks = mpmap(split_task_by_labels, tasks, debug_modes, desc=desc)
    
    # Выбрасываем пустые задачи:
    tasks = [task for task in tasks if len(task)]
    
    return tasks


def fuse_multipoly_in_df(points):  
    '''
    Избавляет контур от повторяющихся точек.
    '''
    return CVATPoints(points).fuse_multipoly().flatten()


def post_process_task(task           : 'Задача'                                             ,
                      fuse_multipoly : 'Флаг очистки контуров от повторяющихся точек' = True):
    '''
    Постобработка задачи.
    Используется для избавления контуров от повторяющихся точек если многоконтурность исключена.
    '''
    # Инициируем новый список подзадач:
    task_ = []
    
    # Перебираем все подзадачи:
    for df, file, true_frames in task:
        
        # Дублируем текущий датафрейм чтобы не редактировать оригинал:
        df = df.copy()
        
        # Формируем маску для выделения только тех объектов, чьи точки реально являются контурами:
        mask = df['type']=='polygon'
        
        # Применяем преобразование для всех контуров датафрейма
        df.loc[mask, 'points'] = df[mask]['points'].apply(fuse_multipoly_in_df)
        
        # Вносим обновлённую подзадачу в конечный список:
        task_.append([df, file, true_frames])
    
    return task_


def post_process_tasks(tasks          : 'Задача'                                             ,
                       fuse_multipoly : 'Флаг очистки контуров от повторяющихся точек' = True,
                       desc           : 'Название статусбара'                          = None):
    '''
    Постобработка всех задач.
    Используется для избавления контуров от повторяющихся точек если многоконтурность исключена.
    '''
    return mpmap(post_process_task, tasks, [fuse_multipoly] * len(tasks), desc=desc)


def cvat_backups2tasks(unzipped_cvat_backups_dir):
    '''
    Формирует список из распарсенных и обработанных задач из папки с распакованными версиями CVAT-бекапов.
    Постаброботка включает в себя интерполяцию контуров и разбиение на сцены по меткам.
    '''
    # Парсим распакованные датасеты и формируем список данных для каждого видео:
    raw_tasks = cvat_backups2raw_tasks(unzipped_cvat_backups_dir, desc='Парсинг распакованных CVAT-датасетов')
    
    # Удаляем повторы в разметке:
    print('') # Отступ
    raw_tasks = drop_label_duplicates_in_tasks(raw_tasks, desc='Удаление повторов в разметке')
    
    # Интерполируем сегменты во всех неключевых кадрах:
    print('') # Отступ
    interp_tasks = interpolate_tasks_df(raw_tasks, desc='Интерполяция контуров в неключевых кадрах')
    
    # Разрезаем последовательности кадров, опираясь на метки 'scene-start', 'scene-finish'.
    # Этот шаг можно пропустить если датасет не содержит меток 'scene-start' и 'scene-finish':
    print('') # Отступ
    splited_tasks = split_true_frames_in_tasks(interp_tasks, desc='Расщепление смонтированных последовательностей')
    
    # Постобработка задач:
    print('') # Отступ
    tasks = post_process_tasks(splited_tasks, desc='Очистка контуров от повторяющихся точек')

    # Сортировка задач для воспроизводимости результата:
    tasks = sort_tasks(tasks)
    
    return tasks


def subtask2xml(subtask, xml_file='./annotation.xml'):
    '''
    Сохраняет cvat-подзадачу в XML-файл с аннотациями, принимаемыми интерфейсом CVAT в качестве разметки.
    '''
    # Расщепление подзадачи на составляющие:
    df, file, true_frames = subtask
    
    # Датафрейм разметки на:
    df_tracks = df[df['track_id'].notna()] # Треки
    df_shapes = df[df['track_id']. isna()] # Формы
    
    # Инициализируем XML-структуру:
    annotations = ET.Element('annotations')
    
    # Сохраняем треки (tracks):
    for track_id in df_tracks['track_id'].unique():
        
        # Весь датафрейм для текущего трека:
        df_track = df_tracks[df_tracks['track_id'] == track_id]
        
        # Получаем метку объекта:
        label = df_track['label'].unique() # Список всех использованных меток текущего объекта
        assert len(label) == 1             # Все метки должны быть одинаковыми
        label = label[0]                   # Берём эту единственную метку
        
        # Инициируем аннотацию изображения:
        track = ET.SubElement(annotations, 'track', label=label)
        
        # Перебор всек вхождений обхекта текущего трека в видеопоследотвательность:
        for dfraw in df_track.iloc:
            
            # Получаем XML-параметры трека в текущем кадре:
            args, kwargs = CVATPoints.from_dfraw(dfraw).xmlparams()
            
            # Вносим описание трека в текущем кадре в XML-структуру:
            ET.SubElement(track                            ,
                          frame    = str(dfraw['frame'   ]),
                          outside  = str(dfraw['outside' ]),
                          occluded = str(dfraw['occluded']),
                          keyframe = "1"                   ,
                          z_order  = str(dfraw['z_order' ]),
                          *args, **kwargs)
    
    # Сохраняем формы (shapes):
    for frame, true_frame in true_frames.items():
        
        # Определяем имя кадра:
        name = os.path.basename(file if isinstance(file, str) else file[frame])
        
        # Инициируем аннотацию изображения:
        image = ET.SubElement(annotations, 'image', id=str(frame), name=name)
        
        # Проходим по всем строкам датафрейма:
        for dfraw in df_shapes[df_shapes['frame'] == frame].iloc:
            
            # Получаем XML-параметры контуров:
            args, kwargs = CVATPoints.from_dfraw(dfraw).xmlparams()
            
            # Вносим описание контура в XML-структуру:
            ET.SubElement(image                            ,
                          label    =     dfraw['label'   ] ,
                          occluded = str(dfraw['occluded']),
                          source   =     dfraw['source'  ] ,
                          z_order  = str(dfraw['z_order' ]),
                          *args, **kwargs)
    
    # Пишем XML-структуру в файл:
    ET.ElementTree(annotations).write(xml_file, encoding="utf-8", xml_declaration=True)


def task_auto_annottation(task, img2df, label=None, store_prev_annotation = True):
    '''
    Применяет img2df для автоматической разметки бекапа
    cvat-задачи и сохраняет результат в файл annotation_file.
    '''
    # Инициализируем конечный список подзадач:
    task_ = []
    
    # Инициализируем буфер для чтения изображений:
    img_buffer = ImReadBuffer()
    
    # Перебор подзадач:
    for df, file, true_frames in task:
        
        # Инициализируем список датафреймов для последующего объединения:
        frame_dfs = []
        
        # Перебор кадров:
        for frame, true_frame in true_frames.items():
            
            # Читаем очередной кадр и переводим его в RGB:
            img = img_buffer(file, true_frame)[..., ::-1]

            # Получаем датафрейм, содержащий результат авторазметки:
            frame_df = img2df(img)

            # Коррекция значений в столбцах ...
            frame_df[     'label'] =      label # ... метки класса ...
            frame_df[     'frame'] =      frame # ... номера кадра прореженной последовательности ...
            frame_df['true_frame'] = true_frame # ... номера кадра полной      последовательности.

            # Добавление очередного датафрейма в общий список:
            frame_dfs.append(frame_df)

        # Формируем объединённый датафремй, содержащий или исключающий исходную разметку: 
        df = pd.concat([df] + frame_dfs if store_prev_annotation else frame_dfs)

        # Внесение очередной подзадачи в итоговую задачу:
        task_.append((df, file, true_frames))
    
    return task_


def cvat_backup_task_dir2auto_annotation_xml(cvat_backup_task_dir, img2df, label=None, store_prev_annotation=True, xml_file=None):
    '''
    Выполняет автоматическую разметку задачи по её бекапу и сохраняет в cvat-совместимый xml-файл.
    '''
    # Размещаем разметку в папке с задачей, если файл для сохранения явно не указан:
    if xml_file is None:
        xml_file = os.path.join(cvat_backup_task_dir, 'annotation.xml')
    
    # Читаем имеющуюся разметку:
    task = cvat_backup_task_dir2task(cvat_backup_task_dir)
    
    # Выполняем доразметку:
    task_ = task_auto_annottation(task, img2df, label, store_prev_annotation = True)
    
    # AФормируем один или несколько xml-файлов для каждой подзадачи:
    
    # Если подзадача всего одна, то сохраняем её в файле с заданным именем:
    if len(task_) == 1:
        subtask2xml(task_[0], xml_file)
    
    # Если подзадач несколько:
    else:
        
        # Перебор подзадач:
        for job_ind, subtask in enumerate(task, 1):
            
            # Разбиваем путь до xml-файла на имя и разширение:
            xml_file_name, xml_file_ext = os.path.splitext(xml_file)
            
            # Сохраняем разметку в файл с суффиксом "_{номер подзадачи}":
            subtask2xml(task_[0], xml_file_name + f'_{job_ind}')
    
    return task_


def cvat_backup_dir2auto_annotation_xmls(cvat_backup_dir, img2df, label=None, store_prev_annotation=True, xml_dir=None):
    '''
    Выполняет автоматическую разметку распакованного бекапа cvat-датасета и сохраняет в cvat-совместимые xml-файлы.
    '''
    tasks = []

    # Перебираем все вложенные в папку с датасетом директории:
    for cvat_backup_task_dir in os.listdir(cvat_backup_dir):
        print(cvat_backup_task_dir)
        
        # Уточняем полный путь до поддиректории:
        cvat_backup_task_dir = os.path.join(cvat_backup_dir, cvat_backup_task_dir)
        
        # Если это не папка, то пропускаем:
        if not os.path.isdir(cvat_backup_task_dir):
            continue
        
        # Определяем полный путь до xml-файла:
        xml_file = cvat_backup_task_dir + '.xml' if xml_dir is None else os.path.join(xml_dir, os.path.basename(cvat_backup_task_dir) + '.xml')
        # Если путь до папки с итоговой разметкой задан, то каждый xml-файл размещается в ней под именем папки с соответствующей задачей.
        # Если путь не задан, то каждый xml-файл размещается в папке с соответствующей задачей.
        
        tasks.append(cvat_backup_task_dir2auto_annotation_xml(cvat_backup_task_dir, img2df, label, store_prev_annotation, xml_file))
    
    return tasks


def subtask_shapes2tracks(subtask, minIoU=0.8):
    '''
    Трекинг несвязных форм в подзадаче.
    Полезен при доразметке видео после прогона через автоматическую
    разметку, которая работает покадрово (генерирует только формы).
    Для лучшего результата следует корректно разметить первый кадр.
    Остальные кадры будут размечены по аналогии.
    '''
    # Расщепление подзадачи на составляющие:
    df, file, true_frames = subtask
    
    # Устанавливаем ещё неиспользованный track_id:
    track_id = df['track_id'].max()
    if np.isnan(track_id): track_id = 0 # Устанавливаем в 0, если они не использовались вообще
    
    # Номера столбцов датафрейма:
    track_id_ind = np.where(df.columns == 'track_id')[0][0] # track_id
    label_ind    = np.where(df.columns == 'label'   )[0][0] # label
    
    # Определяем размер видео по первому кадру:
    with ImReadBuffer() as buffer:
        imsize = buffer(file, 0).shape[:2]
    
    # Создание чистого изображения для инициализации отрисовки всех масок:
    clear_frame = np.zeros(imsize, np.uint8)
    
    # Инициируем переменные предыдущего кадра пустыми списками:
    prev_df, prev_points, prev_masks = [[]] * 3
    dfs = []
    
    # Перебор кадров:
    for frame, true_frame in tqdm(true_frames.items()):
        
        # Датафрейм объекдов для текущего кадра:
        cur_df = df[df['frame'] == frame]
        #cur_df.index = range(len(cur_df))
        
        # Список объектовкачестве экземпляров класса текущего кадра в  CVATPoints:
        cur_points = [CVATPoints.from_dfraw(cur_df.iloc[i, :], imsize) for i in range(len(cur_df))]
        
        # Список объектов текущего кадра в качестве экземпляров класса Mask:
        cur_masks = [Mask(p.draw(clear_frame, color=255, thickness=-1).astype(bool), p.asrectangle()) for p in cur_points]
        
        # Выполняем обработку если на текущем и предыдущем кадрах есть объекты:
        if len(prev_df) and len(cur_df):
            
            # Построение матрицы связностей объектов:
            IoUmat = np.zeros((len(cur_df), len(prev_df))) # Инициация матрицы связностей объектов
            
            # Перебираем все объекты текущего кадра:
            for cur_ind in range(len(cur_df)):
                
                # Пропускаем этот объект, если он уже track, a не shape:
                if cur_df.iloc[cur_ind, track_id_ind] is not None:
                    continue
                
                # Перебираем все объекты предыдущего кадра:
                for prev_ind in range(len(prev_df)):
                    
                    # Только на первом кадре допускается искать shape, а не track:
                    if frame > 1 and prev_df.iloc[prev_ind, track_id_ind] is None:
                        continue
                    
                    # Считаем IoU для текущей пары объектов и вносим результат в матрицу связностей:
                    IoUmat[cur_ind, prev_ind] = cur_masks[cur_ind].IoU(prev_masks[prev_ind])
            
            # Выявляем связи объектов до тех пор, пока хоть одна связь в матрице превышает заданный порог:
            while IoUmat.max() > minIoU:
                
                # Устанавливаем связь между объектами (берём пару с максимальным IoU):
                cur_ind, prev_ind = np.unravel_index(IoUmat.argmax(), IoUmat.shape)
                
                # Если track_id предыдущего объекта не задан:
                if prev_df.iloc[prev_ind, track_id_ind] is None:
                    
                    # Ставим в его качестве текущий неиспользованный track_id для обоих объектов:
                    prev_df.iloc[prev_ind, track_id_ind] = cur_df.iloc[cur_ind, track_id_ind] = track_id
                    
                    # Обновляем неиспользованный track_id путём прирощения:
                    track_id += 1
                
                # Если track_id предыдущего объекта задан, то ставим его и на текущий объект:
                else:
                    cur_df.iloc[cur_ind, track_id_ind] = prev_df.iloc[prev_ind, track_id_ind]
                
                # Ставим текущему объекту метку от предыдущего: 
                cur_df.iloc[cur_ind, label_ind] = prev_df.iloc[prev_ind, label_ind]
                
                # Исключаем для этих объектов возможность образовывать иные пары:
                IoUmat[cur_ind, :] = IoUmat[:, prev_ind] = 0
        
        # Делаем все параметры текущего кадра параметрами предыдущего кадра:
        prev_df, prev_points, prev_masks = cur_df, cur_points, cur_masks
        # Готовимся к следующей итерации.
        
        # Вносим обработанный датафрейм текущего кадра в список:
        dfs.append(cur_df)
    
    # Объединяем данные для каждого кадра в единий датафрейм:
    df = pd.concat(dfs)
    
    # Возвращаем подзадачу с обновлённым датафреймом:
    return df, file, true_frames


def flat_tasks(tasks):
    '''
    Разбивает задачи для более эффективного распараллеливания при сохранении.
    Разделение выборки на train/val/test должно произойти до!
    '''
    # Словарь файл -> Задача:
    files = {}
    
    # Перебор задач:
    for task in tasks:
        
        # Перебор подзадач:
        for subtask in task:
            
            # Получаем имя файла для текущей подзадачи:
            file = subtask[1]
            
            # Если файл уже встречался, то добавляем эту ...
            # ... задачу в соответствующий список в словаре:
            if file in files:
                files[file] += [subtask]
            
            # Если файл ещё не встречался, то добавляем ...
            # ... эту задачу в новый список в словаре:
            else:
                files[file]  = [subtask]
    
    # Собираем и возвращаем новый список задач:
    return list(files.values())


def drop_bbox_labled_cvat_tasks(tasks):
    '''
    Оставляет только те задачи, что размечены не обрамляющими прямоугольниками.
    Тип датасета определяется по имени папки, в которой он хранится.
    Датасет, размеченный прямоугольниками содержит в своём имени следующую строку:
    'project_video_annotations_bounding boxes_backup'
    '''
    # Перебираем все задачи:
    tasks_ = []
    for task in tasks:
        
        # Переносим лишь те подзадачи, что не имеют "project_video_annotations_bounding boxes_backup" в пути к файлу:
        task_ = [subtask for subtask in task if 'project_video_annotations_bounding boxes_backup' not in subtask[1]]
        
        # Пустые задачи не вносим:
        if len(task_):
            tasks_.append(task_)
    
    return tasks_


def crop_df_labels(df, bbox, area_part_th):
    '''
    Вовзращает датафрейм с метками, сохранившимися после вырезания фрагмента из исходного изображения.
    '''
    # Результирующий датафрейм:
    cropped_df = df.copy()
    
    # Функция, возвращающая пересечение текущего сегмента с заданной рамкой:
    def crop_func(row):
        points0 = CVATPoints(row['points'], type_=row['type'], rotation=row['rotation']).asrectangle()
        points1 = points0.crop(bbox)
        
        # Возвращаем пересечение текущего сегмента только если ...
        # ... отношение площади новой фигуры к площади старой ...
        # ... не меньше заданного порога:
        
        if points1 is None:
            return
        
        else:
            s0 = points0.area()
            s1 = points1.area()
            '''
            if not isinstance(s0, (float, int)):
                print(type(s0))
                raise
            if not isinstance(s1, (float, int)):
                print(type(s1))
                raise
            '''
            if s0 == 0 or s1 / s0 < area_part_th:
                return
            
            else:
                return points1.flatten()
    
    # Применение crop_func ко всем меткам датафрейма, если последний не пуст:
    if len(cropped_df):
        cropped_df['points'] = cropped_df[['points', 'type', 'rotation']].apply(crop_func, axis=1)
        cropped_df['type'] = 'rectangle'
    # При этом все метки конвертируются в обрамляющие прямоугольники.
    
    # Возвращаем строки, прямоугольники которых имели пересечение с рамкой:
    return cropped_df[cropped_df['points'].notna()]


def split_image_and_labels2tiles(df           : 'Исходный датафрейм с разметкой'                                  ,
                                 image        : 'Исходное изображение'                                            ,
                                 max_im_size  : 'Максимальный размер целевых изобаржений'           = (1080, 1920),
                                 area_part_th : 'Доля площади, при которой сегмент ещё сохраняется' = 0.5         ):
    '''
    Разрезает кадр и метки на части заданного рзмера (полученные части могут пересекаться).
    Полезно при приведении к заданному размеру слишком больших изображений. Даже если
    исходное изображение меньше целевого, перед возвращением исходных данных все метки
    вписываются в рамки исходного изображения (исправляется выход за границы в исходной разметке).
    '''
    # Получаем размеры исходного и целевого изображений в виде numpy-векторов для удобства работы:
    im_size     = np.array(image.shape[:2])
    max_im_size = np.array(max_im_size)
    
    # Возвращаем данные, если исходное изображение меньше целевого:
    if (max_im_size >= im_size).all():
        im_rect = [0, 0, im_size[1], im_size[0]]
        return [(crop_df_labels(df, im_rect, area_part_th), image)]
    # При этом обрезаем края всех меток, что вышли за границы исходного изображения.
    
    # Определяем число сдвигов рамки вырезания по каждой оси:
    nij = np.ceil(im_size / max_im_size).astype(int)
    
    # Определяем шаг сдвига рамки вырезания по каждой оси:
    step = [0 if nij[ind] == 1 else (im_size[ind] - max_im_size[ind]) / (nij[ind] - 1) for ind in range(2)]
    # Заменяем нулями все inf-ы, оставшиеся после деления на 0.
    
    #step = (im_size - max_im_size) / (nij - 1)
    #np.nan_to_num(step, False, 0, 0, 0)s
    
    # Целевой список:
    tiles = []
    
    # Перебираем все сдвиги по веркикали:
    for i in range(nij[0]):
        
        # Определяем вертикальные границы рамки:
        y_min = int(i * step[0])
        y_max = min(y_min + max_im_size[0], im_size[0])
        
        # Перебираем все сдвиги по горизонтали:
        for j in range(nij[1]):
            
            # Определяем горизонтальные границы рамки:
            x_min = int(j * step[1])
            x_max = min(x_min + max_im_size[1], im_size[1])
            
            # Вырезаем фрагмент из изображения и разметки:
            cropped_image = image[y_min:y_max, x_min:x_max, :]
            cropped_df    = crop_df_labels(df, [x_min, y_min, x_max, y_max], area_part_th)
            
            # Внесение новой пары разметка-изображение в целевой список:
            tiles.append((cropped_df, cropped_image))
    
    # Возвращаем целевой список:
    return tiles


def fill_na_in_track_id(df):
    '''
    Заполняет незаполненные идентификаторы объектов уникальными номерами.
    '''
    # Копируем датафрейм, чтобы не перезаписывать исходный:
    df = df.copy()
    
    # Получаем маску объектов с незаполненным полем 'track_id':
    None_in_track_id_mask = df['track_id'] != df['track_id']
    
    # Если пропуски вообще есть:
    if None_in_track_id_mask.any():
        
        # Если не задан ни один 'track_id', то заполняем все подряд:
        if None_in_track_id_mask.all():
            df['track_id'] = range(len(df))
        
        # Если есть как заданные, так и пропущенные значения поля 'track_id':
        else:
            # Получаем номер, с которого можно продолжить нумерацию 'track_id':
            start_track_id = df[~None_in_track_id_mask]['track_id'].max()
            
            # Продолжаем нумерацию для незаполненных объектов:
            df.loc[None_in_track_id_mask, 'track_id'] = range(start_track_id, start_track_id + None_in_track_id_mask.sum())
    
    return df


def fill_na_in_track_id_in_all_tasks(tasks):
    '''
    Заполняет незаполненные идентификаторы объектов уникальными номерами.
    '''
    # Инициализация конечного списка задач:
    tasks_ = []
    
    # Перебор всех задач:
    for task in tasks:
        
        # Меняем df в каждой подзадаче:
        task = [(fill_na_in_track_id(df), file, true_frames) for df, file, true_frames in task]
        
        # Обновлённую задачу вносим в конечный список:
        tasks_.append(task)
    
    return tasks_


def sort_tasks(tasks):
    '''
    Сортирует все задачи по имени файла первой подзадачи.
    Используется перед конвертацией для унификации результатов конвертации.
    '''
    # Список имён первых файлов для первой подзадачи каждой из задачь:
    files = []
    for task in tasks:
        
        # Имя файла первой подзадачи текущей задачи:
        file = task[0][1]
        
        # Если файлов несколько, берём первый:
        if isinstance(file, (list, tuple)):
            file = file[0]
        
        # Вносим в список для сортировки
        files.append(file)
    
    # Возвращаем отсортированный список задач:
    return [tasks[ind] for ind in np.argsort(files)]


"""
# Код тестирования поворотов и отражений в CVATPoints:
for rotation in [0, 15]:
    for points in [CVATPoints([100, 200, 400, 350], 'rectangle', imsize=(500, 600), rotation=rotation, rotate_immediately=False),
                   CVATPoints([100, 200, 400, 350], 'ellipse'  , imsize=(500, 600), rotation=rotation, rotate_immediately=False),
                   CVATPoints([100, 200, 400, 350 ,
                               300, 150, 250, 120],              imsize=(500, 600), rotation=rotation, rotate_immediately=False)]:
        
        img = points.draw(show_start_point=True)
        
        gt_flip = np.hstack([              img ,
                                 np.fliplr(img),
                                 np.flipud(img),
                                 np.flip  (img, (0, 1))])
        
        my_flip = np.hstack([points.flip().flip().draw(show_start_point=True),
                             points.fliplr()     .draw(show_start_point=True),
                             points.flipud()     .draw(show_start_point=True),
                             points.flip  ()     .draw(show_start_point=True)])
        
        flip = np.dstack([gt_flip, my_flip, my_flip])
        
        gt_rot = np.hstack([img] * 4)
        
        my_rot = np.hstack([np.rot90(points.rot90(k).draw(show_start_point=True), -k) for k in range(4)])
        
        rot = np.dstack([gt_rot, my_rot, my_rot])
        
        plt.figure(figsize=(24, 24))
        plt.imshow(np.vstack([flip, rot]));
        plt.axis(False)
"""


__all__ = CVATPoints, cvat_backups2tasks, drop_bbox_labled_cvat_tasks, sort_tasks