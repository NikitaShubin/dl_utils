import os
import numpy as np
from tempfile import TemporaryDirectory
from hashlib import sha256

import pandas as pd
import cv2
import torch
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
import warnings
from contextlib import contextmanager

from pt_utils import AutoDevice
from video_utils import VideoGenerator
from cvat import (df_save, df_load, new_df, add_row2df, CVATPoints,
                  concat_dfs, ergonomic_draw_df_frame, subtask2preview,
                  hide_skipped_objects_in_df, subtask2xml, df_list2tuple,
                  df_tuple2list)
from utils import (unflatten_list, mkdirs,
                   color_float_hsv_to_uint8_rgb)
from samal import fit_segments_in_df


class Prompts:
    '''
    Упрощает хранение и изменение подсказок.

    Данные хранятся в CVAT-датафрейме. Однако смысл некоторых полей изменён:
        'track_id'  : 'obj_id' - номер отслеживаемого объекта;
        'type'      : Маска хранится в типе 'polygon', точки в 'points'
                      (включающие и исключающие в разных записях, отличающихся
                      значением в столбце outside), а прямоугольники в
                      'polyline', т.к. даже bbox в IPYInteractiveSegmentation
                      задаётся точками;
        'attributes': 'start_frame' - номер опорного кадра, от которого идёт
                      отслеживание объекта в обе стороны по оси времени (един
                      для всех записей с одинаковым 'track_id').
    '''

    def __init__(self, seq_len, imsize, file='./prompts.tsv', auto_save=True):

        # Длина видеопоследовательности:
        self.seq_len = seq_len

        # Размер изображения:
        self.imsize = imsize
        # Нужен для растеризации.

        # Файл для хранения подсказок.
        self.file = file

        # Флаг автосохранения:
        self.auto_save = auto_save and file

        # Создаём новый или читаем имеющийся датафрейм:
        self.df = df_load(file) if file and os.path.isfile(file) else new_df()

    def make_auto_save(self):
        if self.auto_save:
            df = df_tuple2list(self.df)
            df_save(df, self.file)

    # Просто возвращает список всех obj_id:
    def get_all_obj_ids(self, df=None):
        if df is None:
            df = self.df
        return sorted(list(df['track_id'].unique()))

    # Просто возвращает список всех кадров, где есть подсказки:
    def get_all_key_frames(self, df=None):
        if df is None:
            df = self.df
        return sorted(list(df['frame'].unique()))

    # Строит маску, выделяющую строки датафрейма по значению:
    def select_rows(self, column, values=None, df=None):

        # Используем исходный датафрейм, если он не передан:
        if df is None:
            df = self.df

        # Если значение не указано, значит, берём все строки:
        if values is None:
            return pd.Series([True] * len(df))

        # Если дан список значений, выбираем все совпадающие:
        elif isinstance(values, (list, tuple)):
            return df[column].isin(values)

        # Если передан не None, не список и не кортеж - берём его, как одно
        # значение:
        else:
            return df[column] == values

    # Очищает разметку выделенного кадра, или на всех кадрах;
    # выделенного объекта или всех объектов:
    def clear(self, frames=None, obj_ids=None):

        # Если не указан ни один из параметров - очищаем весь датафрейм:
        if frames is None and obj_ids is None:
            self.df = new_df()
            self.make_auto_save()
            return self.df

        # Выделяем строки, содержащие запретные кадры:
        frame_selection_mask = self.select_rows('frame', frames)

        # Выделяем строки, содержащие запретные объекты:
        obj_id_selection_mask = self.select_rows('track_id', obj_ids)

        # Исключаем все сочетания запретных объектов с запретными кадрами:
        selection = (obj_id_selection_mask & frame_selection_mask) == False
        self.df = self.df[selection]
        self.make_auto_save()

        return self.df

    # Возвращает ещё не использованный id (для нового объекта):
    def get_new_obj_id(self):
        used_ids = self.df['track_id'].unique()
        new_id = 0
        while new_id in used_ids:
            new_id += 1
        return new_id

    # Возвращает одно значение, для всех записей заданного трека:
    def get_single_value_by_obj(self, obj_id, column, df=None):

        # Используем исходный датафрейм, если он не передан:
        if df is None:
            df = self.df

        # Переводим все списки в кортежи для корректной работы метода unique
        # в следующей строке:
        # df = df_list2tuple(df.copy())

        # Переводим в столбце все списки в кортежи для дальнейшей индексации
        # в ф-ии unique:
        all_values = df_list2tuple(df[df['track_id'] == obj_id][column])

        # Список уникальных значений:
        values = list(all_values.unique())

        # Поступаем по-разному в зависимости от числа вариантов:
        if len(values) == 0:
            return
        elif len(values) == 1:
            return values[0]
        else:
            raise IndexError(
                f'Для "{column}" найдено более одного значения: {values}!'
            )

    # Возвращает начальный кадр для заданного объекта:
    def get_start_frame(self, obj_id, df=None):
        return self.get_single_value_by_obj(obj_id, 'attributes', df)

    # Возвращает группу заданного объекта:
    def get_group(self, obj_id, df=None):
        return self.get_single_value_by_obj(obj_id, 'group', df)

    # Возвращает метку заданного объекта:
    def get_label(self, obj_id, df=None):
        return self.get_single_value_by_obj(obj_id, 'label', df)

    def set_single_value_by_obj(self, obj_id, column, value, df=None):

        # Используем исходный датафрейм, если он не передан:
        if df is None:
            df = self.df

        # Вносим значение во все нужные строки:
        df.loc[df['track_id'] == obj_id, column] = value

        return df

    # Задаёт начальный кадр для всех записей об объекте:
    def set_start_frame(self, obj_id, start_frame, df=None):
        return self.set_single_value_by_obj(obj_id, 'attributes',
                                            start_frame, df=None)

    # Задаёт группу для всех записей об объекте:
    def set_group(self, obj_id, group, df=None):
        return self.set_single_value_by_obj(obj_id, 'group',
                                            group, df=None)

    # Вносит новую подсказку:
    def set_points(self, msk_points, box_points, pos_points, neg_points,
                   frame, obj_id='new', label=None):

        # Делаем координаты плоским списком:
        msk_points = list(msk_points.flatten())
        box_points = list(box_points.flatten())
        pos_points = list(pos_points.flatten())
        neg_points = list(neg_points.flatten())

        # Берём новый объект, если надо:
        if obj_id == 'new':
            obj_id = self.get_new_obj_id()

        # Если метка не задана, ищем её по obj_id в уже имеющихся подсказках:
        if label is None:

            # Получаем список использованных этим объектом меток:
            labels = list(self.get_sub_df(obj_ids=obj_id)['label'].unique())

            if len(labels) == 0:
                label = 'unlabeled'
            elif len(labels) == 1:
                label = labels[0]
            else:
                raise ValueError('Более одного значения метки для obj_id=' +
                                 f'{obj_id}: {labels}!')

        # Определяем глобальные параметры трека:
        start_frame = self.get_start_frame(obj_id)  # Начальный кадр
        group = self.get_group(obj_id)              # Индекс группы

        # Если у этого объекта ещё нет записей в датафрейме, то ставим
        # значения по-умолчанию.
        if start_frame is None:
            start_frame = frame
        if group is None:
            group = 0

        # Если объект не новый, вычищаем все его старые метки из этого кадра:
        else:
            self.clear(frame, obj_id)

        # Общие параметры для внутренних и внешних точек:
        kws = {'frame': frame,
               'true_frame': frame,
               'track_id': obj_id,
               'group': group,
               'attributes': start_frame,
               'source': 'human'}

        # Вносим новые точки:
        if len(msk_points):
            self.df = add_row2df(self.df, points=msk_points, type='polygon',
                                 **kws)
        if len(box_points):
            self.df = add_row2df(self.df, points=box_points, type='polyline',
                                 **kws)
        if len(pos_points):
            self.df = add_row2df(self.df, points=pos_points, type='points',
                                 outside=False, **kws)
        if len(neg_points):
            self.df = add_row2df(self.df, points=neg_points, type='points',
                                 outside=True, **kws)

        # Ставим нужную метку всем записям текущего трека:
        self.df.loc[self.select_rows('track_id', obj_id), 'label'] = label

        self.make_auto_save()

    # Читает сохранённую подсказку:
    def get_points(self, frame, obj_id):

        # Берём все записи заданного объекта в заданном кадре:
        df = self.df
        df = df[(df['frame'] == frame) & (df['track_id'] == obj_id)]

        msk_points = df[df['type'] == 'polygon']['points']
        box_points = df[df['type'] == 'polyline']['points']
        points_mask = df['type'] == 'points'
        pos_points = df[points_mask & (df['outside'] == False)]['points']
        neg_points = df[points_mask & df['outside']]['points']

        assert len(msk_points) <= 1
        assert len(box_points) <= 1
        assert len(pos_points) <= 1
        assert len(neg_points) <= 1

        msk_points = msk_points.values[0] if len(msk_points) else []
        box_points = box_points.values[0] if len(box_points) else []
        pos_points = pos_points.values[0] if len(pos_points) else []
        neg_points = neg_points.values[0] if len(neg_points) else []
        msk_points = unflatten_list(msk_points, (-1, 2))
        box_points = unflatten_list(box_points, (-1, 2))
        pos_points = unflatten_list(pos_points, (-1, 2))
        neg_points = unflatten_list(neg_points, (-1, 2))

        return msk_points, box_points, pos_points, neg_points

    # Переводит датафрейм с многоугольником в маску в виде numpy-массива,
    # используемого в SAM2:
    def mask_df2mask(self, mask_df):
        assert len(mask_df) == 1
        points = CVATPoints.from_dfrow(mask_df.iloc[0, :],
                                       imsize=self.imsize[:2])
        mask = points.draw(thickness=-1)
        return mask

    # Переводит датафрейм с нужными точками в точки в виде numpy-массива,
    # используемого в SAM2:
    @staticmethod
    def points_df2np_points(points_df):

        # Если в датафрейме нет строк, возвращаем пустой массив:
        if len(points_df) == 0:
            return np.zeros((0, 2), np.float32)

        # Если в датафрейме ровно одна строка, берём её:
        elif len(points_df) == 1:
            return np.array(points_df.values[0],
                            dtype=np.float32).reshape(-1, 2)

        else:
            raise ValueError('В подсказках найдено несколько записей ' +
                             'одного объекта в кадре!')

    # Переводит датафрейм с нужным прямоугольником в прямоугольник в виде
    # numpy-массива, используемого в SAM2:
    @classmethod
    def box_df2np_box(cls, points_df):
        points_array = cls.points_df2np_points(points_df)
        return np.hstack([points_array.min(0), points_array.max(0)])

    # Возвращает список объектов, удовлетворяющих требованиям:
    def _obj_ids_preprop(self, obj_ids):
        if obj_ids is None:
            return self.get_all_obj_ids()
        elif isinstance(obj_ids, (list, tuple, set)):
            return obj_ids
        else:
            return [obj_ids]

    # Возвращает список ключевых кадров, удовлетворяющих требованиям:
    def _key_frames_preprop(self, frames):
        if frames is None:
            return self.get_all_key_frames()
        elif isinstance(obj_ids, (list, tuple, set)):
            return frames
        else:
            return [frames]

    # _obj_ids_preprop и _key_frames_preprop работают схожим образом:
    # 1) если требования не заданы - возвращаются все объекты;
    # 2) если передан список значений - он возвращается без изменений;
    # 3) если передано одно значение - возвращается список из одного
    #    элемента.

    # Генератор аргументов для методов add_new_mask и add_new_points_or_box
    # из SAM2:
    def GenerateSAM2kwargs(self, obj_ids=None, reverse=False):

        # Обробатываем список объектов:
        obj_ids = self._obj_ids_preprop(obj_ids)

        # Формируем упорядоченный список кадров с подсказками:
        frames = sorted(self.df['frame'].unique(), reverse=reverse)

        # Перебираем все кадры с подсказками:
        for frame in frames:

            # Берём все подсказки текущего кадра:
            frame_df = self.df[self.df['frame'] == frame]

            # Перебираем все объекты в кадре:
            for obj_id in frame_df['track_id'].unique():

                # Пропускаем объект, если он не в списке нужных:
                if obj_id not in obj_ids:
                    continue

                # Получаем подсказки текущего объекта в текущем кадре:
                obj_df = frame_df[frame_df['track_id'] == obj_id]

                # Разделяем данные на точки прямоугольник и маски:
                points_df = obj_df[obj_df['type'] == 'points']
                box_df = obj_df[obj_df['type'] == 'polyline']
                mask_df = obj_df[obj_df['type'] == 'polygon']

                # Инициируем словарь общих для add_new_mask и
                # add_new_points_or_box аргументов:
                base_kwargs = {'frame_idx': frame, 'obj_id': obj_id}

                # Инициируем словари для add_new_mask и
                # add_new_points_or_box аргументов:
                mask_kwargs = {}
                points_kwargs = {}

                # Если маска указана, формируем соотвествующий словарь:
                if len(mask_df):
                    mask = self.mask_df2mask(mask_df)
                    mask_kwargs = base_kwargs | {'mask': mask}

                # Обрабатываем точки, если есть:
                if len(points_df):

                    # Извлекаем списки координат и переводим в нужный формат:
                    pos_points = self.points_df2np_points(
                        points_df[points_df['outside'] == False]['points'])
                    neg_points = self.points_df2np_points(
                        points_df[points_df['outside']]['points'])

                    points = np.vstack([pos_points, neg_points])
                    labels = np.array([1] * len(pos_points) +
                                      [0] * len(neg_points))

                    # Дополняем словарь параметров точками:
                    points_kwargs |= {'points': points,
                                      'labels': labels}

                # Обрабатываем прямоугольник, если есть:
                if len(box_df):
                    box = self.box_df2np_box(box_df['points'])
                    points_kwargs |= {'box': box}

                # Если точки или прямоугольник есть, то добавляем к
                # соотетствующему словарю базовые параметры:
                if points_kwargs:
                    points_kwargs |= base_kwargs

                yield mask_kwargs, points_kwargs

    def get_sub_df(self, frames=None, obj_ids=None):
        '''
        Возвращает датафрейм, содержащий только указанные объекты в указанных
        кадрах.
        '''
        # Предобработка входных данных:
        frames = self._key_frames_preprop(frames)
        obj_ids = self._obj_ids_preprop(obj_ids)

        return self.df[self.df['frame'].isin(frames) &
                       self.df['track_id'].isin(obj_ids)]

    def hash(self, frames=None, obj_ids=None):
        '''
        Возвращает sha256 для всех подсказок заданного объекта.
        Полезно для проверки отсутствия изменений:
        '''
        # Получаем датафрейм с нужными записями:
        df = self.get_sub_df(frames, obj_ids)

        # Нумерация строк (нужно для корректного хеширования):
        df.index = range(len(df))

        # Сериализируем датафрейм:
        df_json = df.to_json()

        # Подсчитываем хеш:
        hash_object = sha256(df_json.encode())
        hash_code = hash_object.hexdigest()

        return hash_code


class SAM2:
    '''
    Обёртка вокруг Segment Anything Model 2
    '''
    checkpoint_name2yaml_name = {
        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
        'sam2.1_hq_hiera_large.pt': 'sam2.1_hq_hiera_l.yaml'
    }

    # Контекст инференса:
    @contextmanager
    def inference_context(self, device=None):

        # Берём сохранённое в объекте устройство, если явно не указано:
        if device is None:
            device = self.device

        # Извлекаем строку имени устройства, если надо:
        if not isinstance(device, str):
            deivce = device.type

        # Объединяем 2 контекста в 1:
        with torch.inference_mode() as im, \
                torch.autocast(deivce, dtype=torch.bfloat16) as ac, \
                warnings.catch_warnings() as cw:
            warnings.simplefilter("ignore")
            yield (im, ac, cw)

    # Перевод масок, возвращаемых моделью, в привычные для отрисовки и
    # векторизации:
    @staticmethod
    def mask_logit2mask(mask_logit):
        mask = (mask_logit > 0.0).cpu().numpy()
        assert mask.shape[0] == 1
        mask = mask[0, ...].astype(np.uint8) * 255
        return mask

    # Переводит точки формата IPYInteractiveSegmentation в данные для SAM2:
    def points2predictor(self,
                         msk_points, box_points, pos_points, neg_points):
        # Маска:
        mask = None
        if len(msk_points):
            imsize = self.imgs[self.frame].shape[:2]
            mask = CVATPoints(msk_points, imsize=imsize).draw(thickness=-1)
            if not mask.any():
                mask = None

        # прямоугольник:
        box = box_points.flatten() if len(box_points) else None

        # Точки:
        points = np.vstack([pos_points, neg_points])
        labels = np.array([1] * len(pos_points) + [0] * len(neg_points))
        if not len(labels):
            points = labels = None

        return mask, box, points, labels

    def __init__(self                                     ,
                 model_path   = '../sam2.1_hiera_large.pt',
                 config       = 'auto'                    ,
                 device       = 'auto'                    ,
                 tmp_dir      = None                      ):

        # Путь для модели:
        self.model_path = model_path

        # Доопределяем конфигурационый файл, если надо:
        if config == 'auto':
            self.config = self.checkpoint_name2yaml_name[
                os.path.basename(model_path)]
        else:
            self.config = config

        # Доопределяем устройство:
        if device == 'auto':
            self.device = AutoDevice()()
        else:
            self.device = device

        # TSV-Файл, хранящий все подсказки:
        self.prompts_file = None

        # Путь до врЕменной папки:
        self.tmp_dir = TemporaryDirectory(dir=tmp_dir)
        self.imgs_dir = os.path.join(self.tmp_dir.name, 'imgs_dir')

        # Создаём генератор масок:
        with self.inference_context():
            self.predictor = build_sam2_video_predictor(self.config,
                                                        self.model_path,
                                                        self.device)

        # Храним все подсказки здесь:
        self.prompts = None

        # Инициируем поля имён фото и видео:
        self.video_file = None

        # Ставим текущий кадр, номер трека и метку в значения по-умолчанию:
        self.frame = 0
        self.obj_id = 0
        self.label = 'unlabeled'

    # Удаляем врЕменную папку перед удалением объекта:
    def __del__(self):
        self.tmp_dir.cleanup()

    # Внесение видеофайла для обработки:
    def load_video(self, video_file, prompts_file=None, **kwargs):

        # Если файл подсказок не указан, размещаем его рядом с видео:
        if prompts_file is None:
            prompts_file = os.path.splitext(video_file)[0] + '.tsv'

        # Если в качестве файла подсказок указана папка, сохраняем в неё:
        elif os.path.isdir(prompts_file):
            prompts_file = os.path.join(prompts_file,
                                        os.path.splitext(
                                            os.path.basename(video_file)
                                        )[0] + '.tsv')

        self.video_file = video_file
        self.prompts_file = prompts_file

        # Сбрасываем содержимое врЕменной папки:
        self.tmp_dir.cleanup()
        mkdirs(self.imgs_dir)

        # Заполняем врЕменную подпапку JPEG-кадрами из видео:
        for ind, img in enumerate(tqdm(VideoGenerator(self.video_file),
                                       desc='Пересжатие видео')):

            # Если появился первый пропущенный кадр, прерываем конвертацию:
            if img is None:
                break

            # Пишем очередной кадр:
            cv2.imwrite(os.path.join(self.imgs_dir, '%08d.jpg' % ind),
                        img[:, :, ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Читатель кадров из врЕменной папки:
        self.imgs = VideoGenerator(self.imgs_dir)

        # База данных подсказок:
        self.prompts = Prompts(len(self.imgs), self.imgs[0].shape,
                               self.prompts_file)

        # Инициируем внутреннее состояние трекера:
        with self.inference_context():
            self.state = self.predictor.init_state(self.imgs_dir, **kwargs)

        # Инициируем хеш подсказок объектов, кеш датафреймов их треков
        # и общий кеш всей предразметки:
        self.obj_id2obj_promtps_hash = {}
        self.obj_id2obj_df_cache = {}
        self.last_df_hash = None
        self.last_df= None
        # Хеш позволяет проверять были ли изменены подсказки со времён
        # сохранения последнего трекинга.

    # Сброс состояния трекера:
    def _reset_state(self):
        with self.inference_context():
            self.predictor.reset_state(self.state)

    # Актуализируем объект во внутренних состояниях инференса:
    def _set_state(self, obj_ids=None, reverse=False):
        # Сброс состояний:
        self._reset_state()

        # Запись состояний с нуля:
        with self.inference_context():

            # Последовательно изменяем внутреннее состояние, внесением
            # новых подсказок:
            for mask_kwargs, points_kwargs in \
                    self.prompts.GenerateSAM2kwargs(obj_ids, reverse):
                if mask_kwargs:
                    self.predictor.add_new_mask(self.state,
                                                **mask_kwargs)
                if points_kwargs:
                    self.predictor.add_new_points_or_box(self.state,
                                                         **points_kwargs)

    def build_obj_df(self, obj_id):
        '''
        Формирует датафрейм заданного трека:
        '''

        # Вычисляем хеш актуальных подсказок для трека:
        prompts_hash = self.prompts.hash(obj_ids=obj_id)

        # Вытаскиваем датафрейм сразу из кеша, если хеши совпадают:
        if self.obj_id2obj_promtps_hash.get(obj_id, None) == prompts_hash:
            return self.obj_id2obj_df_cache[obj_id]

        # Если подсказки с тех пор поменялись, выполняем трекинг заново:

        # Инициируем итоговый датафрейм трека:
        obj_df = new_df()

        # Определяем глобальные параметры трека:
        start_frame = self.prompts.get_start_frame(obj_id)  # Начальный кадр
        group = self.prompts.get_group(obj_id)              # Индекс группы
        label = self.prompts.get_label(obj_id)              # Метка (класс)

        # Формируем датафрейм прямого прохода:
        with self.inference_context():

            # Применяем последовательно прямой и обратный проходы:
            for reverse in [False, True]:

                # Актуализируем внутреннее состояние инференса:
                self._set_state(obj_id, reverse)

                # Инициируем список для объектов текущего прохода:
                obj_dfs = []

                # Генератор масок (трекер):
                generator = self.predictor.propagate_in_video(
                    self.state,
                    start_frame_idx=start_frame,
                    reverse=reverse
                )

                # Перебираем все кадры:
                for frame, obj_ids, mask_logits in generator:

                    # Перебираем все маски в кадре:
                    for obj_id_, mask_logit in zip(obj_ids, mask_logits):

                        assert obj_id_ == obj_id

                        # На обратном проходе начальный кадр пропускаем, т.к.
                        # он уже был обработан на прямом проходе:
                        if reverse and frame == start_frame:
                            continue

                        # Извлекаем маску:
                        mask = self.mask_logit2mask(mask_logit)

                        # Выполняем векторизацию если маска не пуста:
                        points = None
                        if mask.any():
                            points = CVATPoints.from_mask(
                                mask, cv2.CHAIN_APPROX_TC89_KCOS)

                        # Если маска не опустела и после векторизации, то
                        # преводим её в строку датафрейма и добавляем в
                        # список:
                        if points is None:
                            continue
                        obj_dfs.append(points.to_dfrow(track_id=obj_id,
                                                       label=label,
                                                       frame=frame,
                                                       true_frame=frame,
                                                       group=group,
                                                       source='SAM2'))

                # Добавляем результаты последнего прохода в датафрейм трека:
                obj_df = concat_dfs([obj_df] + obj_dfs)

        # Обновляем хеш и кеш:
        self.obj_id2obj_promtps_hash[obj_id] = prompts_hash
        self.obj_id2obj_df_cache[obj_id] = obj_df

        return obj_df

    def build_subtask(self, fit_segments=False):

        # Путь к видеофайлу и словарь прореженных кадров:
        file = self.video_file
        true_frames = {frame: frame for frame in range(len(self.imgs))}

        # Достаём разметку с подгоном контуров из кеша, если подсказки не
        # менялись:
        if fit_segments:
            prompts_hash = self.prompts.hash()
            if (self.last_df_hash == prompts_hash):
                return self.last_df, file, true_frames

        # Инициируем список датафреймов:
        dfs = []

        # Перебираем все треки и объединяем результаты в один датафрейм:
        all_obj_ids = self.prompts.get_all_obj_ids()
        for obj_id in all_obj_ids:
            label = self.prompts.get_label(obj_id)
            print(f'Трассировка объекта №{obj_id}',
                  f'(из {max(all_obj_ids)}):',
                  label)
            dfs.append(self.build_obj_df(obj_id))
        df = concat_dfs(dfs)
        if df is None:
            df = new_df()

        # Выполняем подгонку контуров, если надо:
        if fit_segments:
            df = fit_segments_in_df(
                df, self.imgs[self.frame].shape[:2],
                mpmap_kwargs={'desc': 'Подгонка контуров'},
                fuse_by_groups=True
            )

        # Скрываем объекты в местах пропусков (отрубаем хвосты):
        df = hide_skipped_objects_in_df(df, true_frames)

        # Обновляем кеш и хеш датафрейма с подгонкой контуров:
        if fit_segments:
            self.last_df = df
            self.last_df_hash = prompts_hash

        return df, file, true_frames

    # Меняем текущие кадр, объект и метку:
    def go_to(self, frame=None, obj_id='new', label=None):

        # Если номер кадра не указан, то не меняем:
        if frame is None:
            pass
        # Если номер кадра указан, то:
        else:
            # Исправляем выход за диапазон кадров:
            while frame < 0:
                frame = frame + len(self.imgs)
            while frame >= len(self.imgs):
                frame = frame - len(self.imgs)
            # Фиксируем полученный кадр:
            self.frame = frame

        # Если нужен новый объект:
        if obj_id == 'new':
            self.obj_id = self.prompts.get_new_obj_id()
        # Если нужен объект со старшим индексом:
        elif obj_id == 'last':
            self.obj_id = max(self.prompts.get_all_obj_ids())
        # Если объект не указан, то не меняем:
        elif obj_id is None:
            pass
        # Если указан, то вносим:
        else:
            self.obj_id = int(obj_id)

        # Меняем метку класса:
        if label is not None:
            self.label = label

        # Возвращаем сам текущий кадр:
        return self.imgs[self.frame].copy()

    # Меняем текущую метку кдасса:
    def set_label(self, label):
        self.go_to(frame=None, obj_id=None, label=label)
        return

    def points2masks(self, msk_points, box_points, pos_points, neg_points):

        # Временно сбрасываем все предыдущие и последующие точки трекера:
        self._reset_state()

        # Заносим точки в базу данных:
        self.prompts.set_points(msk_points,
                                box_points, pos_points, neg_points,
                                self.frame, self.obj_id, self.label)

        mask, box, points, labels = self.points2predictor(
            msk_points, box_points, pos_points, neg_points)

        # Словарь общих для add_new_mask и add_new_points_or_box параметров:
        kwargs = {'inference_state': self.state,
                  'frame_idx': self.frame,
                  'obj_id': self.obj_id}
        with self.inference_context():
            # Если маски нет, то заносим только её:
            if mask is not None:
                _, obj_ids, mask_logits = \
                    self.predictor.add_new_mask(mask=mask, **kwargs)
            # Если маски нет, но есть точки, то вносим их:
            elif labels is not None or box is not None:
                _, obj_ids, mask_logits = \
                    self.predictor.add_new_points_or_box(box=box,
                                                         points=points,
                                                         labels=labels,
                                                         **kwargs)
            # Если никаких данных не передано, то возвращаем список без масок:
            else:
                return []

        # Восстанавливаем маску, полученную после внесения подсказок:
        mask = np.zeros(self.imgs[self.frame].shape[:2], dtype=np.uint8)
        for obj_id, mask_logit in zip(obj_ids, mask_logits):
            if obj_id == self.obj_id:
                mask |= self.mask_logit2mask(mask_logit)

        return [mask]

    def gen_preview(self, fit_segments=False):
        df, _, _ = self.build_subtask(fit_segments)

        obj_ids = sorted(list(df['track_id'].unique()))

        label2color = {}
        for ind, obj_id in enumerate(obj_ids):
            label2color[obj_id] = \
                color_float_hsv_to_uint8_rgb(ind / len(obj_ids))

        for frame, img in enumerate(self.imgs):
            img = ergonomic_draw_df_frame(df[df['frame'] == frame], img)

            yield img

    def render_preview(self, target_file=None, label2color=None,
                      fit_segments=False, return_subtask=False):

        # Доопределяем имя файла, если не задан:
        if target_file is None:
            target_file = 'Clean' if fit_segments else 'Druft'
            target_file = target_file + '_Preview.mp4'
            target_file = os.path.join(self.tmp_dir.name, target_file)

        # Формируем CVAT-подзадачу с разметкой:
        df, file, true_frames = self.build_subtask(fit_segments)

        # Подменяем метки, чтобы разные объекты отрисовывались разными цветами:
        mod_df = df.copy()
        if not fit_segments:
            mod_df['label'] = mod_df['track_id'].apply(str)

        # Если в качестве файла передан список изображений, то берём имя
        # папки:
        if isinstance(file, (list, tuple)):
            name = os.path.basename(os.path.dirname(os.path.abspath(file[0])))

        # Иначе берём имя самого исходного файла:
        else:
            name = os.path.splitext(os.path.basename(file))[0]

        # Генерируем само превью:
        subtask2preview((mod_df, file, true_frames),
                        target_file, label2color,
                        desc='Отрисовка превью')

        # Возвращачем имя файла-превью и, опционально, всю подзадачу с
        # разметкой:
        if return_subtask:
            subtask = (df, file, true_frames)
            return target_file, subtask
        else:
            return target_file

    # Созадёт словарь входных параметров для инициализации
    # IPYInteractiveSegmentation из ipy_utils:
    def init_ipis_kwargs(self):
        # Получаем подсказки для текущего объекта в текущем кадре:
        msk_points, box_points, pos_points, neg_points = \
            self.prompts.get_points(frame=self.frame, obj_id=self.obj_id)
        # Формируем словарь параметров:
        kwargs = {'img': self.imgs[self.frame],
                  'points2masks': self.points2masks,
                  'init_msk_points': msk_points,
                  'init_box_points': box_points,
                  'init_pos_points': pos_points,
                  'init_neg_points': neg_points}

        return kwargs

    # Создаёт файл CVAT-разметки:
    def save2cvat_xml(self,
                      xml_file=None,
                      fit_segments=True):

        # Размещаем xml во временной папке, если путь не указан явно:
        if xml_file is None:
            xml_file = os.path.join(self.tmp_dir.name, 'annotation.xml')

        # Формируем подзадачу:
        df, file, true_frames = self.build_subtask(fit_segments)

        # Сохраняем в xml-файл:
        return subtask2xml((df, file, true_frames), xml_file)

    # Возвращает параметры создания/обновления задачи в CVAT:
    def cvat_new_task_kwargs(self, proj_name=None, fit_segments=True):

        # Определяем имя задачи:
        task_name = os.path.splitext(os.path.basename(self.video_file))[0]
        if proj_name:
            task_name = f'{proj_name}_{task_name}'

        # Генерируем XML-файл с разметкой:
        xml_file = self.save2cvat_xml(fit_segments=fit_segments)

        return {'name': task_name,
                'file': self.video_file,
                'annotation_file': xml_file}

    # Создаёт в заданном CVAT-датасете новую задачу, или обновляет разметку в
    # уже имеющейся:
    def export2cvat(self, cvat_proj, fit_segments=True):

        # Формируем словарь аргументов:
        kwargs = self.cvat_new_task_kwargs(cvat_proj.name,
                                           fit_segments=fit_segments)

        # Создаём или обновляем задачу в CVAT:
        if kwargs['name'] in cvat_proj.keys():
            task = cvat_proj[kwargs['name']]
            task.set_annotations(kwargs['annotation_file'])
        else:
            task = cvat_proj.new(**kwargs)
            task.update({'subset': 'Train'})

        return task.values()[0].url