import os
from PIL import Image
import numpy as np
from cvat_sdk import make_client

from cvat import add_row2df, concat_dfs, ergonomic_draw_df_frame
from utils import mkdirs, rmpath, mpmap


class Client:
    def __init__(self, host, username, password):
        self.host = host
        self.username = username
        self.password = password
        self.obj = make_client(host=host,
                               credentials=(username, password))

    def kwargs4init(self):
        '''
        Возвращает словарь аргументов для создания копии экземпляра класса:
        '''
        return {'host': self.host,
                'username': self.username,
                'password': self.password}
    # Пример использования:
    # client = Client(...)
    # client2 = Client(**client.kwargs4init())

    def __getattr__(self, attr):
        '''
        Проброс атрибутов вложенного объекта наружу.
        '''
        return getattr(self.obj, attr)


def get_name(obj):
    '''
    Возвращает имя заданного объекта.
    Используется для сортировки списка объектов по их именам.
    '''
    return obj.name


class _CVATSRVObj:
    '''
    Абстрактный класс для проектов, задач, подзадач и прочих сущностей из
    cvat_sdk. Составляет основу древовидной структуру данных:
    Сервер > Проекты > Задачи > Подзадачи ...

    Объект ведёт себя как словарь, ключами к которому являются имена
    подобъектов и самими подобъектами в качестве значений. Т.е. реализованы
    методы: __getitem__, keys, values, items.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = None
    # У каждого потомка будет свой номер.

    def __init__(self, client, obj=None):
        self.client = client
        self.obj = obj

        # Иерархия всех потомков данного суперкласса:
        self._hier_lvl2class = {0: CVATSRV,
                                1: CVATSRVProject,
                                2: CVATSRVTask,
                                3: CVATSRVJob}

        # Классы предков и потомков:
        self.child_class = self._hier_lvl2class.get(self._hier_lvl + 1, None)
        self.parent_class = self._hier_lvl2class.get(self._hier_lvl - 1, None)

    @classmethod
    def from_id(cls, client, id):
        '''
        Должен возвращать экземпляр класса объекта по его id.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def name2id(self, name):
        '''
        Возвращает id объекта-потомка по его имени.
        '''
        for key, val in self.items():
            if key == name:
                return val.id
        return None

    @classmethod
    def _backup(cls, client_kwargs, child_class, child_id, path):
        '''
        Создаёт бекап, предварительно пересоздавая клиент.
        Нужно для использования mpmap, работающей только с сериализуемыми
        объектами.
        '''
        client = Client(**client_kwargs)
        child = child_class.from_id(client, child_id)
        return child.backup(path)

    def backup(self,
               path='./',
               name=None,
               desc='Загрузка бекапов',
               **mpmap_kwargs):
        '''
        Сохраняет бекап текущей сущности, либо набора его подобъектов.
        '''
        # Присовокупляем описание к остальным параметрам mpmap:
        mpmap_kwargs = mpmap_kwargs | {'desc': desc}

        # Если передан целый список имён подсущностей, которые надо бекапить:
        if isinstance(name, (tuple, list, set)):

            # Множество принудительно делаем списком:
            if isinstance(name, set):
                name = list(name)
            # Чтобы порядок элементов был фиксирован.

            # Если путь только один - считаем его именем папки:
            if isinstance(path, str):
                path = [os.path.join(path, name_ + '.zip') for name_ in name]
            # Имена файлов по именам подсущностей.

            if len(name) != len(path):
                raise ValueError('Число имён и файлов должно совпадать!')

            # Бекапим все сущности:
            client_kwargs = [self.client.kwargs4init()] * len(name)
            child_classes = [self.child_class] * len(name)
            child_ids = list(map(self.name2id, name))
            return mpmap(self._backup, client_kwargs, child_classes,
                         child_ids, path, **mpmap_kwargs)

        # Если бекапить надо всю текущую сущность:
        elif name is None:

            # Если текущая сущность вообще бекапится:
            if hasattr(self.obj, 'download_backup'):

                # Если указан не путь до архива, считаем его папкой:
                if path.lower()[-4:] != '.zip':
                    path = os.path.join(path, self.name + '.zip')
                    # Дополняем путь именем сущности.

                # Создаём папки до файла, если надо:
                mkdirs(os.path.dirname(path))

                # Удаляем файл, если он уже существует:
                if os.path.isfile(path):
                    rmpath(path)

                # Качаем сам бекап:
                self.obj.download_backup(path)

                # Возвращаем путь до файла:
                return path

            else:
                raise Exception('Объект не бекапится!')

        # Если передано имя лишь одной подсущности:
        else:
            return self[name].backup(path)

    def values(self):
        '''
        Должен возвращать список вложенных объектов.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def sorted_values(self):
        return sorted(self.values(), key=get_name)

    def __iter__(self):
        return iter(self.values())

    def __len__(self):
        return len(self.values())

    def parend_id(self):
        '''
        Должен возвращать id объекта-предка.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def parent(self):
        '''
        Dозвращает объект-предок.
        '''
        if self._hier_lvl == 0:
            raise NotImplementedError('Объект не имеет предков!')

        return self._hier_lvl2class[self._hier_lvl - 1].from_id(
            self.client, self.parend_id()
        )

    def __getattr__(self, attr):
        '''
        Проброс атрибутов вложенного объекта наружу.
        '''
        return getattr(self.obj, attr)

    def keys(self):
        '''
        Возвращает список имён входящих в объект подобъектов.
        '''
        return [subobject.name for subobject in self.values()]

    def items(self):
        '''
        Возращает список кортежей из пар (имя_подобъекта, подобъект).
        '''
        return [(subobject.name, subobject) for subobject in self.values()]

    def __getitem__(self, key):
        '''
        Возвращает нужный подобъект по его имени.
        '''
        for name, subobject in self.items():
            if name == key:
                return subobject
        raise IndexError(f'Не найден "{key}"!')

    def __str__(self):
        '''
        Возвращает име объекта при выводе содержимого объетка в виде текста.
        '''
        return str(self.name)


class CVATSRVJob(_CVATSRVObj):
    '''
    Поздазача CVAT-сервера.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 3

    def values(self):
        raise NotImplementedError('У подзадачи нет составляющих!')

    def parend_id(self):
        return self.obj.task_id

    @property
    def name(self):
        '''
        У поздазач нет имён, поэтому используем их ID.
        '''
        return self.id

    # Создаёт датафрейм всех проблем:
    def issues2df(self):
        dfs = []
        for issue in self.obj.get_issues():

            # Извлекаем описание проблемы:
            comments = issue.get_comments()
            assert len(comments) == 1

            # Фиксируем основные параметры проблемы:
            message = comments[0].message
            bbox = issue.position
            frame = issue.frame
            resolved = issue.resolved

            # Добавляем новый датафрейм:
            df = add_row2df(label=message, frame=frame, true_frame=frame,
                            type='rectangle', points=bbox, outside=resolved)
            dfs.append(df)

        # Объединяем список в один датафреймф и возвращаем:
        return concat_dfs(dfs)

    # Возвращает кадр из подзадачи:
    def get_frame(self, frame, quality='original'):
        data = self.obj.get_frame(int(frame), quality=quality)
        pil_img = Image.open(data)
        return np.array(pil_img)

    # Создаёт превью кадров с проблемами:
    def draw_issues(self, df=None):
        # Берём все проблемы, если датафрейм не задан явно:
        if df is None:
            df = self.issues2df()

        # Инициируем список проблемных кадров:
        previews = []

        # Заполняем список:
        for frame in sorted(df['frame'].unique()):
            # Формируем датафрейм проблем текущего кадра:
            frame_df = df[df['frame'] == frame]

            # Читаем сам кадр:
            img = self.get_frame(frame)

            # Выполняем отрисовку проблем на кадре:
            preview = ergonomic_draw_df_frame(frame_df, img)

            # Пополняем список превою проблемных кадров:
            previews.append(preview)

        return previews

    @classmethod
    def from_id(cls, client, id):
        return cls(client, client.jobs.retrieve(id))


class CVATSRVTask(_CVATSRVObj):
    '''
    Здазача CVAT-сервера.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 2

    def values(self):
        return [CVATSRVJob(self.client, job) for job in self.obj.get_jobs()]

    def parend_id(self):
        return self.obj.project_id

    @classmethod
    def from_id(cls, client, id):
        return cls(client, client.tasks.retrieve(id))


class CVATSRVProject(_CVATSRVObj):
    '''
    Датасет CVAT-сервера.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 1

    def values(self):
        return [CVATSRVTask(self.client, task)
                for task in self.obj.get_tasks()]

    def parent(self):
        return CVATSRV(**self.client.kwargs4init())

    @classmethod
    def from_id(cls, client, id):
        return cls(client, client.projects.retrieve(id))


class CVATSRV(_CVATSRVObj):
    '''
    CVAT-сервер.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 0

    # У сервера нет своего ID:
    id = None

    def values(self):
        return [CVATSRVProject(self.client, project)
                for project in self.client.projects.list()]

    def parent(self):
        raise NotImplementedError('У сервера нет предков!')

    def __init__(self, host, username, password):
        self.name = host
        client = Client(host, username, password)

        super().__init__(client)