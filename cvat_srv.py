from cvat_sdk import make_client
from PIL import Image
import numpy as np

from cvat import add_row2df, concat_dfs, ergonomic_draw_df_frame


def get_name(obj):
    '''
    Возвращает имя заданного объекта.
    Используется для сортировки списка объектов по их именам.
    '''
    return obj.name


class CVATSRVObj:
    '''
    Абстрактный класс для проектов, задач, подзадач и прочих сущностей из
    cvat_sdk. Составляет основу древовидной структуру данных:
    Сервер > Проекты > Задачи > Подзадачи ...

    Объект ведёт себя как словарь, ключами к которому являются имена
    подобъектов и самими подобъектами в качестве значений. Т.е. реализованы
    методы: __getitem__, keys, values, items.
    '''

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

    def parent(self):
        '''
        Должен возвращать объект-предок.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def __init__(self, client, obj=None):
        self.client = client
        self.obj = obj

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


class CVATSRVJob(CVATSRVObj):
    '''
    Поздазача CVAT-сервера.
    '''

    def values(self):
        raise NotImplementedError('У подзадачи нет составляющих!')

    def parent(self):
        for task in self.client.get_tasks():
            if task.id == self.obj.task_id:
                return CVATSRVTask(self.client, task)
        return None

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


class CVATSRVTask(CVATSRVObj):
    '''
    Здазача CVAT-сервера.
    '''

    def values(self):
        return [CVATSRVJob(self.client, job) for job in self.obj.get_jobs()]

    def parent(self):
        for project in self.client.projects.list():
            if project.id == self.obj.project_id:
                return CVATSRVProject(self.client, project)
        return None


class CVATSRVProject(CVATSRVObj):
    '''
    Датасет CVAT-сервера.
    '''

    def values(self):
        return [CVATSRVTask(self.client, task)
                for task in self.obj.get_tasks()]

    def parent(self):
        return self.client


class CVATSRV(CVATSRVObj):
    '''
    CVAT-сервер.
    '''

    def values(self):
        return [CVATSRVProject(self.client, project)
                for project in self.client.projects.list()]

    def parent(self):
        raise NotImplementedError('У сервера нет предков!')

    def __init__(self, host, username, password):
        self.name = host
        client = make_client(host=host,
                             credentials=(username, password))
        super().__init__(client)