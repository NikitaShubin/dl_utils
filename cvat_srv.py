import os
from PIL import Image
import numpy as np
from cvat_sdk import make_client, core
from getpass import getpass
from zipfile import ZipFile

from cvat import (add_row2df, concat_dfs, ergonomic_draw_df_frame,
    cvat_backups2raw_tasks)
from utils import mkdirs, rmpath, mpmap, get_n_colors, unzip_dir


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

    @staticmethod
    def _adapt_labels(labels):
        '''
        Добавляет к списку классов цвета, если надо.
        '''
        assert isinstance(labels, (tuple, list, set))

        # Определяем, описаны метки строками или словарями:
        is_str = False
        is_dict = False
        for label in labels:
            if isinstance(label, dict):
                is_dict = True
            elif isinstance(label, str):
                is_str = True
            else:
                raise ValueError(
                    'Список классов должен содержать строки, словари ' +
                    'или `cvat_sdk.api_client.model.label.Label`. ' +
                    f'Получен "{type(label)}!"')

        # Если передан словарь, то оставляем в нём только самое необъодимое:
        if is_dict and not is_str:

            # Инициируем список новых меток:
            labels_ = []

            # Перебираем исходные метки:
            for label in labels:

                # Инициируем новую метку
                label_ = {}
                for key, val in label.items():
                    if key in {'name', 'attributes', 'color'}:
                        label_[key] = val

                # Пополняем список новых меток:
                labels_.append(label_)

            # Заменяем старый список новым:
            labels = labels_

        # Полезно в случае дублирования меток из проекта в задачу,
        # т.к. не все поля проекта соответствуют полям задач.

        # Дополняем названия цветами, если переданы только строки:
        elif is_str and not is_dict:
            colors = get_n_colors(len(labels))
            labels = [
                {
                    'name': label,
                    'attributes': [],
                    'color': '#%02x%02x%02x' % color,
                } for label, color in zip(labels, colors)
            ]

        else:
            print(f'is_str = {is_str}; is_dict = {is_dict}')
            raise ValueError('Параметр labels может быть списком ' +
                             'либо строк, либо словарей!')

        return labels

    @staticmethod
    def _adapt_file(file):
        '''
        Делает путь к файлу списком файлов.
        '''
        if isinstance(file, str):
            file = [file]
        elif not isinstance(file, list):
            raise ValueError('Параметр file должен быть строкой или ' +
                             f'списком строк. Получен {type(file)}!')

        # Проверяем наличие каждого файла из списка:
        for file_ in file:
            if not os.path.isfile(file_):
                raise FileNotFoundError(f'Не найден файл "file_"!')

        return file

    def new_task(self,
                 name,
                 file,
                 labels,
                 annotation_path=None):
        '''
        Создаёт новую задачу без разметки.
        '''

        # Доводим до ума входные параметры, если надо:
        labels = self._adapt_labels(labels)
        file = self._adapt_file(file)

        task_spec = {'name': name,
                     'labels': labels}

        task = self.obj.tasks.create_from_data(
            spec=task_spec,
            resource_type=core.proxies.tasks.ResourceType.LOCAL,
            resources=file,
            annotation_path=annotation_path,
            annotation_format='CVAT 1.1',
        )

        # Возвращаем обёрнутый объект:
        return CVATSRVTask(self, task)

    def new_project(self, name, labels):
        '''
        Создаёт пустой датасет.
        '''
        # Добавляет классам цвета, если надо:
        labels = self._adapt_labels(labels)

        # Создаём проект:
        proj_spec = {'name': name,
                     'labels': labels}
        project = self.obj.projects.create_from_dataset(proj_spec)

        # Возвращаем обёрнутрый объект:
        return CVATSRVProject(self, project)

    def restore_task(self, backup_file):
        '''
        Восстанавливает задачу из бекапа.
        '''
        task = self.obj.api_client.tasks_api.create_backup(
            filename=backup_file
        )
        return CVATSRVTask(self, task)

    def restore_project(self, backup_file):
        '''
        Восстанавливает проект из бекапа.
        '''
        project = self.obj.api_client.projects_api.create_backup(
            filename=backup_file
        )
        return CVATSRVProject(self, project)


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
                mkdirs(os.path.abspath(os.path.dirname(path)))

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

    def restore(self, backup_file):
        '''
        Восстанавливает задачу или проект по его бекапу.
        '''
        # Если сейчас мы на уровне сервера, то восстанавливаем проект:
        if self._hier_lvl == 0:
            return self.client.restore_project(backup_file)

        # Если сейчас мы на уровне проекта, то восстанавливаем задачу:
        if self._hier_lvl == 1:
            task = self.client.restore_task(backup_file)

            # Привязываем задачу к текущему датасету:
            task.update({'project_id': self.id})

            return task

        else:
            raise NotImplementedError(
                'Восстанавливать можно лишь проекты и задачи!'
            )

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

    @property
    def parend_id(self):
        '''
        Должен возвращать id объекта-предка.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def parent(self):
        '''
        Возвращает объект-предок.
        '''
        if self._hier_lvl == 0:
            raise NotImplementedError('Объект не имеет предков!')

        # Возвращаем None, если предка нет:
        if self.parend_id is None:
            return

        return self._hier_lvl2class[self._hier_lvl - 1].from_id(
            self.client, self.parend_id
        )

    def new(self):
        '''
        Создаёт и возвращает подобъект.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

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

    @property
    def url(self):
        '''
        Возвращает URL объекта.
        '''
        return self.obj.url.replace('/api/', '/')

    def set_annotations(self,
                        file='annotations.xml',
                        format_name='CVAT 1.1',
                        *args, **kwargs):
        '''
        Заменяет имеющуюся разметку объекта новой.
        '''
        return self.obj.import_annotations(format_name, file, *args, **kwargs)

    def get_annotations(self,
                        file='annotations.xml',
                        format_name='CVAT for video 1.1',
                        *args, **kwargs):
        '''
        Сохраняет имеющуюся разметку объекта в локальный файл.
        '''
        if os.path.isfile(file):
            raise FileExistsError(f'Файл "{file}" уже существует!')

        # Создаём целевую папку, если надо:
        dirname = os.path.dirname(file)
        if not os.path.isdir(dirname):
            mkdirs(dirname)

        # Если требуется скачать именно архив:
        if os.path.splitext(file)[1].lower() == '.zip':

            # Качаем:
            self.obj.export_dataset(format_name,
                                    file,
                                    include_images=False,
                                    *args, **kwargs)

        # Если требуется распаковка файла:
        else:

            # Определяем имя архива для закачки:
            zip_file = file + '.zip'
            if os.path.isfile(zip_file):
                raise FileExistsError(f'Файл "{zip_file}" уже существует!')

            # Качаем архив:
            self.obj.export_dataset(format_name,
                                    zip_file,
                                    include_images=False,
                                    *args, **kwargs)

            # Имя конечного файла без пути к нему:
            basename_file = os.path.basename(file)

            # Открываем архив:
            with ZipFile(zip_file, 'r') as archive:

                # Перебираем все файлы в нём:
                for zipped_file in archive.namelist():

                    # Находим файл разметки:
                    if zipped_file.startswith('annotations.'):

                        # Распаковываем его под новым именем:
                        archive.getinfo(zipped_file).filename = basename_file
                        archive.extract(zipped_file, os.path.dirname(file))

                        # Выходим из цикла:
                        break

                # Если файл не найден:
                else:
                    raise FileNotFoundError('В ахриве не найден файл, ' +
                                            'начинающийся с "annotations."!')
            # Удаляем архив:
            rmpath(zip_file)

        return file


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

    @property
    def url(self):
        '''
        Возвращает URL подзадачи.
        '''
        task_substr = f'/tasks/{self.parend_id()}/jobs/'
        return super().url.replace('/jobs/', task_substr)


class CVATSRVTask(_CVATSRVObj):
    '''
    Здазача CVAT-сервера.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 2

    def values(self):
        return [CVATSRVJob(self.client, job) for job in self.obj.get_jobs()]

    @property
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

    def labels(self):
        '''
        Возвращает список словарей, описывающих каждую используемую метку
        датасета:
        '''
        return [label.to_dict() for label in self.obj.get_labels()]

    def new(self, name, file, annotation_file=None, tmp_name='unfinished'):
        '''
        Создаёт новую задачу в текущем датасете.
        '''
        # Задачи с таким именем ещё не должно существовать:
        if name in self.keys():
            raise KeyError(f'Задача "{name}" уже существует ' +
                           f'в датасете "{self.name}"!')

        # Извлекаем метки из дадасета:
        labels = self.labels()

        # Создаём задачу под временным именем:
        task = self.client.new_task(tmp_name, file, labels,
                                    annotation_path=annotation_file)

        # Привязываем задачу к текущему датасету:
        task.update({'project_id': self.id})

        # Даём задаче законченное имя:
        task.update({'name': name})
        # В случае прерывания процесса создания задачи её можно будет легко
        # найти по временному имени, чтобы удалить.

        return task

    def project_json(self):
        '''
        Возвращает словарь, описывающий весь датасет.
        Бекапы проектов содержат файл project.json с подобным описанием.
        '''
        return {'name': self.name,
                'bug_tracker': self.bug_tracker,
                'status': self.status,
                'labels': self.labels(),
                'version': '1.0'}


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

    def __init__(self, host=None, username=None, password=None):
        host = host or input('Адрес CVAT-сервера:')
        username = username or input(f'{host} логин: ')
        password = password or getpass(f'{host} пароль: ')

        self.name = host
        client = Client(host, username, password)

        super().__init__(client)

    def new(self, name, labels):
        '''
        Создаёт новый датасет на текущем сервере.
        '''
        # Датасета с таким именем ещё не должно существовать:
        if name in self.keys():
            raise KeyError(f'Датасет "{name}" уже существует!')

        # Создаём датасет:
        return self.client.new_project(name, labels)


class ReadTasks:
    '''
    Контекст, чтения данных из CVAT. По завершении контектста все данные,
    закаченные из CVAT удаляются, но при использовании как функтера удаления
    не происходит. Полезно для формирования актуального датасета и прочих
    операций, требующих создания временных копий актуальных данных из CVAT.

    backup_path и extract_path удаляются целиком!
    '''

    def __init__(self,
                 backup_path,
                 extract_path,
                 cvat_obj,
                 cvat_subobj_names=None,
                 desc=None):
        self.backup_path = backup_path
        self.extract_path = extract_path
        self.cvat_obj = cvat_obj
        self.cvat_subobj_names = cvat_subobj_names
        self.desc = desc

    def __call__(self):

        # Очистка дирректорий для временных файлов:
        self.rmdirs()
        mkdirs(self.backup_path)
        mkdirs(self.extract_path)

        # Доопределяем текстовые описания грядущих процессов:
        if self.desc:
            backup_desc = f'{self.desc}: Загрузка бекапа(ов)'
            extract_desc = f'{self.desc}: Распаковка бекапа(ов)'
            parse_desc = f'{self.desc}: Парсинг бекапа(ов)'
        else:
            backup_desc = extract_desc = parse_desc = ''

        # Закачиваем бекапы:
        self.cvat_obj.backup(self.backup_path,
                             self.cvat_subobj_names,
                             backup_desc)

        # Распаковываем бекапы:
        unzip_dir(self.backup_path, self.extract_path, extract_desc)

        # Парсим бекапы:
        return cvat_backups2raw_tasks(self.extract_path, parse_desc)

    # Очистка всех временных папок
    def rmdirs(self):

        # Определяем, существуют ли папки, что должны быть удалены:
        backup_path_exists = os.path.isdir(self.backup_path)
        extract_path_exists = os.path.isdir(self.extract_path)

        # Ничего не делаем, если папки не существуют:
        if not (backup_path_exists or extract_path_exists):
            return

        # Формируем строку описания процесса:
        if self.desc:
            desc = f'{self.desc}: Удаление папок с бекапами и ' + \
                'их распакованными версиями'
        else:
            desc = ''

        # Выполняем удаление:
        rmpath((self.backup_path, self.extract_path), desc)

    def __enter__(self):
        return self()

    def __exit__(self, type, value, traceback):
        self.rmdirs()
        return