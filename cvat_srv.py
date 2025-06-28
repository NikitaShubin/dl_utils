import os
import numpy as np
import shutil
import json
import time
from PIL import Image
from cvat_sdk import make_client, core
from cvat_sdk.api_client import models
from getpass import getpass
from zipfile import ZipFile
from tqdm import tqdm

from cvat import (
    add_row2df, concat_dfs, ergonomic_draw_df_frame, cvat_backups2raw_tasks,
    cvat_backup_task_dir2task, get_related_files, df2annotations
)
from utils import (
    mkdirs, rmpath, mpmap, get_n_colors, unzip_dir, AnnotateIt, cv2_exts,
    cv2_vid_exts, get_file_list, json2obj, get_empty_dir, Zipper, obj2json,
    get_video_info, split_dir_name_ext
)


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
                    'Список классов должен содержать строки, словари '
                    'или `cvat_sdk.api_client.model.label.Label`. '
                    f'Получен "{type(label)}!"'
                )

        # Если передан словарь, то оставляем в нём только самое необходимое:
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
                 file=None,
                 labels=None,
                 annotation_path=None):
        '''
        Создаёт новую задачу без разметки.
        '''

        # Если не задано ничего, кроме имени:
        if (file, labels, annotation_path) == (None,) * 3:

            # Создаём пустую задачу:
            data, response = self.api_client.tasks_api.create(
                api_client.models.TaskWriteRequest(
                    name=name,
                    labels=[
                        api_client.models.PatchedLabelRequest(
                            name='Empty_label'
                        )
                    ]
                )
            )
            task_id = data.id

            # Возвращаем обёрнутый объект:
            return CVATSRVTask.from_id(self, task_id)

        # Если всё задано:
        else:
            # Доводим до ума входные параметры, если надо:
            labels = self._adapt_labels(labels)
            file = self._adapt_file(file)

            task_spec = {'name': name,
                         'labels': labels}

            task = self.tasks.create_from_data(
                spec=task_spec,
                resource_type=core.proxies.tasks.ResourceType.LOCAL,
                resources=file,
                annotation_path=annotation_path,
                annotation_format='CVAT 1.1',
            )

            # Возвращаем обёрнутый объект:
            return CVATSRVTask(self, task)

    def new_project(self, name, labels=None):
        '''
        Создаёт пустой датасет.
        '''

        # Если метки не указаны:
        if labels is None:
            data, response = api_client.projects_api.create(
                api_client.models.ProjectWriteRequest(name=name)
            )
            project_id = data.id

            # Возвращаем обёрнутый объект:
            return CVATSRVProject.from_id(self, project_id)

        # Если метки указаны:
        else:

            # Добавляет классам цвета, если надо:
            labels = self._adapt_labels(labels)

            # Создаём проект:
            proj_spec = {'name': name,
                         'labels': labels}
            project = self.projects.create_from_dataset(proj_spec)

            # Возвращаем обёрнутрый объект:
            return CVATSRVProject(self, project)

    def _send_backup(self, backup_file: str, backup_type: str) -> str:
        '''
        Отправляет бекап на CVAT-сервер.
        '''
        # Создаем объект запроса:
        api_client = self.obj.api_client
        if backup_type == 'task':
            file_request = models.TaskFileRequest
            restore_from_backup = api_client.tasks_api.create_backup
            backup_request_name = 'task_file_request'
        elif backup_type == 'project':
            file_request = models.ProjectFileRequest
            restore_from_backup = api_client.projects_api.create_backup
            backup_request_name = 'backup_write_request'
        else:
            raise ValueError('Неверное значение "backup_type": '
                             f'{backup_type}')

        # Формируем запрос на выгрузку бекапа:
        with open(backup_file, 'rb') as f:

            # Создаем объект запроса:
            backup_request = file_request(f)

            # Отправляем запрос:
            data, response = restore_from_backup(
                **{backup_request_name: backup_request},
                _content_type='multipart/form-data'  # Без этого не работает!
            )

        # Выясняем rq_id, по которому отследить процесс закачки:
        response_body = response.data.decode('utf-8')
        response_data = json.loads(response_body)
        return response_data['rq_id']

    def _rq_id2id(self, rq_id: str, timeout=300, interval=5) -> int | None:
        '''
        Ожидает успешного выполнения CVAT-сервером запроса по его rq_id и
        возвращает id полученного объекта.

        :param rq_id: ID запроса на импорт
        :param timeout: Максимальное время ожидания в секундах
        :param interval: Интервал проверки статуса в секундах
        :return: ID созданного объекта
        '''
        # Повторяем запросы пока не кончится время:
        start_time = time.time()
        while (time.time() - start_time) < timeout:

            # Получаем статус запроса:
            data, _ = self.api_client.requests_api.retrieve(rq_id)
            status = data['status'].to_str()

            # Если процесс завершён:
            if status == 'finished':
                return data['result_id']

            # Если процесс прерван:
            elif status == "failed":
                error_msg = data.get('message', '') or data.get('error', '')
                raise RuntimeError(f"Ошибка импорта: {error_msg}")

            time.sleep(interval)  # Ждем перед следующей проверкой

        raise TimeoutError(f'Превышено время ожидания ({timeout} секунд)')

    def restore_task(self, backup_file, return_id_only=False):
        '''
        Восстанавливает задачу из бекапа.
        '''
        # Запуск передачи бекапа на сервер:
        req_id = self._send_backup(backup_file, 'task')

        if hasattr(self.obj.api_client, 'requests_api'):
            # Ожидание результата:
            task_id = self._rq_id2id(req_id)

            # Возвращаем новую задачу, или только её ID:
            if return_id_only:
                return task_id
            else:
                return CVATSRVTask.from_id(self, task_id)

    def restore_project(self, backup_file, return_id_only=False):
        '''
        Восстанавливает проект из бекапа.
        '''
        # Запуск передачи бекапа на сервер:
        req_id = self._send_backup(backup_file, 'project')

        if hasattr(self.obj.api_client, 'requests_api'):
            # Ожидание результата:
            proj_id = self._rq_id2id(req_id)

            # Возвращаем новый проект, или только её ID:
            if return_id_only:
                return proj_id
            else:
                return CVATSRVProject.from_id(self, proj_id)

    @staticmethod
    def parse_url(url):
        '''
        Извлекает информацию из URL-строки страницы в CVAT
        '''
        url_parts = url.split('/')

        parsed_url = {}

        # Из строки извлекаются номера датасетов, задач и подзадач:
        for class_name in ['project', 'task', 'job']:
            class_name_s = class_name + 's'
            if class_name_s in url_parts:
                class_ind = url_parts.index(class_name_s)
                url_parts.pop(class_ind)
                class_id = url_parts.pop(class_ind)
                parsed_url[class_name + '_id'] = int(class_id.split('?')[0])

        # Всё, что осталось, счтитается адресом самого сервера:
        if len(url_parts) == 3:  # 'http://*'.split('/') -> ['http:', '', '*']
            if url_parts[1] != '':
                raise ValueError(
                    f'Не удалось распарсить остаток URL: {url_parts}'
                )
        elif len(url_parts) != 1:
            raise ValueError(
                f'Не удалось распарсить остаток URL: {url_parts}'
            )
        parsed_url['host'] = '/'.join(url_parts)

        return parsed_url

    def from_url(self, url):
        '''
        Возвращает объект найденный по url.
        '''
        parsed_url = self.parse_url(url)

        if 'job_id' in parsed_url:
            return CVATSRVJob.from_id(self, parsed_url['job_id'])
        elif 'task_id' in parsed_url:
            return CVATSRVTask.from_id(self, parsed_url['task_id'])
        elif 'project_id' in parsed_url:
            return CVATSRVProject.from_id(self, parsed_url['project_id'])
        else:
            raise ValueError(f'Некорректный URL: "{url}", {parsed_url}')

    # Возвращает список всех задач сервера вне зависимости от принадлежности
    # проекту:
    def all_tasks(self):
        tasks = []
        for task in self.obj.tasks.list():
            tasks.append(CVATSRVTask(self, task))
        return tasks

    # Возвращает список всех подзадач вне зависимости от принадлежности
    # проекту:
    def all_jobs(self):
        jobs = []
        for job in self.obj.jobs.list():
            jobs.append(CVATSRVJob(self, job))
        return jobs


def get_name(obj):
    '''
    Возвращает имя заданного объекта.
    Используется для сортировки списка объектов по их именам.
    '''
    return obj.name


class CVATSRVBase:
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

        if self._hier_lvl in {1, 2}:
            self.editor = None

    @classmethod
    def from_id(cls, client, id):
        '''
        Должен возвращать экземпляр класса объекта по его id.
        '''
        raise NotImplementedError('Метод должен быть переопределён!')

    def get_api(self, lvl=None):
        '''
        Возвращает API нужного уровня.
        Если уровень не указан - берётся уровень текущего объекта.
        '''
        # Берём текущий уровень, если он не указан:
        if lvl is None:
            lvl = self._hier_lvl

        # Определяем имя нужного атрибута:
        attr = ['server_api', 'projects_api', 'tasks_api', 'jobs_api'][lvl]

        # Возвращаем нужный API:
        return getattr(self.client.obj.api_client, attr)

    def name2id(self, name):
        '''
        Возвращает id объекта-потомка по его имени.
        '''
        for key, val in self.items():
            if key == name:
                return val.id
        return None

    @staticmethod
    def _backup(client_kwargs, child_class, child_id, path):
        '''
        Создаёт бекап, предварительно пересоздавая клиент.
        Нужно для использования mpmap, работающей только с сериализуемыми
        объектами.
        '''
        client = Client(**client_kwargs)
        while True:
            try:
                child = child_class.from_id(client, child_id)
                return child.backup(path)
            except Exception as e:
                print(e)
                print(f'Перезапуск бекапа "{path}"')

    @staticmethod
    def _restore(client_kwargs, parent_class, parent_id, path):
        '''
        Воссоздаёт объект из бекапа, предварительно пересоздавая клиент.
        Нужно для использования mpmap, работающей только с сериализуемыми
        объектами.
        '''
        client = Client(**client_kwargs)
        while True:
            try:
                if parent_id is None:
                    return client.restore_project(path, return_id_only=True)
                else:
                    parent = parent_class.from_id(client, parent_id)
                    return parent.restore(path, return_id_only=True)
            except Exception as e:
                print(e)
                print(f'Перезапуск восстановления "{path}"')

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

    def backup_all(self,
                   path='./',
                   desc='Загрузка бекапов',
                   **mpmap_kwargs):
        '''
        Выполняет бекап всех элементов, входящих в объект, по отдельности.
        '''
        # Присовокупляем описание к остальным параметрам mpmap:
        mpmap_kwargs = mpmap_kwargs | {'desc': desc}

        # Формируем список объектов, подлежащих бекапу:
        objs = self.values()

        # Формируем список ID всех сущностей:
        ids = [obj.id for obj in objs]

        # Формируем список имён для бекапов из ID соответствующих сущностей:
        paths = [os.path.join(path, f'{id}.zip') for id in ids]

        # Бекапим все сущности:
        client_kwargs = [self.client.kwargs4init()] * len(objs)
        child_classes = [self.child_class] * len(objs)
        return mpmap(self._backup, client_kwargs, child_classes,
                     ids, paths, **mpmap_kwargs)

    def restore(self, backup_file, return_id_only=False):
        '''
        Восстанавливает задачу или проект по его бекапу.
        '''
        # Если сейчас мы на уровне сервера, то восстанавливаем проект:
        if self._hier_lvl == 0:
            return self.client.restore_project(
                backup_file, return_id_only=return_id_only
            )

        # Если сейчас мы на уровне проекта, то восстанавливаем задачу:
        elif self._hier_lvl == 1:
            task = self.client.restore_task(backup_file)

            # Привязываем задачу к текущему датасету:
            task.update({'project_id': self.id})

            # Возвращаем либо сам восстановленный объект, либо только его ID:
            return task.id if return_id_only else task

        else:
            raise NotImplementedError(
                'Восстанавливать можно лишь проекты и задачи!'
            )

    def restore_all(self,
                    paths: list | tuple | set,
                    desc='Восстановление из бекапов',
                    **mpmap_kwargs):
        '''
        Восстанавливает элементы текущего объекта из списка бекапов.
        '''
        # Присовокупляем описание к остальным параметрам mpmap:
        mpmap_kwargs = mpmap_kwargs | {'desc': desc}

        # Восстанавливаем все сущности:
        client_kwargs = [self.client.kwargs4init()] * len(paths)
        parent_class = self._hier_lvl2class.get(self._hier_lvl)
        parent_classes = [parent_class] * len(paths)
        parent_ids = [self.id if self._hier_lvl else None] * len(paths)
        return mpmap(self._restore, client_kwargs, parent_classes,
                     parent_ids, paths, **mpmap_kwargs)

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

    def delete(self):
        '''
        Удаляет текущий объект на сервере!
        '''
        if self._hier_lvl:

            # Запускаем удаление и получаем ответ:
            _, response = self.get_api().destroy(self.id)

            # Удаление прошло успешно, если статус = 204:
            if response.status == 204:
                return True
            else:
                return response.status

        else:
            raise Exception('Невозможно удалить сам сервер!')

    def __getattr__(self, attr):
        '''
        Проброс атрибутов вложенных объекта наружу.
        '''
        # Сначала ищем атрибуты в низкоуровневом представителе объекта в CVAT:
        if hasattr(self.obj, attr):
            return getattr(self.obj, attr)

        # Затем смотрим в локальной копии данных, если она создана:
        elif self._hier_lvl in {1, 2} and hasattr(self.editor, attr):
            return getattr(self.editor, attr)
        # Создать её можно методом pull()

        '''
        else:
            raise NotImplementedError('')
        '''

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

    def from_url(self, url):
        '''
        Возвращает объект, соответствующий заданному URL:
        '''
        return self.client.from_url(url)

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

    # Возвращает экземпляр класса, дочерего к CVATBase, позволяющего
    # редактировать бекапы и отправлять изменённую версию обратно на сервер
    # (аргументы соответствуют аргументам CVATBase):
    def pull(self, *args, **kwargs):

        # Если локальный представитель ещё не создан:
        if self.editor is None:

            # Если мы на урове проекта:
            if self._hier_lvl == 1:
                self.editor = CVATProject(self, *args, **kwargs)

            # Если мы на урове задачи:
            elif self._hier_lvl == 2:
                self.editor = CVATTask(self, *args, **kwargs)

            else:
                raise NotImplementedError('Редактировть можно только '
                                          'проекты и задачи')

        return self.editor

    # Деструктор класса:
    def __del__(self):

        # Явно удаляем редактор, т.к. он имеет кучу временных файлов,
        # требующих удаления:
        del self.editor
        self.editor = None


class CVATSRVJob(CVATSRVBase):
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

    def __len__(self):
        return len(self.obj.get_frames_info())


class CVATSRVTask(CVATSRVBase):
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


class CVATSRVProject(CVATSRVBase):
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
            raise KeyError(f'Задача "{name}" уже существует '
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


class CVATSRV(CVATSRVBase):
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


#########################################################################
# Объекты, инкапсулирующие значительную часть функциональности cvat.py: #
#########################################################################


class CVATBase:
    '''
    Базовый класс объектов редактирования данных в CVAT.
    '''

    def __init__(self,
                 cvat_srv_obj: CVATSRVBase | None = None,
                 zipped_backup: str | None = None,
                 unzipped_backup: str | None = None,
                 parted: bool = True,
                 verbose: bool = True):
        '''
        cvat_srv_obj    - объект-представитель задачи или проекта из CVAТ,
                          через который осуществляется доступ к серверу.
        zipped_backup   - путь до zip-архива, содержащего бекап CVAT-объекта.
        unzipped_backup - путь до папки, содержащей распакованную версию
                          zipped_backup.
        parted          - флаг экспорта/импорта бекапа по частям (работает
                          только для проектов, которые можро разрезать на
                          задачи).
        verbose         - флаг вывода информации о статусе процессов.

        Исходная логика такова:
            1) посредством cvat_srv_obj закачиваем бекап в файл
               zipped_backup;
            2) распаковываем архив zipped_backup в папку unzipped_backup;
            3) парсим (загружаем в оперативную память) всю разметку из папки
               unzipped_backup.

        Детали исходной логики и исценарии отхода от неё:
                1) Если не задана unzipped_backup, то создаём в этом качестве
            временную папку.
                2) Если не задан zipped_backup, создаём временную папку и
            задаём для него в ней файл bu.zip.
                3) Если задана только unzipped_backup, то рассчитываем, что в
            ней уже есть все необходимые данные и сразу парсим их.
                4) Если задан только zipped_backup, то выполняем (1) и парсим
            результат.
                5) Если unzipped_backup уже не пуста, но так же существует
            zipped_backup или задан cvat_srv_obj - возвращаем ошибку, чтобы не
            перезаписывать данные в unzipped_backup, которые могут
            представлять какую-то важность.
                5) Если задан только cvat_srv_obj, то выполняем (1, 2),
            закачку бекапа, распаковываем его и парсим результат.
        '''
        # Доопределяет переменные:
        self.__prepare_variables(cvat_srv_obj,
                                 zipped_backup,
                                 unzipped_backup,
                                 parted,
                                 verbose)

        # Парсим распакованный бекап:
        self.__prepare_resources()

    # Приводит переменные к рабочему состоянию:
    def __prepare_variables(self,
                            cvat_srv_obj,
                            zipped_backup,
                            unzipped_backup,
                            parted,
                            verbose):
        '''
        if (cvat_srv_obj, zipped_backup, unzipped_backup) == (None,) * 3:
            raise ValueError('Хотя бы один параметр должен быть задан!')
        '''
        # Инициируем список файлов, подлежащих удалению перед закрытием
        # объекта:
        self.paths2remove = set()

        # Если cvat_srv_obj не является представителем проекта, то
        # синхронизация по частям невозможноа:
        if not isinstance(cvat_srv_obj, CVATSRVProject):
            parted = False

        # Доопределяем путь к архиву бекапа, если он не задан:
        if zipped_backup is None:
            # Указываем его место во временной папке, которая потом будет
            # удалена:
            zipped_backup = get_empty_dir()
            self.paths2remove.add(zipped_backup)
            zipped_backup = os.path.join(zipped_backup, 'bu.zip')
        else:
            # Даже если архив задан, он подлежит удалению перед деструкцией
            # экземлпяра класса:
            self.paths2remove.add(zipped_backup)

        # Создаём папку для распакованных данных, если она не задана,
        # или не существует:
        if unzipped_backup is None or not os.path.isdir(unzipped_backup):
            unzipped_backup = get_empty_dir(unzipped_backup, False)
            self.paths2remove.add(unzipped_backup)

        # Фиксируем объекты:
        self.cvat_srv_obj = cvat_srv_obj
        self.zipped_backup = zipped_backup
        self.unzipped_backup = unzipped_backup
        self.parted = parted
        self.verbose = verbose

    # Выполнчет сжатие бекапа(ов):
    def compress(self):

        # Определяем текстовое сопровождение процесса:
        desc = 'Сжатие бекапа' if self.verbose else ''

        # Если нужно архивировать по частям:
        if self.parted:

            # Формируем аргументы для параллельной архивации:
            unzippeds = []
            zippeds = []
            num_tasks = 0
            # Перебираем все поддирректории в папке с распакованным бекапом:
            for dir_name in os.listdir(self.unzipped_backup):

                # Файлы пропускаем:
                task_dir = os.path.join(self.unzipped_backup, dir_name)
                if not os.path.isdir(task_dir):
                    continue

                num_tasks += 1

                # Пополняем списки путей:
                unzippeds.append(os.path.join(task_dir, '*'))
                zippeds.append(
                    os.path.join(self.zipped_backup, f'{dir_name}.zip')
                )

            # Выполняем параллельное сжатие:
            mpmap(Zipper.compress, unzippeds, zippeds,
                  [False] * num_tasks, [True] * num_tasks,
                  desc=desc)

        # Если вносится весь архив целиком:
        else:
            with AnnotateIt(desc):
                Zipper.compress(
                    unzipped=os.path.join(self.unzipped_backup, '*'),
                    zipped=self.zipped_backup,
                    remove_source=False,
                    rewrite_target=True,
                    desc=desc
                )

    # Выполнчет распаковку бекапа(ов):
    def extract(self):

        # Определяем текстовое сопровождение процесса:
        desc = 'Извлечение бекапа' if self.verbose else ''

        # Если нужно распаковывать по частям:
        if self.parted:

            # Формируем аргументы для параллельной архивации:
            zippeds = get_file_list(self.zipped_backup, '.zip', False)
            unzippeds = []
            num_tasks = len(zippeds)

            for zipped in zippeds:
                _, name, _ = split_dir_name_ext(zipped)
                unzippeds.append(
                    os.path.join(self.unzipped_backup, name, '*')
                )

            # Выполняем параллельную распаковку:
            mpmap(Zipper.extract, zippeds, unzippeds,
                  [True] * num_tasks, [False] * num_tasks,
                  desc=desc)

        # Если вносится весь архив целиком:
        else:
            with AnnotateIt(desc):
                Zipper.extract(
                    unzipped=os.path.join(self.unzipped_backup, '*'),
                    zipped=self.zipped_backup,
                    remove_source=True,
                    rewrite_target=False,
                    desc=desc
                )

    # Приводит все файлы к рабочему состоянию:
    def __prepare_resources(self):

        # Если unzipped_backup уже не пуста:
        if os.listdir(self.unzipped_backup):

            # Если при этом архив бекапа существует или хотя бы задан
            # объект взаимодействия с сервером:
            if os.path.isfile(self.zipped_backup) or \
                    self.cvat_srv_obj is not None:

                # Значит, в папке может быть что-то важное, так что
                # очищать её не будем, а вернём ошибку.
                raise ValueError(f'Папка "{self.unzipped_backup}" не пуста!')

        # Если unzipped_backup пуста:
        else:

            # Если при этом объект взаимодействия с сервером задан, то
            # сразу создаём свежий бекап:
            if self.cvat_srv_obj is not None:
                rmpath(self.zipped_backup)  # Уже существующий архив удаляем

                # Качаем бекап частями или целиком:
                desc = 'Загрузка бекапа' if self.verbose else ''
                if self.parted:
                    self.cvat_srv_obj.backup_all(self.zipped_backup,
                                                 desc=desc,
                                                 num_procs=1)
                else:
                    with AnnotateIt(desc):
                        self.cvat_srv_obj.backup(self.zipped_backup)

            # Распаковываем бекап(ы):
            self.extract()
            # Архив(ы) после этого удаляе(ю)тся.

            # Если задачи качались по отдельности, то project.json нужно
            # создать отдельно:
            if self.parted:

                # Составляем список меток:
                keys = {'name', 'color', 'attributes', 'type', 'sublabels'}
                labels = []
                for label in self.cvat_srv_obj.get_labels():
                    label = label.to_dict()
                    labels.append({k: label[k] for k in label.keys() & keys})

                # Собираем общее описание:
                info = {'name': self.cvat_srv_obj.name,
                        'bug_tracker': self.cvat_srv_obj.bug_tracker,
                        'status': self.cvat_srv_obj.status,
                        'labels': labels,
                        'version': '1.0'}

                # Пишем всю инфу в файл:
                json_file = os.path.join(self.unzipped_backup, 'project.json')
                obj2json(info, file=json_file)

        # Должен создавать поля data и info:
        self._parse_unzipped_backup()

    # Парсит распакованный бекап:
    def _parse_unzipped_backup(self):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Создаёт новый экземпляр класса из фото/видео/дирректории с данными:
    @classmethod
    def from_raw_data(cls,
                      data_path: str | list[str] | tuple[str],
                      cvat_srv_obj: str | None = None,
                      unzipped_backup: str | None = None,
                      include_as_is: bool = False):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Пишет текущее состояние разметки в папку с распакованным бекапом,
    # а при необходимости собирает архив и отправляет его на CVAT-сервер.
    def push(self, local_only=False):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Размер объекта = размеру поля data:
    def __len__(self):
        return len(self.data)

    # Удаляет все файлы и папки, подлежащие удалению:
    def rm_all_paths2remove(self):
        for path in self.paths2remove:

            if self.verbose:
                print(f'"{path}" удалён!')

            rmpath(path)

    # Перед закрытием объекта надо удалять все файлы/папки из списка:
    def __del__(self):
        print('Base Del')
        self.rm_all_paths2remove()


class CVATProject(CVATBase):
    '''
    Интерфейс для работы с CVAT-датасетами.
    '''

    # Парсит распакованный бекап:
    def _parse_unzipped_backup(self):

        # Список задач:
        self.data = []

        # Список файлов в распакованном датасете:
        base_names = os.listdir(self.unzipped_backup)

        # Загружаем основную инфу о датасете:
        if 'project.json' not in base_names:
            raise Exception('"project.json" не найден!')
        self.info = json2obj(
            os.path.join(self.unzipped_backup, 'project.json')
        )
        base_names.remove('project.json')  # Выкидываем файл из списка

        # Читаем гайд датасета, если он есть:
        guide_name = 'annotation_guide.md'
        if guide_name in base_names:
            guide_file = os.path.join(self.unzipped_backup, guide_name)
            with open(guide_file, 'r') as f:
                self.guide = f.read()
            base_names.remove(guide_name)  # Выкидываем файл из списка
        else:
            self.guide = ''

        # Перебираем все папки в корне:
        for base_name in tqdm(base_names,
                              desc='Чтение бекапа',
                              disable=not self.verbose):

            # Доопределяем путь до файла:
            subdir = os.path.join(self.unzipped_backup, base_name)

            # Если это дирректория с задачей, то парсим её:
            if os.path.isdir(subdir):
                self.data.append(CVATTask(unzipped_backup=subdir))

            # Если это файл:
            else:
                raise FileExistsError(f'Неожиданный файл "{subdir}"!')

    # Пишет текущее состояние разметки в папку с распакованным бекапом,
    # а при необходимости собирает архив и отправляет его на CVAT-сервер.
    def push(self, local_only=False):

        # Вызываем синхронизацю каждого из подобъектов:
        for task_data in tqdm(self.data,
                              desc='Обновление локальной копии',
                              disable=not self.verbose):
            task_data.push(True)

        # Если надо отправлять результат на CVAT-сервер:
        if not local_only:
            if self.cvat_srv_obj is None:
                raise Exception('Не задан cvat_srv_obj. '
                                'Отправка на CVAT-сервер невозможна!')

            # Формируем архив(ы):
            self.compress()

            # Отправляем архив(ы) на сервер
            desc = 'Выгрузка бекапа' if self.verbose else ''
            if self.parted:  # Если отправлять надо по частям

                # Составляем список архивов:
                paths = get_file_list(self.zipped_backup, '.zip')

                # Отправляем все архивы на сервер:
                self.cvat_srv_obj.restore_all(paths, desc=desc)

            else:  # Если отправлять надо один архив целиком
                with AnnotateIt(desc):

                    # Переходим в CVAT на один уровень выше:
                    parent = self.cvat_srv_obj.parent()

                    # Выполняем Выгрузку бекапа и заменяем интерфейсный
                    # объект:
                    self.cvat_srv_obj = parent.restore(self.zipped_backup)

            # Удаляем все архивы после успешной отправки их на сервер:
            for zip_file in get_file_list(self.zipped_backup):
                rmpath(zip_file)


class CVATTask(CVATBase):
    '''
    Интерфейс для работы с CVAT-задачами.
    '''

    # Парсит распакованный бекап:
    def _parse_unzipped_backup(self):

        # Извлекаем список подзадач и метаданные задачи:
        task, self.info = cvat_backup_task_dir2task(self.unzipped_backup,
                                                    True)

        # Создаём для каждой подзадачи экземпляр класса CVATJob:
        self.data = mpmap(CVATJob.from_subtask, task, num_procs=0)

    @staticmethod
    def __prepare_single_raw_file(data_path: str,
                                  task_data_dir: str | None = None,
                                  include_as_is: bool = False):
        '''
        Подготовка бекапа из единственного файла с неразмеченными данными.
        '''
        # Если это файл:
        if os.path.isfile(data_path):

            # Определяем тип файла:
            ext = os.path.splitext(data_path)[-1].lower()

            # Если это видео или фото - копируем его в подпапку:
            if ext in cv2_exts:
                shutil.copy(data_path, task_data_dir)

                # Определяем путь до него в новой папке:
                base_name = os.path.basename(data_path)
                file_list = [os.path.join(task_data_dir, base_name)]

            # Если это архив и его надо брать как есть, то распаковываем
            # его в подпапку:
            elif ext in {'.zip'} and include_as_is:
                Zipper.extract(data_path, task_data_dir)

                # Составляем список файлов:
                file_list = get_file_list(task_data_dir, cv2_exts, False)

            else:
                raise Exception(
                    f'Неподдерживаемый тип файла "{data_path}"'
                )

        # Если это дирректория:
        elif os.path.isdir(data_path):

            # Если её надо брать как есть, то копируем всё её
            # содержимое:
            if include_as_is:
                shutil.copytree(data_path, task_data_dir)

                # Составляем отсортированный список файлов в итоговой папке:
                file_list = get_file_list(task_data_dir, cv2_exts, False)
                file_list = sorted(file_list)

            # Если придётся парсить, ищем в корне подходящие по типу
            # файлы:
            else:
                # Составляем отсортированный по имени список всех файлов
                # подходящего типа:
                file_list = []
                num_videos = 0
                for file in sorted(os.listdir(data_path)):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in cv2_exts:
                        file_list.append(os.path.join(data_path, file))

                        # Проверка на неизбыточность данных:
                        if ext in cv2_vid_exts:
                            if num_videos > 0:
                                raise FileExistsError(
                                    'В задаче не может быть более '
                                    'одного видеофайла!'
                                )
                        else:
                            num_videos += 1

                # Копируем все найденные файлы подходящего типа в
                # подпапку:
                for file in file_list:
                    shutil.copy(file, task_data_dir)

        # Если data_path не файл и не папка:
        else:
            raise FileNotFoundError(f'Не найден "{data_path}"!')

        return file_list

    @staticmethod
    def __prepare_multiple_raw_files(data_path: list[str] | tuple[str],
                                     task_data_dir: str | None = None,
                                     include_as_is: bool = False):
        '''
        Подготовка бекапа из списка/кортежа неразмеченных данных.
        '''
        # Если требуется использование как есть:
        if include_as_is:

            # Множество дирректорий, в которых лежат файлы из списка:
            data_paths = set(map(os.path.dirname, data_path))

            # Все файлы из списка должны лежать в одной дирректории:
            if len(data_paths) != 1:
                raise ValueError(
                    'В режиме include_as_is все файлы в списке должны '
                    'лежать в одной дирректории!'
                )

            # Определяем место расположения всех файлов:
            source_dir = data_paths.pop()

            # Копируем все файлы из этой дирректории в целевую:
            shutil.copytree(source_dir, task_data_dir)

            # Составляем итоговый список файлов с учётом нового
            # расположения:
            file_list = [os.path.join(source_dir, basename)
                         for basename in map(os.path.basename, data_path)]

        # Если брать нужно только сами файлы:
        else:

            file_list = []
            for file in data_paths:

                # Копируем файл в целевую папку:
                shutil.copy(file, task_data_dir)

                # Пополняем итоговый список файлов:
                basename = os.path.basename(file)
                file_list.append(os.path.join(task_data_dir, basename))

        return file_list

    # Создаёт новый экземпляр класса из фото/видео/дирректории с данными:
    @classmethod
    def from_raw_data(cls,
                      data_path: str | list[str] | tuple[str],
                      include_as_is: bool = False,
                      *init_args, **init_kwargs):
        '''
        Флаг include_as_is используется если нужно включить контекст
        (см. https://docs.cvat.ai/docs/manual/advanced/contextual-images/).
        В этом случае передавать надо либо папку, в которой уже всё
        подготовлено, либо архив содержимого подобной папки (не включая саму
        папку!). Можно использовать и список файлов, но тогда они должны все
        лежать в одной папке, содержимое которой будет скопировано целиком.
        Во всех остальных случаях в корне папки будут выискиваться файлы
        подходящего типа, а архив игнорироваться (возвращаться ошибка).
        '''
        # Инициируем экземпляр класса, но пока без работы с файлами:
        cvat_task = cls(*init_args, **init_kwargs, _skip_parse=True)

        # Получаем целевую папку с распакованным бекапом:
        unzipped_backup = cvat_task.unzipped_backup

        # Подпапка для данных в дирректории с распакованным бекапом:
        task_data_dir = os.path.join(unzipped_backup, 'data')
        mkdirs(task_data_dir)  # Создаём подпапку для данных

        # Копируем данные в целевую папку и получаем список файлов в
        # итоговом пути:
        if isinstance(data_path, str):  # Если передан всего один файл
            prepare_raw_data = cls.__prepare_single_raw_file
        else:                           # Если файлов несколько
            prepare_raw_data = cls.__prepare_multiple_raw_files
        file_list = prepare_raw_data(data_path,
                                     task_data_dir,
                                     include_as_is)

        # Иницируем пустую разметку:
        data = cls._init_annotations(file_list)

        # Пишем эту разметку в папку с бекапом:
        cls._write_annotations2unzipped_backup(data,
                                               task_data_dir,
                                               unzipped_backup,
                                               info)

        # Выполняем работу с файлами, которую отложили в начале:
        cvat_task.__prepare_resources()

        return cvat_task

    # Создаёт список элементов типа CVATJob для неразмеченных данных:
    @classmethod
    def _init_annotations(cls, file_list):
        return [CVATJob.from_raw_data(file_list)]

    # Возвращает пути до распакованной задачи, папки с её данными и первый
    # из файлов:
    @staticmethod
    def _jobs2paths(jobs):

        # Получаем общий список файлов для всех подзадач:
        files = [job.file for job in jobs]
        file = files[0]
        if len(files) > 1:
            for file_ in files[1:]:
                if file_ != file:
                    raise Exception('Подзадачи имеют несовпадающий'
                                    'список файлов!')

        # Берём первый файл для образца:
        first_file = file if isinstance(file, str) else file[0]

        # Пути до данных и всей папки задачи:
        task_data_dir = os.path.dirname(first_file)
        task_dir = os.path.dirname(task_data_dir)

        return task_dir, task_data_dir, file, first_file

    # Пишет указанную разметку в папку с распакованным бекапом:
    @classmethod
    def _write_annotations2unzipped_backup(cls,
                                           jobs,
                                           task_data_dir,
                                           unzipped_backup,
                                           info):
        # Определяем гланвые пути:
        task_dir, task_data_dir, file, first_file = cls._jobs2paths(jobs)

        # Пишем файлы описания сырых данных:
        cls._write_manifest(jobs, task_dir, task_data_dir,
                            file, first_file)  # manifest.json
        cls._write_index(jobs, task_data_dir)   # index.json

        # Пишем файлы описания задачи и разметки:
        cls._write_info(jobs, info, task_data_dir)  # task.json
        cls._write_annotanions(jobs, task_dir)      # annotations.json

    # Пишет файл manifest.jsonl:
    @staticmethod
    def _write_manifest(jobs, task_dir, task_data_dir, file, first_file):

        # Формируем содержимое файла:
        manifest = [{'version': '1.1'}]
        if os.path.splitext(first_file)[1] in cv2_vid_exts:  # Если это видео

            # Указываем тип "видео":
            manifest.append({'type': 'video'})

            # Определяем размеры и число кадров в видео:
            file_info = get_video_info(first_file)

            # Прописываем свойства видео:
            manifest.append({
                'properties': {
                    'name': os.path.basename(first_file),
                    'resolution': [
                        file_info['width'],
                        file_info['height']
                    ],
                    'length': file_info['length']
                }
            })

            # Добавляем хвост:
            manifest.append({'number': 0, 'pts': 0})

        else:  # Если это изображения

            # Указываем тип "изображения":
            manifest.append({'type': 'images'})

            # Формируем список описаний файлов:
            manifest += get_related_files(file,
                                          images_only=False,
                                          as_manifest=True)

        # Определяем путь до нужного файла:
        manifest_file = os.path.join(task_data_dir, 'manifest.jsonl')

        # Пишем результат в файл и возвращаем путь до него:
        return obj2json(manifest, manifest_file)

    # Пишет файл index.json (на самом деле удаляет его, т.к. он не обязателен):
    @staticmethod
    def _write_index(jobs, task_data_dir):

        # Определяем путь до нужного файла:
        index_file = os.path.join(task_data_dir, 'index.json')

        # Удаляем этот файл, т.к. его существование не обязательно:
        return rmpath(index_file)

    # Пишет файл описания задачи task.json:
    @staticmethod
    def _write_info(jobs, info, task_data_dir):

        # Определяем путь до нужного файла:
        info_file = os.path.join(task_data_dir, 'task.json')

        # Пишем всю информацию о задаче в соответствующий файл:
        return obj2json(info, info_file)

    # Пишет файл разметки annotations.json:
    @staticmethod
    def _write_annotanions(jobs, task_dir):

        # Строим список разметок для каждой задачи:
        annotations = [job._build_annotations() for job in jobs]

        # Определяем путь до нужного файла:
        annotations_file = os.path.join(task_dir, 'annotations.json')

        # Пишем всю разметку в соответствующий файл:
        return obj2json(annotations, annotations_file)

    # Пишет текущее состояние разметки в папку с распакованным бекапом,
    # а при необходимости собирает архив и отправляет его на CVAT-сервер.
    def push(self, local_only=False):

        task_data_dir = os.path.join(self.unzipped_backup, 'data')

        # Обновляем содержимое файлов в распакованном датасете:
        self._write_annotations2unzipped_backup(self.data,
                                                task_data_dir,
                                                self.unzipped_backup,
                                                self.info)

        # Если нужно отправлять результат на CVAT-сервер:
        if not local_only:

            with AnnotateIt('Отправка задачи в CVAT' if self.verbose else ''):

                # Создаём архив(ы):
                self.compress()

                # Отправка архива на CVAT-сервер:
                self.cvat_srv_obj.parent().restore(self.zipped_backup)

    def __del__(self):
        print('Task Del')
        self.rm_all_paths2remove()


class CVATJob:
    '''
    Интерфейс для работы с CVAT-подзадачами.
    '''

    def __init__(self, df, file, true_frames, issues=None):
        self.df = df
        self.file = file
        self.true_frames = true_frames
        self.issues = issues

    @classmethod
    def from_subtask(cls, subtask, issues=None):
        df, file, true_frames = subtask
        return cls(df, file, true_frames, issues)

    def __len__(self):
        return len(self.true_frames)

    # Создаёт экземпляр класса с пустой разметкой:
    @classmethod
    def from_raw_data(cls, file: str | list[str] | tuple[str]):

        if isinstance(file, str):
            first_file = file

        else:
            first_file = file[0]

            # Если список из одного элемента - берём его вместо списка:
            if len(file) == 1:
                file = file[0]

        # Определяем общее число кадров:
        total_frames = VideoGenerator.get_file_total_frames(file)

        # Собираем экземпляр класса с пустой разметкой:
        return cls(None, file, {frame: frame for frame in range(total_frames)})

    # Строит словарь разметки формата annotations.json:
    def _build_annotations(self):
        return df2annotations(self.df)
