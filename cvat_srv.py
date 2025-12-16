import os
import numpy as np
import shutil
from PIL import Image
from cvat_sdk import make_client, core, api_client
from getpass import getpass
from zipfile import ZipFile
from tqdm import tqdm
from http.client import IncompleteRead

from cvat import (
    add_row2df, concat_dfs, ergonomic_draw_df_frame, cvat_backups2raw_tasks,
    cvat_backup_task_dir2task, get_related_files, df2annotations, new_df,
    interpolate_df, split_df_by_visibility, hide_skipped_objects_in_df,
    CVATLabels
)
from utils import (
    mkdirs, rmpath, mpmap, get_n_colors, unzip_dir, AnnotateIt, cv2_exts,
    cv2_vid_exts, get_file_list, json2obj, get_empty_dir, Zipper, obj2json,
    get_video_info, split_dir_name_ext, invert_index_vector
)
from video_utils import VideoGenerator


# Множество ключей, приемлемыз для описания конкретного класса:
cvat_label_keys = {'name', 'color', 'type', 'svg', 'attributes'}


class AccurateProgressReporter(core.progress.ProgressReporter):
    '''
    Прогрессбар для передачи файлов
    '''

    def __init__(self, desc=None):
        self.progress_bar = None
        self.desc = desc

    def start(self, total, **kwargs):
        self.progress_bar = tqdm(
            total=total,
            unit='B',
            unit_scale=True,
            desc=self.desc,
        )

    def advance(self, delta):
        if self.progress_bar:
            self.progress_bar.update(delta)

    def finish(self):
        if self.progress_bar:
            self.progress_bar.close()


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

    """
    def __getattr__(self, attr):
        '''
        Проброс атрибутов вложенного объекта наружу.
        '''
        return getattr(self.obj, attr)
    """

    @staticmethod
    def _adapt_labels(labels:
                      tuple[dict | str | api_client.model.label.Label] |
                      list[dict | str | api_client.model.label.Label] |
                      set[dict | str | api_client.model.label.Label]):
        '''
        Добавляет к списку классов цвета, если надо.
        '''
        # Если объект пуст - возвращаем пустой список:
        if len(labels) == 0:
            return []

        # Кортеж и множество переводим в список:
        labels = list(labels)

        # Если передан список строк - преобразуем элементы в словари:
        if isinstance(labels[0], str):
            labels_ = []
            for color, label in zip(labels, get_n_colors(len(labels))):
                labels_.append({'name': label,
                                'attributes': [],
                                'color': '#%02x%02x%02x' % color})
            labels = labels_

        # Если передан список CVAT-меток - преобразуем в словари:
        elif isinstance(labels[0], api_client.model.label.Label):
            labels = [label.to_dict() for label in labels]

        # Оставляем только нужные пары ключ-значение для каждого класса:
        labels = [
            {key: val for key, val in label.items() if key in cvat_label_keys}
            for label in labels
        ]

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
                raise FileNotFoundError(f'Не найден файл "{file_}"!')

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
            data, response = self.obj.api_client.tasks_api.create(
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

            task = self.obj.tasks.create_from_data(
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

            # Создаём датасет:
            proj_spec = {'name': name,
                         'labels': labels}
            project = self.obj.projects.create_from_dataset(proj_spec)

            # Возвращаем обёрнутрый объект:
            return CVATSRVProject(self, project)

    @staticmethod
    def _backup(client,
                path: str,
                type: str,
                id: int,
                desc: None | str = None):
        '''
        Выполняет бекап объекта.
        Поддерживает работу в mpmap, работающей только с сериализуемыми
        объектами.
        '''
        # Собираем клиент сами, если переданы его параметры:
        if isinstance(client, dict):
            client = Client(**client)

        # Определяем класс для объекта, который требуется бекапить:
        if type == 'task':
            CVATSRVClass = CVATSRVTask
        elif type == 'project':
            CVATSRVClass = CVATSRVProject
        else:
            raise ValueError(f'Недопустимое значение "type": {type}')

        # Получаем доступ к объекту:
        cvat_obj = CVATSRVClass.from_id(client, id)

        # Создаём папки до файла, если надо:
        mkdirs(os.path.abspath(os.path.dirname(path)))

        # Удаляем файл, если он уже существует:
        if os.path.isfile(path):
            rmpath(path)

        # Выполняем бекап:
        pbar = AccurateProgressReporter(desc) if desc else None
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cvat_obj.obj.download_backup(path, pbar=pbar)
                break
            except IncompleteRead:
                pass
            except Exception as e:
                print(e)
                print(f'Перезапуск бекапа "{path}"')
        else:
            print('Попытки исчерпаны!')
            raise e


        # Возвращаем путь до бекапа:
        return path

    @staticmethod
    def _restore(client,
                 path: str,
                 type: str,
                 parent_id: None | int = None,
                 return_id_only: bool = False,
                 desc: None | str = None):
        '''
        Воссоздаёт объект из бекапа.
        Поддерживает работу в mpmap, работающей только с сериализуемыми
        объектами.
        '''
        # Собираем клиент сами, если переданы его параметры:
        if isinstance(client, dict):
            client = Client(**client)

        # Определяем класс для объекта, который требуется бекапить:
        if type == 'task':
            create_from_backup = client.obj.tasks.create_from_backup
            CVATSRVClass = CVATSRVTask
        elif type == 'project':
            create_from_backup = client.obj.projects.create_from_backup
            CVATSRVClass = CVATSRVProject
        else:
            raise ValueError(f'Недопустимое значение "type": {type}')

        # Выполняем восстановление:
        pbar = AccurateProgressReporter(desc) if desc else None
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cvat_obj = create_from_backup(
                    filename=path,
                    pbar=pbar
                )
                break
            except Exception as e:
                print(e)
                print(f'Перезапуск восстановления "{path}"')
        else:
            print('Попытки исчерпаны!')
            raise e

        # Если указан ID датасета, к которому задачу надо присовокупить:
        if parent_id is not None:
            assert type == 'task'
            cvat_obj.update({'project_id': parent_id})

        # Возвращаем ID восстановленного объекта, или целый экземпляр класса:
        if return_id_only:
            return cvat_obj.id
        else:
            return CVATSRVClass.from_id(client, cvat_obj.id)
        return cvat_obj

    def restore_task(self, backup_file, proj_id=None,
                     return_id_only=False, desc=None):
        '''
        Восстанавливает задачу из бекапа.
        '''
        return self._restore(self, backup_file, 'task',
                             proj_id, return_id_only, desc)

    def restore_project(self, backup_file, return_id_only=False, desc=None):
        '''
        Восстанавливает датасет из бекапа.
        '''
        return self._restore(self, backup_file, 'project',
                             None, return_id_only, desc)

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
    # датасету:
    def all_tasks(self):
        tasks = []
        for task in self.obj.tasks.list():
            tasks.append(CVATSRVTask(self, task))
        return tasks

    # Возвращает список всех подзадач вне зависимости от принадлежности
    # датасету:
    def all_jobs(self):
        jobs = []
        for job in self.obj.jobs.list():
            jobs.append(CVATSRVJob(self, job))
        return jobs

    def get_guide(self, obj):
        '''
        Возвращает строку описания объекта (guide) в формате Markdown.
        '''
        if obj.obj.guide_id is None:
            return ''
        else:
            return self.obj.api_client.guides_api.retrieve(
                id=obj.obj.guide_id
            )[0].markdown

    def set_guide(self, obj, guide: str = ''):
        '''
        Переписывает строку описания объекта (guide) в формате Markdown.
        '''
        # Интерфейс возаимодействия со всеми гайдами:
        guides_api = self.obj.api_client.guides_api

        # Если строка не пуста:
        if guide:

            # Если передано не описание, а путь к файлу, то читаем его:
            if os.path.isfile(guide):
                with open(guide, 'r') as f:
                    guide = f.read()

            # Иницируем параметры запроса:
            kwargs = {'markdown': guide}

            # Получаем доступ к существующему guide или создаём новый:
            if obj.obj.guide_id is None:

                # Дополняем параметры запроса:
                if obj._hier_lvl == 1:
                    kwargs['project_id'] = obj.obj.id
                elif obj._hier_lvl == 2:
                    kwargs['task_id'] = obj.obj.id
                else:
                    raise ValueError('Описание можно добавлять только '
                                     'датасетам и задачам!')

                # Создаём сам запрос на создание описания:
                request = api_client.models.AnnotationGuideWriteRequest(
                    **kwargs
                )

                # Отправляем запрос на исполнение:
                guides_api.create(
                    annotation_guide_write_request=request
                )

            else:
                # Создаём сам запрос на редактирование описания:
                request = api_client.models.PatchedAnnotationGuideWriteRequest(
                    **kwargs
                )

                # Отправляем запрос на исполнение:
                guides_api.partial_update(
                    id=obj.obj.guide_id,
                    patched_annotation_guide_write_request=request
                )

        # Если строка пуста, а сам гайд не пустой - очищаем:
        elif obj.obj.guide_id is not None:
            guides_api.destroy(obj.obj.guide_id)

        # Сам объект лучше обновить после таких изменений:
        obj._update_obj()
        # Иначе его obj.obj.guide_id будет неактуален при следующем
        # использовании.

        # Возвращаем id Описания:
        return obj.obj.guide_id


def get_name(obj):
    '''
    Возвращает имя заданного объекта.
    Используется для сортировки списка объектов по их именам.
    '''
    return obj.name


class CVATSRVBase:
    '''
    Абстрактный класс для датасетов, задач, подзадач и прочих сущностей из
    cvat_sdk. Составляет основу древовидной структуру данных:
    Сервер > Датасеты > Задачи > Подзадачи ...

    Объект ведёт себя как словарь, ключами к которому являются имена
    подобъектов и самими подобъектами в качестве значений. Т.е. реализованы
    методы: __getitem__, keys, values, items.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = None
    # У каждого потомка будет свой номер.

    # В качестве имени объекта берётся:
    #    - его имя в cvat (для ов и задач);
    #    - его id (для подзадач);
    #    - адрес сервера (для сервера).
    @property
    def name(self):
        if self.obj is None:
            return self.client.host
        elif hasattr(self.obj, 'name'):
            return self.obj.name
        else:
            return self.obj.id

    def __init__(self, client, obj=None):

        # Собираем клиент сами, если переданы его параметры:
        if isinstance(client, dict):
            client = Client(**client)

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

        # Инициируем редактор (актуален только для датасетов и задач):
        self.editor = None

    def _update_obj(self, new_id: None | int = None):
        if new_id is None:
            new_id = self.obj.id
        if self._hier_lvl == 1:  # Если это датасет
            self.obj = self.client.obj.projects.retrieve(new_id)
        if self._hier_lvl == 2:  # Если это задача
            self.obj = self.client.obj.tasks.retrieve(new_id)
        if self._hier_lvl == 3:  # Если это подзадача
            self.obj = self.client.obj.jobs.retrieve(new_id)

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
                return val.obj.id
        return None

    def backup(self,
               path='./',
               name=None,
               desc=None,
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
            clients = [self.client.kwargs4init()] * len(name)
            types = ['task' if self._hier_lvl else 'project'] * len(name)
            ids = list(map(self.name2id, name))
            return mpmap(self.client._backup, clients, path, types, ids,
                         **mpmap_kwargs)

        # Если бекапить надо всю текущую сущность:
        elif name is None:

            # Если указан не путь до архива, считаем его папкой:
            if path.lower()[-4:] != '.zip':
                path = os.path.join(path, self.name + '.zip')
                # Дополняем путь именем сущности.

            # Качаем сам бекап и возвращаем путь до него:
            type = 'task' if self._hier_lvl == 2 else 'project'
            return self.client._backup(self.client,
                                       path,
                                       type,
                                       self.obj.id,
                                       desc=mpmap_kwargs['desc'])

        # Если передано имя лишь одной подсущности:
        else:
            return self[name].backup(path, **mpmap_kwargs)

    def restore(self, path, desc=None):
        '''
        Восстанавливает задачу или датасет по его бекапу.
        '''
        # Восстанавливать можно только находясь на уровне сервера или
        # датасета:
        assert self._hier_lvl < 2

        type = 'task' if self._hier_lvl else 'project'
        parent_id = self.obj.id if self._hier_lvl else None
        return self.client._restore(self.client, path, type,
                                    parent_id, None, desc)

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

        # Бекапим все сущности:
        ids = [obj.obj.id for obj in objs]  # Формируем список ID всех сущностей
        paths = [os.path.join(path, f'{id}.zip') for id in ids]
        clients = [self.client.kwargs4init()] * len(ids)
        types = ['task' if self._hier_lvl else 'project'] * len(ids)
        return mpmap(self.client._backup, clients, paths, types, ids,
                     **mpmap_kwargs)

    def restore_all(self,
                    path: list | tuple | set | str,
                    desc='Восстановление из бекапов',
                    return_id_only=False,
                    **mpmap_kwargs):
        '''
        Восстанавливает элементы текущего объекта из списка бекапов.
        '''
        # Присовокупляем описание к остальным параметрам mpmap:
        mpmap_kwargs = mpmap_kwargs | {'desc': desc}

        # Если передан лишь один путь, он должен быть дирректорией:
        if isinstance(path, str):
            assert os.path.isdir(path)
            path = get_file_list(path, '.zip', False)

        # Восстанавливаем все сущности:
        clients = [self.client.kwargs4init()] * len(path)
        types = ['task' if self._hier_lvl else 'project'] * len(path)
        parent_ids = [self.obj.id if self._hier_lvl else None] * len(path)
        return_id_onlys = [True] * len(path)
        ids = mpmap(self.client._restore, clients, path, types, parent_ids,
                    return_id_onlys, desc=mpmap_kwargs['desc'])

        if return_id_only:
            return ids
        else:
            return [self.child_class.from_id(self.client, id) for id in ids]

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
    def parent_id(self):
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
        if self.parent_id is None:
            return

        return self._hier_lvl2class[self._hier_lvl - 1].from_id(
            self.client, self.parent_id
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
            _, response = self.get_api().destroy(self.obj.id)

            # Удаление прошло успешно, если статус = 204:
            if response.status == 204:
                return True
            else:
                return response.status

        else:
            raise Exception('Невозможно удалить сам сервер!')

    def keys(self):
        '''
        Возвращает список имён входящих в объект подобъектов.
        '''
        return [get_name(subobject) for subobject in self.values()]

    def items(self):
        '''
        Возращает список кортежей из пар (имя_подобъекта, подобъект).
        '''
        return [(get_name(subobject), subobject)
                for subobject in self.values()]

    def __getitem__(self, key):
        '''
        Возвращает нужный подобъект по его имени.
        '''
        # Получаем все элементы с заданным имененм:
        if isinstance(key, str):
            items = [item for name, item in self.items() if name == key]
        elif isinstance(key, int):
            items = [item for name, item in enumerate(self.values())
                     if name == key]
        else:
            raise IndexError('Элемент ищется по имени (str) '
                             'или номеру (int)!')

        if len(items) > 1:
            raise IndexError('Найдено более одного элемента '
                             f'с именем "{key}"!')
        elif len(items) == 0:
            raise IndexError(f'Не найден элемент "{key}"!')
        else:
            return items.pop()

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

    # Возвращает экземпляр класса, дочерего к CVATBase, создающего локальную
    # копию бекапа исходного объекта, который можно редактировать и отправлять
    # изменённую версию обратно на сервер (аргументы соответствуют аргументам
    # CVATBase):
    def clone(self, *args, **kwargs):

        # Если локальный представитель ещё не создан:
        if self.editor is None:

            # Если мы на урове датасета:
            if self._hier_lvl == 1:
                self.editor = CVATProject(self, *args, **kwargs)

            # Если мы на урове задачи:
            elif self._hier_lvl == 2:
                self.editor = CVATTask(self, *args, **kwargs)

            else:
                raise NotImplementedError('Редактировть можно только '
                                          'датасеты и задачи')

        return self.editor

    # Чтение и запись описания датасетов и задач:
    @property
    def guide(self):
        return self.client.get_guide(self)

    @guide.setter
    def guide(self, guide: str = ''):
        return self.client.set_guide(self, guide)

    '''
    # Деструктор класса:
    def __del__(self):

        # Явно удаляем редактор, т.к. он имеет кучу временных файлов,
        # требующих удаления:
        del self.editor
        self.editor = None
    '''

    def export(self,
               file='ds.zip',
               format='YOLO 1.1',
               include_images=True,
               desc=None):
        '''
        Экспорт данных в заданный формат
        '''
        if self._hier_lvl not in {1, 2}:
            raise NotImplementedError('Экспорт возможен только для '
                                      'датасетов и задач!')
        pbar = AccurateProgressReporter(desc) if desc else None
        self.obj.export_dataset(format_name=format,
                                filename=file,
                                include_images=include_images,
                                pbar=pbar)
        return file


class CVATSRVJob(CVATSRVBase):
    '''
    Поздазача CVAT-сервера.
    '''
    # Уровень класса в иерархии:
    _hier_lvl = 3

    def values(self):
        raise NotImplementedError('У подзадачи нет составляющих!')

    @property
    def parent_id(self):
        return self.obj.task_id

    @property
    def name(self):
        '''
        У поздазач нет имён, поэтому используем их ID.
        '''
        return self.obj.id

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
        return cls(client, client.obj.jobs.retrieve(id))

    @property
    def url(self):
        '''
        Возвращает URL подзадачи.
        '''
        task_substr = f'/tasks/{self.obj.task_id}/jobs/'
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
    def parent_id(self):
        return self.obj.project_id

    @classmethod
    def from_id(cls, client, id):
        return cls(client, client.obj.tasks.retrieve(id))


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
        return cls(client, client.obj.projects.retrieve(id))

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

        # Создаём задачу под временным именем:
        task = self.client.new_task(tmp_name, file, self.labels(),
                                    annotation_path=annotation_file)

        # Привязываем задачу к текущему датасету:
        task.obj.update({'project_id': self.obj.id})

        # Даём задаче законченное имя:
        task.obj.update({'name': name})
        # В случае прерывания процесса создания задачи её можно будет легко
        # найти по временному имени, чтобы удалить.

        return task

    def project_json(self):
        '''
        Возвращает словарь, описывающий весь датасет.
        Бекапы датасетов содержат файл project.json с подобным описанием.
        '''
        return {'name': self.name,
                'bug_tracker': self.obj.bug_tracker,
                'status': self.obj.status,
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
                for project in self.client.obj.projects.list()]

    def parent(self):
        raise NotImplementedError('У сервера нет предков!')

    def __init__(self, host=None, username=None, password=None):
        host = host or input('Адрес CVAT-сервера:')
        username = username or input(f'{host} логин: ')
        password = password or getpass(f'{host} пароль: ')

        # self.name = host
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
                 verbose: bool = True,
                 take_data_from: str = 'auto'):
        '''
        cvat_srv_obj    - объект-представитель задачи или датасета из CVAТ,
                          через который осуществляется доступ к серверу.
        zipped_backup   - путь до zip-архива, содержащего бекап CVAT-объекта.
        unzipped_backup - путь до папки, содержащей распакованную версию
                          zipped_backup.
        parted          - флаг экспорта/импорта бекапа по частям (работает
                          только для датасетов, которые можро разрезать на
                          задачи).
        verbose         - флаг вывода информации о статусе процессов.
        take_data_from  - источник, который считается первичным.
                          Возможные значения:
                            'cvat_srv_obj',
                            'zipped_backup',
                            'unzipped_backup',
                            'auto'.

        Общий алгоритм подготовки файлов (при инициализации экземпляра класса)
        к работе следующий:
            1) посредством cvat_srv_obj закачиваем бекап в файл
               zipped_backup;
            2) распаковываем архив zipped_backup в папку unzipped_backup;
            3) парсим (загружаем в оперативную память) всю разметку из папки
               unzipped_backup.

        Если take_data_from = 'cvat_srv_obj', то весь алгоритм воспроизводится
        полностью. Если take_data_from = 'zipped_backup', то пропускается
        первый пункт. Если take_data_from = 'unzipped_backup', то выполняется
        только пункт 3.

        Если take_data_from = 'auto', то:
            Если папка unzipped_backup не пуста, то:
                take_data_from = 'unzipped_backup'
            Иначе, если zipped_backup существует:
                take_data_from = 'zipped_backup'
            Иначе, если cvat_srv_obj задан, то:
                take_data_from = 'cvat_srv_obj'
            Иначе:
                Ошибка!
        '''
        # Доопределяет переменные:
        self.__prepare_variables(cvat_srv_obj,
                                 zipped_backup,
                                 unzipped_backup,
                                 parted,
                                 verbose,
                                 take_data_from,
                                 self)
        # Это статический метод, поэтому self передаём явно.

        # Парсим распакованный бекап:
        self.__prepare_resources()

    @staticmethod
    def __redetermine_data_source(cvat_srv_obj,
                                  zipped_backup,
                                  unzipped_backup,
                                  verbose,
                                  take_data_from):
        '''
        Проверяет существование нужного ресурса и доопределяет take_data_from.
        Используется в __prepare_variables.
        '''
        # Проверяем существование нужного ресурса:
        if take_data_from == 'cvat_srv_obj':
            if cvat_srv_obj is None:
                raise ValueError('Параметр "cvat_srv_obj" не задан!')
        elif take_data_from == 'zipped_backup':
            if zipped_backup is None:
                raise ValueError('Параметр "zipped_backup" не задан!')
            elif not os.path.isfile(zipped_backup):
                raise FileNotFoundError(f'Файл "{zipped_backup}" не найден!')
        elif take_data_from == 'unzipped_backup':
            if unzipped_backup is None:
                raise ValueError('Параметр "unzipped_backup" не задан!')
            elif not os.path.isdir(unzipped_backup):
                raise FileNotFoundError(f'Путь "{unzipped_backup}" не найден!')
            elif len(os.listdir(unzipped_backup)) == 0:
                raise FileNotFoundError(f'Папка "{unzipped_backup}" пуста!')

        # Доопределяем источник автоматически:
        elif take_data_from == 'auto':
            if unzipped_backup is not None \
                    and os.path.isdir(unzipped_backup) \
                    and len(os.listdir(unzipped_backup)):
                take_data_from = 'unzipped_backup'
            elif zipped_backup is not None and os.path.isfile(zipped_backup):
                take_data_from = 'zipped_backup'
            elif cvat_srv_obj is not None:
                take_data_from = 'cvat_srv_obj'
            else:
                raise ValueError('Не задан ни один существующий ресурс!')

        else:
            raise ValueError('Недопустимое значение "take_data_from": '
                             f'{take_data_from}')

        return take_data_from

    # Приводит переменные к рабочему состоянию:
    @classmethod
    def __prepare_variables(cls,
                            cvat_srv_obj,
                            zipped_backup,
                            unzipped_backup,
                            parted,
                            verbose,
                            take_data_from,
                            self=None):

        # Проверяем существование нужного ресурса и доопределяем
        # take_data_from:
        take_data_from = cls.__redetermine_data_source(cvat_srv_obj,
                                                       zipped_backup,
                                                       unzipped_backup,
                                                       verbose,
                                                       take_data_from)

        # Инициируем список файлов, подлежащих удалению перед закрытием
        # объекта:
        paths2remove = set()

        # Если cvat_srv_obj не является представителем датасета, то
        # синхронизация по частям невозможноа:
        if not isinstance(cvat_srv_obj, CVATSRVProject):
            parted = False

        # Доопределяем путь к архиву бекапа, если он не задан:
        if zipped_backup is None:
            # Указываем его место во временной папке, которая потом будет
            # удалена:
            zipped_backup = get_empty_dir()
            paths2remove.add(zipped_backup)
            zipped_backup = os.path.join(zipped_backup, 'bu.zip')
        else:
            # Даже если архив задан, он подлежит удалению перед деструкцией
            # экземлпяра класса:
            paths2remove.add(zipped_backup)

        # Создаём папку для распакованных данных, если она не задана,
        # или не существует:
        if unzipped_backup is None or not os.path.isdir(unzipped_backup):
            unzipped_backup = get_empty_dir(unzipped_backup, False)
            paths2remove.add(unzipped_backup)

        # Возвращаем полученные параметры:
        if self is None:
            return (cvat_srv_obj, zipped_backup, unzipped_backup,
                    parted, verbose, paths2remove)
        else:
            self.cvat_srv_obj = cvat_srv_obj
            self.zipped_backup = zipped_backup
            self.unzipped_backup = unzipped_backup
            self.parted = parted
            self.verbose = verbose
            self.paths2remove = paths2remove

    # Выполнчет сжатие бекапа(ов):
    def _compress(self):

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
    def _extract(self):

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
            self._extract()
            # Архив(ы) после этого удаляе(ю)тся.

            # Если задачи качались по отдельности, то project.json нужно
            # создать отдельно:
            if self.parted:

                # Собираем общее описание:
                labels = CVATLabels(self.cvat_srv_obj.labels())
                info = {'name': self.cvat_srv_obj.name,
                        'bug_tracker': self.cvat_srv_obj.obj.bug_tracker,
                        'status': self.cvat_srv_obj.obj.status,
                        'labels': labels.to_dicts(),
                        'version': '1.0'}

                # Пишем всю инфу в файл:
                json_file = os.path.join(self.unzipped_backup, 'project.json')
                obj2json(info, file=json_file)

                # Отдельно сохраняем описание датасета:
                self._write_guide(self.cvat_srv_obj.guide,
                                  self.unzipped_backup)

        # Должен создавать поля data и info:
        self._parse_unzipped_backup()

    # Парсит распакованный бекап:
    def _parse_unzipped_backup(self):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Пишет текстовое описания (гайд) датасета annotation_guide.md:
    @staticmethod
    def _write_guide(guide, unzipped_backup):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Создаёт новый экземпляр класса из фото/видео/дирректории с данными:
    @classmethod
    def from_raw_data(cls,
                      data_path: str | list[str] | tuple[str],
                      include_as_is: bool = False,
                      # Дальше параметры, аналогичные __init__:
                      **init_kwargs):

        # Параметр take_data_from должен отсутствовать или быть равен
        # 'zipped_backup':
        if init_kwargs.get('take_data_from',
                           'unzipped_backup') != 'unzipped_backup':
            raise ValueError('Параметр "take_data_from" не используется '
                             'при создании бекапа с нуля!')

        # Реальное значение "take_data_from" будет недокументированным:
        init_kwargs['take_data_from'] = 'unzipped_backup'

        # Создаём папку для распакованных данных и доопределяем путь до неё,
        # если надо:
        unzipped_backup = init_kwargs.get('unzipped_backup', None)
        unzipped_backup = get_empty_dir(unzipped_backup)
        init_kwargs = dict(init_kwargs)  # Копирование переменной
        init_kwargs['unzipped_backup'] = unzipped_backup

        # Копируем файлы во папку с распакованным дадасетом:
        init_files_kwargs = cls._copy_files2unzipped_backup(data_path,
                                                            include_as_is,
                                                            unzipped_backup)
        # Возвращает параметры, используемые далее в
        # _init_files_in_unzipped_backup.

        # Создаём остальные файлы в папке с распакованным дадасетом:
        cls._init_files_in_unzipped_backup(unzipped_backup,
                                           **init_files_kwargs)
        # init_files_kwargs берётся из _copy_files2unzipped_backup

        # Создаём новый экземпляр класса для уже подготовленной дирректории:
        cvat_obj = cls(**init_kwargs)

        # Явно добавляем папку с распакованным бекапов в список на удаление:
        cvat_obj.paths2remove.add(unzipped_backup)

        return cvat_obj

    # Копирует файлы в папку с распакованным бекапом:
    @classmethod
    def _copy_files2unzipped_backup(cls,
                                    data_path,
                                    include_as_is,
                                    unzipped_backup):
        raise NotImplementedError('Метод должен быть переопределён!')

    # Создаёт метаданные (всё кроме самих фото/видео) в папке с распакованным
    # бекапом:
    @classmethod
    def _init_files_in_unzipped_backup(cls, unzipped_backup, **kwargs):
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
            rmpath(path)
            '''
            if self.verbose:
                print(f'"{path}" удалён!')
            '''

    # Перед закрытием объекта надо удалять все файлы/папки из списка:
    def __del__(self):
        self.rm_all_paths2remove()

    # Запускает метод с заданными параметрами для каждой подзадачи объекта:
    def apply2all_jobs(self, method, *args, **kwargs):

        # Для проектов перебираются задачи, а для задач - поздазачи:
        if hasattr(self, 'data'):
            for subobj in self.data:
                subobj.apply2all_jobs(method, *args, **kwargs)

        # Для подзадач вызывается сам метод:
        elif hasattr(self, method):
            self.method(*args, **kwargs)

        else:
            raise NotImplementedError('Метод {method} не объявлен!')

    def copy2(self, other_cvat_srv_obj, desc=None):
        '''
        Разворачивает текущий бекап в новом месте, не меняя старый.
        Возвращает представитель нового объекта на CVAT-сервере.
        '''

        # Создаём бекап:
        self.push(local_only=True)
        self._compress()

        # Доопределяем описание процесса:
        if desc is None:
            desc = 'Разворачивание бекапа' if self.verbose else None

        # Если копировать надо датасет, причём позадачно, а передан
        # представитель сервера, то создаём датасет:
        if self.parted and \
                self.cvat_srv_obj._hier_lvl == 1 and \
                other_cvat_srv_obj._hier_lvl == 0:
            other_cvat_srv_obj = other_cvat_srv_obj.client.new_project(
                self.info['name'],
                self.info['labels'].to_dicts()
            )

        # Восстанавливаем бекап в новом месте и возвращаем результат:
        if self.parted:

            # Список архивов бекапов:
            backups = get_file_list(self.zipped_backup, '.zip', False)

            # Восстанавливаем каждую задачу отдельно:
            for bu in tqdm(backups, desc=desc, disable=not desc):
                task = other_cvat_srv_obj.restore(bu, desc='')
            return task.parent()

        elif self.cvat_srv_obj._hier_lvl == other_cvat_srv_obj._hier_lvl + 1:
            return other_cvat_srv_obj.restore(self.zipped_backup, desc=desc)

        else:
            print(other_cvat_srv_obj._hier_lvl, self.cvat_srv_obj._hier_lvl)
            raise


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

        # Загружаем основную инфу о датасете из project.json:
        if 'project.json' not in base_names:
            raise Exception('"project.json" не найден!')
        self.info = json2obj(
            os.path.join(self.unzipped_backup, 'project.json')
        )
        self.info['labels'] = CVATLabels(self.info['labels'])  # Парсинг меток
        base_names.remove('project.json')  # Выкидываем файл из списка

        # Читаем гайд датасета annotation_guide.md, если он есть:
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

    # Пишет файл описания датасета project.json:
    @staticmethod
    def _write_info(info, unzipped_backup):

        # Путь до файла:
        info_file = os.path.join(unzipped_backup, 'project.json')

        # Создаём копию info и представляем метки в виде словарей:
        info = dict(info)
        info['labels'] = info['labels'].to_dicts()

        # Пишем содержимое файла:
        return obj2json(info, info_file)

    # Пишет текстовое описания (гайд) датасета annotation_guide.md:
    @staticmethod
    def _write_guide(guide, unzipped_backup):

        # Путь до файла:
        guide_file = os.path.join(unzipped_backup, 'annotation_guide.md')

        # Если гайд не пустой, пишем его в файл:
        if guide:
            with open(guide_file, 'w') as f:
                f.write(guide)

        # Если гайд пустой - удаляем файл:
        else:
            rmpath(guide_file)

    # Пишет метаданные датасета в папкус распакованным бекапом
    # (данные для отдельных задач не входят):
    @classmethod
    def _write_annotations2unzipped_backup(cls,
                                           unzipped_backup,
                                           info,
                                           guide=''):

        # Создаём необъодимые файлы:
        cls._write_info(info, unzipped_backup)    # project.json
        cls._write_guide(guide, unzipped_backup)  # annotation_guide.md

    # Пишет текущее состояние разметки в папку с распакованным бекапом,
    # а при необходимости собирает архив и отправляет его на CVAT-сервер.
    def push(self, local_only=False):

        # Вызываем синхронизацю каждого из подобъектов (задач):
        for task_data in tqdm(self.data,
                              desc='Обновление локальной копии',
                              disable=not self.verbose):
            task_data.push(local_only=True)

        # Добавляем описание датасета в целом:
        self._write_annotations2unzipped_backup(self.unzipped_backup,
                                                self.info, self.guide)

        # Если надо отправлять результат на CVAT-сервер:
        if not local_only:
            if self.cvat_srv_obj is None:
                raise Exception('Не задан cvat_srv_obj. '
                                'Отправка на CVAT-сервер невозможна!')

            # Формируем архив(ы):
            self._compress()

            # Отправляем архив(ы) на сервер
            desc = 'Выгрузка бекапа' if self.verbose else ''
            if self.parted:  # Если отправлять надо по частям

                # Составляем список архивов:
                paths = get_file_list(self.zipped_backup, '.zip')

                # Фиксируем список задач, которые уже находятся в датасета:
                old_tasks = self.cvat_srv_obj.values()

                # Отправляем все архивы на сервер:
                self.cvat_srv_obj.restore_all(paths, desc=desc)

                # Обновляем описание датасета:
                self.cvat_srv_obj.guide = self.guide

                # После успешного восстановления новых задач все старые
                # удаляются:
                for old_task in old_tasks:
                    old_task.delete()

                # На всякий случай актуализируем объект, представляющий
                # датасет в CVAT:
                self.cvat_srv_obj._update_obj()

                # Удаляем все архивы после успешной отправки их на сервер:
                for zip_file in get_file_list(self.zipped_backup):
                    rmpath(zip_file)

            else:  # Если отправлять надо один архив целиком
                with AnnotateIt(desc):

                    # Переходим в CVAT на один уровень выше:
                    parent = self.cvat_srv_obj.parent()

                    # Фиксируем исходный датасет:
                    old_cvat_srv_obj = self.cvat_srv_obj

                    # Выполняем Выгрузку бекапа и заменяем интерфейсный
                    # объект:
                    self.cvat_srv_obj = parent.restore(self.zipped_backup)

                    # После успешного восстановления нового датасета старый
                    # удаляется:
                    old_cvat_srv_obj.delete()

                    # На всякий случай обновляем представителя старого
                    # датасета в CVAT новым:
                    old_cvat_srv_obj._update_obj(self.cvat_srv_obj.obj.id)

                    # Удаляем архив после успешной отправки его на сервер:
                    rmpath(self.zipped_backup)


class CVATTask(CVATBase):
    '''
    Интерфейс для работы с CVAT-задачами.
    '''

    # Парсит распакованный бекап:
    def _parse_unzipped_backup(self):

        # Извлекаем список подзадач и метаданные задачи:
        task, self.info = cvat_backup_task_dir2task(self.unzipped_backup,
                                                    True)
        self.info['labels'] = CVATLabels(self.info['labels'])  # Парсинг меток

        # Создаём для каждой подзадачи экземпляр класса CVATJob:
        self.data = mpmap(CVATJob.from_subtask, task, num_procs=1)

    @staticmethod
    def __prepare_single_raw_file(data_path: str,
                                  task_data_dir: str | None = None,
                                  include_as_is: bool = False):
        '''
        Подготовка бекапа из единственного файла с неразмеченными данными.
        '''
        # Если это файл:
        if os.path.isfile(data_path):

            # Отделяем тип файла от имени:
            full_name, ext = os.path.splitext(data_path)
            ext = ext.lower()

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

            # Берём путь до дирректории:
            full_name = data_path

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

        # Определяем имя задачи как имя файла без его пути и расширения
        # (если оно было):
        task_name = os.path.basename(full_name)

        return file_list, task_name

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

            # Берём имя папки с исходными файлами как имя задачи:
            task_name = split_dir_name_ext(source_dir)[1]

            # Копируем все файлы из этой дирректории в целевую:
            shutil.copytree(source_dir, task_data_dir)

            # Составляем итоговый список файлов с учётом нового
            # расположения:
            file_list = [os.path.join(source_dir, basename)
                         for basename in map(os.path.basename, data_path)]

        # Если брать нужно только сами файлы:
        else:

            file_list = []
            for ind, file in enumerate(data_paths):

                # Копируем файл в целевую папку:
                shutil.copy(file, task_data_dir)

                # Пополняем итоговый список файлов:
                basename = os.path.basename(file)
                file_list.append(os.path.join(task_data_dir, basename))

                # Если это первый файл из списка, то берём имя его папки
                # как имя задачи:
                if ind == 0:
                    task_name = os.path.splitext(basename)[0]

        return file_list, task_name

    # Копирует файлы в папку с распакованным бекапом:
    @classmethod
    def _copy_files2unzipped_backup(cls,
                                    data_path,
                                    include_as_is,
                                    unzipped_backup):
        # Определяем способ копирования:
        if isinstance(data_path, str):  # Если передан всего один файл
            prepare_raw_data = cls.__prepare_single_raw_file
        else:                           # Если файлов несколько
            prepare_raw_data = cls.__prepare_multiple_raw_files

        # Определяем целевую папку для данных:
        task_data_dir = os.path.join(unzipped_backup, 'data')

        # Создаём папку данных для распакованного бекапа, если надо:
        if not os.path.isdir(task_data_dir):
            mkdirs(task_data_dir)

        # Выполняем само копирование:
        file_list, task_name = prepare_raw_data(data_path,
                                                task_data_dir,
                                                include_as_is)

        return {'file_list': file_list,
                'task_name': task_name}

    # Создаёт метаданные (всё кроме самих фото/видео) в папке с распакованным
    # бекапом:
    @classmethod
    def _init_files_in_unzipped_backup(cls,
                                       unzipped_backup,
                                       file_list,
                                       task_name):

        # Определяем номер последнего кадра в данных:
        file = file_list if len(file_list) > 1 else file_list[0]
        stop_frame = VideoGenerator.get_file_total_frames(file) - 1

        # Инициируем инфу о задаче:
        info = cls._init_info(name=task_name,
                              data={
                                  'image_quality': 90,
                                  'chunk_type': 'imageset',
                                  'sorting_method': 'predefined',
                                  'start_frame': 0,
                                  'stop_frame': stop_frame
                              },
                              jobs=[{
                                  'status': 'annotation',
                                  'start_frame': 0,
                                  'stop_frame': stop_frame
                              }])

        # Создаём пустую разметку:
        data = cls._init_annotations(file_list)

        # Определяем папку с данными:
        task_data_dir = os.path.join(unzipped_backup, 'data')

        # Обновляем содержимое файлов в распакованном датасете:
        cls._write_annotations2unzipped_backup(data,
                                               task_data_dir,
                                               unzipped_backup,
                                               info)

    @staticmethod
    def _init_info(name: str = 'unnamed_task',
                   bug_tracker: str = '',
                   status: str = 'completed',
                   subset: str = 'Train',
                   labels: list[dict] = [],
                   version: str = '1.0',
                   data: dict = {
                       'image_quality': 90,
                       'chunk_type': 'imageset',
                       'sorting_method': 'predefined',
                       'start_frame': 0,
                       'stop_frame': 0
                   },
                   jobs: list[dict] = [{'status': 'annotation',
                                        'start_frame': 0,
                                        'stop_frame': 0}],
                   **kwargs):

        # Собираем словарь:
        info = dict(name=name,
                    bug_tracker=bug_tracker,
                    status=status,
                    subset=subset,
                    labels=labels,
                    version=version,
                    data=data,
                    jobs=jobs)

        # Добавляем к словарю доп. параметры и возвращаем:
        return info | kwargs

    """

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
        cvat_task = cls(data_path, *init_args, **init_kwargs,
                        take_data_from='unzipped_backup')

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
    """

    # Создаёт список элементов типа CVATJob для неразмеченных данных:
    @classmethod
    def _init_annotations(cls, file_list):
        return [CVATJob._from_raw_data(file_list)]

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
        cls._write_index(jobs, task_data_dir)  # index.json

        # Пишем файлы описания задачи и разметки:
        cls._write_info(info, unzipped_backup)  # task.json
        cls._write_annotanions(jobs, task_dir)  # annotations.json

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

    # Пишет файл index.json (на самом деле удаляет его, т.к. он необязателен):
    @staticmethod
    def _write_index(jobs, task_data_dir):

        # Определяем путь до нужного файла:
        index_file = os.path.join(task_data_dir, 'index.json')

        # Удаляем этот файл, т.к. его существование не обязательно:
        return rmpath(index_file)

    # Пишет файл описания задачи task.json:
    @staticmethod
    def _write_info(info, unzipped_backup):

        # Определяем путь до нужного файла:
        info_file = os.path.join(unzipped_backup, 'task.json')

        # Создаём копию info и представляем метки в виде словарей:
        info = dict(info)
        info['labels'] = info['labels'].to_dicts()

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

            if self.cvat_srv_obj is None:
                raise ConnectionError('Для отправки задачи в CVAT нужно '
                                      'задать "cvat_srv_obj"!')

            with AnnotateIt('Отправка задачи в CVAT' if self.verbose else ''):

                # Создаём архив(ы):
                self._compress()

                # Инициируем флаг удаления предыдущей задачи:
                delete_old = False

                # Фиксируем исходный CVAT_SRV-объект:
                old_cvat_srv_obj = self.cvat_srv_obj

                # Определяем метод восстановления:
                if self.cvat_srv_obj._hier_lvl == 0:
                    # Если передан сам CVAT-сервер:
                    restore = self.cvat_srv_obj.client.restore_task

                # Если передана именно CVAT-задача:
                elif self.cvat_srv_obj._hier_lvl == 2:

                    # Старую задачу надо будет удалить:
                    delete_old = True

                    # Определяем проект этой задачи:
                    project = self.cvat_srv_obj.parent()

                    # Определяем способ восстановления из бекапа в
                    # зависимости от того, есть у задачи датасет. или нет:
                    if project is None:
                        restore = self.cvat_srv_obj.client.restore_task
                    else:
                        restore = project.restore
                    # Если у задачи есть проект - то восстанавливаем
                    # в нём, а если нет - делаем её свободной.

                # Если передан CVAT-датасет:
                elif self.cvat_srv_obj._hier_lvl == 1:
                    restore = self.cvat_srv_obj.restore

                else:
                    raise ValueError('Параметр "cvat_srv_obj" должен быть'
                                     ' задачей, датасетом или сервером!')

                # Выполняем Выгрузку бекапа и заменяем интерфейсный
                # объект:
                self.cvat_srv_obj = restore(self.zipped_backup)

                # После успешного восстановления нового датасета старый
                # удаляется:
                if delete_old:
                    old_cvat_srv_obj.delete()

    '''
    def __del__(self):
        self.rm_all_paths2remove()
    '''

    def reorder_frames(self, new_order_inds: tuple | list | None = None):
        '''
        Меняет во всех подзадачах очерёдность кадров, если они представлены
        отдельными файлами. Если new_order_inds не указан - сортирует по
        номерам в именах файлов.
        '''
        # Если подзадач не одна (0 или > 1):
        if len(self.data) != 1:

            # Ничего не меняем, если новый порядок не указан:
            if new_order_inds is None:
                return

            # Возвращаем ошибку если указан новый порядок:
            else:
                raise ValueError('Перестановка доступна только для задач с '
                                 'единственной подзадачей, а имеется '
                                 f'{self.data[0]} подзадач!')

        # Получаем список файлов:
        files = self.data[0].file

        # Число файлов (конечно, если это не строка, а список):
        num_files = len(files)

        # Если указан единственный файл, то сортировка тоже не возможна:
        if isinstance(files, str) or num_files == 1:
            if new_order_inds is None:
                return
            else:
                raise ValueError('Перестановка доступна только для задач с '
                                 'несколькими кадрами в отдельных файлах!')

        # Если индексы нового порядка не указаны - доопределяем его для
        # сортировки по номерам в именах кадров:
        if new_order_inds is None:

            # Берём имена файлов:
            nums = [split_dir_name_ext(file)[1] for file in files]

            # На всякий случай удаляем из имён все символы, кроме цифр:
            nums = [''.join(filter(str.isdigit, num)) for num in nums]

            # Интерпретируем наборы цифр как номера кадров:
            nums = list(map(int, nums))

            # Получаем индексы упорядочивания:
            new_order_inds = np.argsort(nums)

        # Если индексы указаны, убеждаемся, что их число соответствует общему
        # числу кадров:
        elif num_files != len(new_order_inds):
            raise ValueError(f'Общее число кадров {num_files}, '
                             f'а не {len(new_order_inds)}!')

        # Определяем начальный и конечный кадры задачи и поздазачи, а так же
        # шаг прореживания:
        data = self.info['data']
        job = self.info['jobs'][0]
        task_start_frame = data['start_frame']
        task_stop_frame = data['stop_frame']
        job_start_frame = job['start_frame']
        job_stop_frame = job['stop_frame']
        step_frame = int(data.get('frame_filter', 'filter=1').split('=')[-1])

        if task_start_frame != 0:
            raise ValueError('Первый кадр задачи должен == 0, '
                             f'а не {task_start_frame}')
        if task_stop_frame != num_files - 1:
            raise ValueError('Последний кадр подзадачи должен == '
                             f'{num_files - 1}, а не {task_stop_frame}')
        if job_start_frame != 0:
            raise ValueError('Первый кадр задачи должен == 0, '
                             f'а не {job_start_frame}')
        if job_stop_frame != num_files - 1:
            raise ValueError('Последний кадр подзадачи должен == '
                             f'{num_files - 1}, а не {job_stop_frame}')
        if step_frame != 1:
            print(step_frame)
            raise ValueError('Последовательность не должна быть прореженной!')

        ################################
        # Выполняем саму перестановку: #
        ################################

        # Меняем очерёдность файлов:
        files = [files[ind] for ind in new_order_inds]

        # Строим вектор индексов перехода:
        frame_reind = invert_index_vector(new_order_inds)

        # Обновляем номера удалённых кадров:
        deleted_frames = []
        for deleted_frame in self.info['data']['deleted_frames']:
            deleted_frames.append(frame_reind[deleted_frame])
        self.info['data']['deleted_frames'] = deleted_frames

        # Меняем метод сортировки:
        self.info['data']['sorting_method'] = 'predefined'

        # Перебираем все подзадачи:
        for cvat_subtask in self.data:
            cvat_subtask._reorder_frames(files, frame_reind.__getitem__)

        return new_order_inds


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
    def _from_raw_data(cls, file: str | list[str] | tuple[str]):

        # Если список из одного элемента - берём его вместо списка:
        if isinstance(file, (tuple, list)) and len(file) == 1:
            file = file[0]

        # Определяем общее число кадров:
        total_frames = VideoGenerator.get_file_total_frames(file)

        # Собираем экземпляр класса с пустой разметкой:
        return cls(None, file, {frame: frame for frame in range(total_frames)})

    # Строит словарь разметки формата annotations.json:
    def _build_annotations(self):

        # Создаём пустой датафрейм, если разметка отсутствует:
        if self.df is None:
            self.df = new_df()

        # Формируем и возвращаем словарь с разметкой:
        return df2annotations(self.df)

    def _reorder_frames(self, files, df_frame_func):
        '''
        Меняет очерёдность кадров, если они представлены отдельными файлами.
        Используется в методе reorder_frames для задач.
        '''
        # Сужаем область применимости метода:
        if (self.df['frame'] != self.df['true_frame']).any():
            raise ValueError('Последовательность не должна '
                             'быть прореженной!')
        for i in range(len(self.true_frames)):
            if self.true_frames.get(i, None) != i:
                raise ValueError('Последовательность должна '
                                 'быть непрореженной и начинаться с 0!')

        # Интерполируем разметку:
        self.df = interpolate_df(self.df, self.true_frames)

        # Отбрасываем скрытые объекты:
        self.df = split_df_by_visibility(self.df)[0]

        # Обновляем номера кадров прореженной и исходной последовательности
        # кадров:
        self.df['frame'] = self.df['frame'].apply(df_frame_func)
        self.df['true_frame'] = self.df['true_frame'].apply(df_frame_func)

        # Восстанавливаем ключи, скрывающие треки:
        self.df = hide_skipped_objects_in_df(self.df, self.true_frames)

        # Обновляем сам список файлов с кадрами:
        self.file = files