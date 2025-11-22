'''
********************************************
* Работа с метками классов и суперклассов. *
*                                          *
*   Предпологается, что таблица классов и  *
* таблицы суперклассов схранены в формате  *
* Excel (*.xlsx).                          *
*                                          *
*                                          *
* Основные классы:                         *
*   LabelsConvertor - класс-конвертор,     *
*       читающий и интерпретирующий        *
*       заданные excel-файлы. Содержит     *
*       все необходимые методы для работы  *
*       с классами и суперклассами.        *
*                                          *
********************************************
'''


import os
import pandas as pd
from treelib import Tree

from utils import rim2arabic

# Имена столбцов в labels.xlsx:
cvat_label_column = 'Метка в CVAT'                     # Метки CVAT-датасета
uuid_label_column = 'Метка в другом источнике данных'  # Метки иного-датасета

# Имена столбцов в superlabels.xlsx:
superlabel_column = 'Наименование суперкласса'         # Имена суперклассов
scl_clsnme_column = 'Классы (содержимое суперкласса)'  # Имена классов
scl_number_column = '№ п/п'      # Номера суперклассов
scl_prrity_column = 'Приоритет'  # Приоритет суперклассов


def _fix_string(s):
    '''
    Заменяет неправильные пробелы на правильные и убирает пробелы в начале и
    конце
    '''
    return s.replace('\xa0', ' ').strip() if isinstance(s, str) else s


def _file2labels_df(file):
    '''
    Загружает xlsx-файл со списком классов.
    '''
    # Загружаем полный список классов:
    df = pd.read_excel(file, engine='openpyxl')

    # Отбрасываем столбцы, чьи имена не заданы явно:
    df = df.drop(columns=[column for column in df.columns
                          if 'Unnamed: ' in column])

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Задаём первый столбец в качестве индекса:
    df = df.set_index(df.columns[0])

    # Отбрасывание пустых строк:
    df = df[~df.index.isna()]

    return df


def _file2superlabels_df(file):
    '''
    Загружает xlsx-файл со списком суперклассов.
    '''
    # Читаем список суперклассов:
    df = pd.read_excel(file, engine='openpyxl')

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Заполняем пропуски:
    for ind in range(len(df)):

        # Считываем текущие значения в строке:
        cur_superlabel_name = df.iloc[ind][superlabel_column]  # Имя
        cur_scl_number = df.iloc[ind][scl_number_column]       # Номер
        cur_scl_priority = df.iloc[ind][scl_prrity_column]     # Приоритет

        # Если cur_superlabel_name не NaN, значит это новый класс:
        if cur_superlabel_name is not None:

            # Остальные параметры тоже должны быть не NaN:
            assert not (pd.isna(cur_scl_number) or pd.isna(cur_scl_priority))

            # Читаем из строки действительные значения для текущего
            # суперкласса:
            superlabel_name = cur_superlabel_name  # Имя
            scl_number = cur_scl_number            # Номер
            scl_priority = cur_scl_priority        # Приоритет

        # Если cur_superlabel_name = NaN, тострока не дозаполнена:
        else:

            # Остальные параметры тоже должны быть NaN:
            assert pd.isna(cur_scl_number) and pd.isna(cur_scl_priority)

            # Пишем в строку пропущенные значения для текущего суперкласса:
            df.loc[ind, superlabel_column] = superlabel_name  # Имя
            df.loc[ind, scl_number_column] = scl_number       # Номер
            df.loc[ind, scl_prrity_column] = scl_priority     # Приоритет

    # Приводим номера суперклассов к целочислоенному типу и сдвигаем, чтобы
    # суперкласс неиспользуемых объектов был под номером -1, а остальные
    # начинались с 0:
    df[scl_number_column] = df[scl_number_column].apply(int) - 1
    # В результате исключённые объекты получат значение -2 !

    return df


def _labels_df2tree(df):
    '''
    Строит дерево данных из pandas-dataframe в индексах которого прописаны
    номера списков с вложенностью.
    '''
    # Создаём дерево и указываем корень:
    tree = Tree()
    tree.create_node('Номер класса', 'Номер класса')

    # Перебор по всем строкам таблицы:
    for ind in df.index:

        # Имена класса для текущей строки:
        cvat_label = df[cvat_label_column][ind]
        uuid_label = df[uuid_label_column][ind]

        # Если параметр не заполнен, то в дерево надо будет вносить None:
        if pd.isna(cvat_label):
            cvat_label = None
        if pd.isna(uuid_label):
            uuid_label = None

        # Объединение меток
        label = (cvat_label, uuid_label)

        # Разделяем строку ind на индекс (позицию во вложенных списках) и
        # расшифровку класса:

        # Заменяем нетипичные пробелы на типичные:
        ind = ind.replace('\xa0', ' ')
        # Расщепляем строку на слова:
        words = [word for word in ind.strip().split(' ') if word]
        ind = words[0]              # Первое слово является позицией в списках
        name = ' '.join(words[1:])  # Остальные слова расшифровывают класс
        assert (ind[-1] == '.')     # В конце индекса должна стоять точка
        ind = ind[:-1]              # Её отбрасываем

        # Парсим строку индекса и вносим данные в нужную ветку дерева:

        # Если индекс с арабскими цифрами:
        if ind[0] in '0123456789':
            # Определяем ветку и листок в дереве:
            ind = (group, *map(int, ind.split('.')))
            parent = tuple(ind[:-1])

            # Вносим в дерево новые данные:
            tree.create_node(name, ind, parent=parent, data=label)

        # Если индекс записан римскими цифрами:
        else:
            # Определяем ветку в дереве:
            group = rim2arabic(ind)

            # Вносим в дерево новые данные:
            tree.create_node(name, (group,),
                             parent='Номер класса', data=label)

    return tree


def _tree2label_meaning_dicts(tree):
    '''
    Формируем словари перехода от меток к их расшифровкам:
    '''
    label2label_meaning = {}  # Label -> расшифровка
    uuid2label_meaning = {}   # UUID  -> расшифровка

    # Перебираем все строки таблицы классов:
    for node in tree.expand_tree(mode=Tree.DEPTH):

        # Пропускаем классы, не имеющие меток:
        if tree[node].data in (None, (None, None)):
            continue

        # Считываем параметры класса
        label_meaning = tree[node].tag            # Расшифровка
        label, uuid_label = tree[node].data  # Метки

        # Вносим существующие метки в cvat-словарь:
        if label is not None:

            # Перевод в нижний регистр:
            label = label.lower()

            # Если такая метка уже встречалась:
            if label in label2label_meaning:

                # Выводим ошибку, если текущая расщифровка не совпадает с
                # предыдущей:
                if label2label_meaning[label] != label_meaning:
                    error_str = \
                        f'Для метки "{label}" встретились' + \
                        'следующие несовпадающие расшифровки:\n' + \
                        f'"{label2label_meaning[label]}"' + \
                        f'и "{label_meaning}"!'
                    raise KeyError(error_str)

            # Добавляем метку, если она не встречалась:
            else:
                label2label_meaning[label] = label_meaning

        # Вносим существующие метки в gg-словарь:
        if uuid_label is not None:

            # Перевод в нижний регистр:
            uuid_label = uuid_label.lower()

            # Если такая метка уже встречалась:
            if uuid_label in uuid2label_meaning:

                # Выводим ошибку, если текущая расщифровка не совпадает с
                # предыдущей:
                if uuid2label_meaning[uuid_label] != label_meaning:
                    error_str = f'Для метки "{uuid_label}" встретились ' + \
                        'следующие несовпадающие расшифровки:\n' + \
                        f'"{uuid2label_meaning[uuid_label]}" и ' + \
                        f'"{label_meaning}"!'
                    raise KeyError(error_str)

            # Добавляем метку, если она не встречалась:
            else:
                uuid2label_meaning[uuid_label] = label_meaning

    return label2label_meaning, uuid2label_meaning


def make_yolo_label2superlabel_meaning(superlabels):
    '''
    Строит словарь перехода от YOLO-меток к имени соответствующего
    суперкласса.
    '''
    # Заполняемый словарь:
    yolo_label2superlabel_meaning = {}

    # Перебираем все строки датафрейма суперклассов:
    for ind in range(len(superlabels)):

        # Берём из строки нужные данные:
        row = superlabels.iloc[ind]
        yolo_label = int(row[scl_number_column])
        superlabel_meaning = row[superlabel_column]

        # Если такой ключ уже внесён в словарь:
        if yolo_label in yolo_label2superlabel_meaning:

            # Текущее значение не должно противоречить уже имеющемуся в
            # словаре:
            if yolo_label2superlabel_meaning[yolo_label] != superlabel_meaning:
                raise KeyError('В таблице суперклассов метка ' +
                               f'"{scl_number_column}" имеет ' +
                               'несколько несовпадающих значений!')

        # Если такого ключа ещё нет, то вносим запись:
        else:
            yolo_label2superlabel_meaning[yolo_label] = superlabel_meaning
            '''
            # При этом отрицательные ключи заменяем на None:
            key = yolo_label if yolo_label >= 0 else None
            yolo_label2superlabel_meaning[key] = superlabel_meaning
            # Это нужно для того, чтобы суперкласс неиспользуемых
            # объектов имел None вместо своего номер.
            '''
    return yolo_label2superlabel_meaning


def make_label_meaning2superlabel_meaning(superlabels):
    '''
    Строит из датафрейма словарь перехода от имени класса к имени
    соответствующего суперкласса.
    '''
    # Заполняемый словарь:
    label_meaning2superlabel_meaning = {}

    # Перебираем все строки датафрейма суперклассов:
    for ind in range(len(superlabels)):

        # Берём из строки нужные данные:
        row = superlabels.iloc[ind]
        # yolo_label = row[scl_number_column]
        label_meaning = row[scl_clsnme_column]
        superlabel_meaning = row[superlabel_column]

        # Если такой ключ уже внесён в словарь:
        if label_meaning in label_meaning2superlabel_meaning:
            raise KeyError(f'В таблице суперклассов метка "{label_meaning}"' +
                           ' встречается минимум дважды!')

        # Если такого ключа ещё нет, то вносим запись:
        else:

            # При этом отрицательные ключи заменяем на None:
            label_meaning2superlabel_meaning[label_meaning.lower()] = \
                superlabel_meaning
            # Это нужно для того, чтобы суперкласс неиспользуемых объектов
            # имел None вместо своего номера

    return label_meaning2superlabel_meaning


def init_df_counter(index, column_name='num'):
    '''
    Создание датафрейма-счётчика.
    '''
    # Инициализация датаферйма с индексами:
    df = pd.DataFrame(index=index)

    # Задаём имя индексирущего столбца:
    df.index.name = 'Класс объекта'

    # Создаётся столб с заданным именем, заполнненный нулями:
    df[column_name] = 0

    return df


label LabelsConvertor:
    '''
    Класс-утилита для работы с классами, метками, суперклассами и т.п.
    '''

    def __init__(self,
                 labels_info: str | dict = 'labels.xlsx',
                 superlabels_info: str | dict = 'superlabels.xlsx',
                 on_call: str = label2yolo):
        '''
            labels_info:
                xlsx-файл, с таблицей или словарь [метка в CVAT -> класс].
            superlabels_info:
                xlsx-файл, с таблицей или словарь [класс -> суперкласс].
            on_call:
                оснонвная ф-ция экземпляра класса: {'label2yolo', label
        '''
        # Папка с файлами, содержащими информацию по всем классам объектов:
        defalut_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'labels_info')

        # Если заданные пути не существуеют - берём файлы из defalut_dir:
        if not os.path.isfile(labels_info):
            labels_info = os.path.join(defalut_dir, labels_info)
        if not os.path.isfile(superlabels_info):
            superlabels_info = os.path.join(defalut_dir, superlabels_info)

        # Загружаем таблицы:
        self.labels = read_labels2df(labels_info)
        self.superlabels = read_superlabels2df(superlabels_info)

        # Строим дерево классов:
        self.tree = labels2tree(self.labels)
        # self.tree.show()

        # Создаём словари перехода от меток к их расшифровкам:
        self.cvat_label2label_meaning, self.uuid2label_meaning = \
            make_label2label_meaning_dicts(self.tree)

        # Строим словари номер_суперкласса <-> имя_суперкласса
        # (используется в YOLO):
        self.yolo_label2superlabel_meaning = \
            make_yolo_label2superlabel_meaning(self.superlabels)
        self.superlabel_meaning2yolo_label = {
            v.lower(): k for k, v in self.yolo_label2superlabel_meaning.items()
        }

        assert len(self.yolo_label2superlabel_meaning) == \
            len(self.superlabel_meaning2yolo_label)

        # Словарь номер_класса -> расшифровка_класса (в отличие от
        # yolo_label2superlabel_meaning не включает не используемый класс):
        self.yolo_label_ind2superlabel_meaning = {
            k: v for k, v in self.yolo_label2superlabel_meaning.items()
            if k is not None
        }

        # Строим словарь перехода от имён классов к именам суперклассов
        self.label_meaning2superlabel_meaning = \
            make_label_meaning2superlabel_meaning(self.superlabels)

        # Списки расшифровок классов, имеющих свои метки:
        self.cvat_meanings_list = [self.cvat_label2label_meaning[label.lower()]
                                   for label in self.labels[cvat_label_column]
                                   if pd.notna(label)]
        self.uuid_meanings_list = [self.uuid2label_meaning[label.lower()]
                                   for label in self.labels[uuid_label_column]
                                   if pd.notna(label)]

        # Общие списки расшифровок классов и суперклассов:
        self.superlabel_meaning_list = \
            list(self.yolo_label2superlabel_meaning.values())
        cvat_label_meaning_list = list(self.cvat_label2label_meaning.values())
        uuid2label_meaning_list = list(self.uuid2label_meaning.values())
        self.label_meaning_list = cvat_label_meaning_list + \
            uuid2label_meaning_list

        # Отбрасываем повторения в списках:
        self.cvat_meanings_list = list(dict.fromkeys(self.cvat_meanings_list))
        self.uuid_meanings_list = list(dict.fromkeys(self.uuid_meanings_list))
        self.superlabel_meaning_list = \
            list(dict.fromkeys(self.superlabel_meaning_list))

        # Создаём счётчики классов каждого датасета и суперклассов:
        self.cvat_counter, self.uuid_counter, self.superlabel_counter = \
            map(init_df_counter, [self.cvat_meanings_list,
                                  self.uuid_meanings_list,
                                  self.superlabel_meaning_lis])

        # Делаем экземпляр класса вызываемым:
        self.set_call(on_call)

    def set_call(self, on_call):
        '''
        Устанавливае ф-ию __call__.
        '''
        # Если указано имя единственной ф-ии:
        if isinstance(on_call, str):
            if hasattr(self, on_call):
                self.__call__ = getattr(self, on_call)
            else:
                raise ValueError(f'Несуществующая функция: {on_call}!')

        # Если указан целый список/кортеж имён функций, то проверяем
        # корректность данных:
        elif isinstance(on_call, (list, tuple)):
            if not len(on_call):
                raise ValueError('Список/кортеж функций пуст!')
            for func in on_call:
                if not isinstance(f, str):
                    raise ValueError('Список/кортеж должен содержать ' +
                                     f'строки. Получен: {type(func)}!')
                elif not hasattr(self, func):
                    raise ValueError(f'Несуществующая функция: {func}!')

        # Сборка комплексной функции:
        def compose(self, arg):
            for func in on_call:
                arg = getattr(self, func)(arg)
            return arg

        self.__call__ = compose

    # Возвращает новые счётчики классов:
    def init_df_counter(self, source_type='superlabels', column_name='num'):

        # На всякий случай принудительно переводим тип датасета в нижний
        # регистр:
        source_type = source_type.lower()

        # Берём нужный уже инициированный датафрейм за основу:
        if source_type == 'cvat':
            df = self.cvat_counter
        elif source_type == 'uuid':
            df = self.uuid_counter
        elif source_type == 'superlabels':
            df = self.superlabel_counter
        else:
            raise ValueError(f'Неизвестный тип датасета: "{source_type}"!')

        # Возвращаем копию датафрейма с заменой имени столбца на заданный:
        return df.rename({'num': column_name}, axis='columns')

    # Конвертация метки любого типа в её расшифровку:
    def label2meaning(self, label):

        # Переводим в нижный регистр:
        lower_label = label.lower()

        # Ищем подходящюю расшифровку по словарям:
        if lower_label in self.cvat_label2label_meaning:
            label_meaning = self.cvat_label2label_meaning[lower_label]

        elif lower_label in self.uuid2label_meaning:
            label_meaning = self.uuid2label_meaning[lower_label]

        else:
            raise KeyError(f'Неизвестная метка "{label}"!')

        return label_meaning

    def label2yolo(self, label):
        '''
        Переводит метку из CVAT или иного датасета в номер суперкласса.
        '''

        # Переводим любую метку в её расшифровку:
        label_meaning = self.label2meaning(label)

        # Получаем расшифровку суперкласса:
        superlabel_meaning = \
            self.label_meaning2superlabel_meaning[label_meaning.lower()]

        # Возвращаем индекс суперкласса:
        return self.superlabel_meaning2yolo_label[superlabel_meaning.lower()]

    def apply2df(self, df):
        '''
        Заменяет в датафрейме все метки на их номера.
        '''
        # Делаем копию исходного датафрейма, чтобы не менять оригинал:
        df = df.copy()

        # Замета меток на номера суперклассов:
        df['label'] = df['label'].apply(self)

        return df

    def apply2subtask(self, subtask):
        '''
        Заменяет в датафрейме подзадачи все метки на их номера.
        '''
        df, file, true_frames = subtask
        return self.apply2df(df), file, true_frames


def checkout_labels_in_tasks(tasks, labels_convertor):
    '''
    Формирует список неподдерживаемых меток по всему списку задач.
    '''
    # Инициализация множества неподдерживаемых меток:
    unsupported_labels = set()

    # Перебор по всем задачам:
    for task in tasks:

        # Перебор по всем подзадачам:
        for df, _, _ in task:

            # Перебор по всем меткам в подзадаче:
            for label in df['label'].unique():

                # Добавляем все незнакомые метки в множество:
                try:
                    labels_convertor(label)
                except KeyError as err:
                    if label not in unsupported_labels:
                        unsupported_labels.add(label)
                        print('Неизвестная метка {:>50} :'.format(label), err)

    return unsupported_labels


def apply_labels_convertor2tasks(tasks, labels_convertor):
    '''
    Применяет labels_convertor ко всем классам во всех задачах.
    Применяется для списка задач, загруженных, например, с помощью
    cvat_backups2tasks из cvat.py.
    '''
    tasks_ = []
    for task in tasks:
        task = [(labels_convertor.apply2df(df), file, true_frames)
                for df, file, true_frames in task]
        tasks_.append(task)
    return tasks_