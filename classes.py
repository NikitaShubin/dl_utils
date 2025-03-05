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

# Имена столбцов в classes.xlsx:
cvat_label_column = 'Метка в CVAT'                    # Метки  CVAT-датасета
uuid_label_column = 'Метка в другом источнике данных' # Метки иного-датасета

# Имена столбцов в superclasses.xlsx:
superclass_column = 'Наименование суперкласса'        # Имена     суперклассов
scl_clsnme_column = 'Классы (содержимое суперкласса)' # Имена          классов
scl_number_column = '№ п/п'                           # Номера    суперклассов
scl_prrity_column = 'Приоритет'                       # Приоритет суперклассов


def fix_string(s):
    '''
    Заменяет неправильные пробелы на правильные и убирает пробелы в начале и конце
    '''
    return s.replace('\xa0', ' ').strip() if isinstance(s, str) else s


def read_classes2df(file):
    '''
    Загружает xlsx-файл со списком классов.
    '''
    # Загружаем полный список классов:
    df = pd.read_excel(file, engine='openpyxl')
    
    # Отбрасываем столбцы, чьи имена не заданы явно:
    df = df.drop(columns=[column for column in df.columns if 'Unnamed: ' in column])
    
    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(fix_string)
    
    # Задаём первый столбец в качестве индекса:
    df = df.set_index(df.columns[0])
    
    # Отбрасывание пустых строк:
    df = df[~df.index.isna()]
    
    return df


def read_superclasses2df(file):
    '''
    Загружает xlsx-файл со списком суперклассов.
    '''
    # Читаем список суперклассов:
    df = pd.read_excel(file, engine='openpyxl')
    
    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(fix_string)
    
    # Заполняем пропуски:
    for ind in range(len(df)):
        
        # Считываем текущие значения в строке:
        cur_superclass_name = df.iloc[ind][superclass_column] # Имя
        cur_scl_number      = df.iloc[ind][scl_number_column] # Номер
        cur_scl_priority    = df.iloc[ind][scl_prrity_column] # Приоритет
        
        # Если cur_superclass_name не NaN, значит это новый класс:
        if cur_superclass_name == cur_superclass_name:
            
            # Остальные параметры тоже должны быть не NaN:
            assert cur_scl_number   == cur_scl_number
            assert cur_scl_priority == cur_scl_priority
            
            # Читаем из строки действительные значения для текущего суперкласса:
            superclass_name = cur_superclass_name # Имя
            scl_number      = cur_scl_number      # Номер
            scl_priority    = cur_scl_priority    # Приоритет
        
        # Если cur_superclass_name NaN, то он не равен сам себе, а, значит, строка не дозаполнена:
        else:
            
            # Остальные параметры тоже должны быть NaN:
            assert cur_scl_number   != cur_scl_number
            assert cur_scl_priority != cur_scl_priority
            
            # Пишем в строку пропущенные значения для текущего суперкласса:
            df.loc[ind, superclass_column] = superclass_name # Имя
            df.loc[ind, scl_number_column] = scl_number      # Номер
            df.loc[ind, scl_prrity_column] = scl_priority    # Приоритет
    
    # Приводим номера суперклассов к целочислоенному типу и сдвигаем, чтобы суперкласс ...
    # ... неиспользуемых объектов был под номером -1, а остальные начинались с 0:
    df[scl_number_column] = df[scl_number_column].apply(int) - 1
    # В результате исключённые объекты получат значение -2 !
    
    return df


def classes2tree(df):
    '''
    Строит дерево данных из pandas-dataframe в индексах которого прописаны
    номера списков с вложенностью. Работает только на данных, полученных через 
    read_full_label_names_df('classes_info/Cписок классов для разметчиков.xlsx').
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
        
        # Разделяем строку ind на индекс (позицию во вложенных списках) и расшифровку класса:
        ind = ind.replace('\xa0', ' ')                            # Заменяем нетипичные пробелы на типичные
        words = [word for word in ind.strip().split(' ') if word] # Расщепляем строку на слова
        ind = words[0]             # В первом слове зашифрован индекс (позиция во вложенных списках)
        name = ' '.join(words[1:]) # Остальные слова расшифровывают класс
        assert (ind[-1] == '.')    # В конце индекса должна стоять точка
        ind = ind[:-1]             # Её отбрасываем
        
        # Парсим строку индекса и вносим данные в нужную ветку дерева:
        
        # Если индекс с арабскими цифрами:
        if ind[0] in '0123456789':
            # Определяем ветку и листок в дереве:
            ind = (group, *map(int, ind.split('.')))
            parent=tuple(ind[:-1])
            
            # Вносим в дерево новые данные:
            tree.create_node(name, ind, parent=parent, data=label)
        
        # Если индекс записан римскими цифрами:
        else:
            # Определяем ветку в дереве:
            group = rim2arabic(ind)
            
            # Вносим в дерево новые данные:
            tree.create_node(name, (group,), parent='Номер класса', data=label)
    
    return tree


def make_label2class_meaning_dicts(tree):
    '''
    Формируем словари перехода от меток к их расшифровкам:
    '''
    cvat_label2class_meaning = {} # CVAT -> расшифровка
    gg_uuid2class_meaning    = {} # UUID -> расшифровка
    
    # Перебираем все строки таблицы классов:
    for node in tree.expand_tree(mode=Tree.DEPTH):
        
        # Пропускаем классы, не имеющие меток:
        if tree[node].data in (None, (None, None)):
            continue
        
        # Считываем параметры класса
        class_meaning          = tree[node].tag  # Расшифровка
        cvat_label, uuid_label = tree[node].data # Метки
        
        # Вносим существующие метки в cvat-словарь:
        if cvat_label is not None:
            
            # Перевод в нижний регистр:
            cvat_label = cvat_label.lower()
            
            # Если такая метка уже встречалась:
            if cvat_label in cvat_label2class_meaning:
                
                # Выводим ошибку, если текущая расщифровка не совпадает с предыдущей:
                if cvat_label2class_meaning[cvat_label] != class_meaning:
                    error_str = f'Для метки "{cvat_label}" встретились следующие несовпадающие расшифровки:\n'
                    error_str += f'"{cvat_label2class_meaning[cvat_label]}" и "{class_meaning}"!'
                    print(error_str)
                    #raise KeyError(error_str)
            
            # Добавляем метку, если она не встречалась:
            else:
                cvat_label2class_meaning[cvat_label] = class_meaning
        
        # Вносим существующие метки в gg-словарь:
        if uuid_label is not None:
            
            # Перевод в нижний регистр:
            uuid_label = uuid_label.lower()
            
            # Если такая метка уже встречалась:
            if uuid_label in gg_uuid2class_meaning:
                
                # Выводим ошибку, если текущая расщифровка не совпадает с предыдущей:
                if gg_uuid2class_meaning[uuid_label] != class_meaning:
                    error_str = f'Для метки "{uuid_label}" встретились следующие несовпадающие расшифровки:\n'
                    error_str += f'"{gg_uuid2class_meaning[uuid_label]}" и "{class_meaning}"!'
                    print(error_str)
                    #raise KeyError(error_str)
            
            # Добавляем метку, если она не встречалась:
            else:
                gg_uuid2class_meaning[uuid_label] = class_meaning
    
    return cvat_label2class_meaning, gg_uuid2class_meaning


def make_yolo_label2superclass_meaning(superclasses):
    '''
    Строит словарь перехода от YOLO-меток к имени соответствующего суперкласса.
    '''
    # Заполняемый словарь:
    yolo_label2superclass_meaning = {}
    
    # Перебираем все строки датафрейма суперклассов:
    for ind in range(len(superclasses)):
        
        # Берём из строки нужные данные:
        row = superclasses.iloc[ind]
        yolo_label         = int(row[scl_number_column])
        superclass_meaning =     row[superclass_column]

        # Если такой ключ уже внесён в словарь:
        if yolo_label in yolo_label2superclass_meaning:

            # Текущее значение не должно противоречить уже имеющемуся в словаре:
            if yolo_label2superclass_meaning[yolo_label] != superclass_meaning:
                raise KeyError(f'В таблице суперклассов метка "{scl_number_column}" имеет несколько несовпадающих значений!')

        # Если такого ключа ещё нет, то вносим запись:
        else:
            yolo_label2superclass_meaning[yolo_label] = superclass_meaning
            '''
            # При этом отрицательные ключи заменяем на None:
            yolo_label2superclass_meaning[yolo_label if yolo_label >= 0 else None] = superclass_meaning
            # Это нужно для того, чтобы суперкласс неиспользуемых объектов имел None вместо своего номера
            '''
    return yolo_label2superclass_meaning


def make_class_meaning2superclass_meaning(superclasses):
    '''
    Строит словарь перехода от имени класса к имени соответствующего суперкласса.
    '''
    # Заполняемый словарь:
    class_meaning2superclass_meaning = {}
    
    # Перебираем все строки датафрейма суперклассов:
    for ind in range(len(superclasses)):
        
        # Берём из строки нужные данные:
        row = superclasses.iloc[ind]
        class_meaning      = row[scl_clsnme_column]
        superclass_meaning = row[superclass_column]
        
        # Если такой ключ уже внесён в словарь:
        if class_meaning in class_meaning2superclass_meaning:
            
            # Текущее значение не должно противоречить уже имеющемуся в словаре:
            if class_meaning2superclass_meaning[yolo_label] != superclass_meaning:
                raise KeyError(f'В таблице суперклассов метка "{scl_number_column}" имеет несколько несовпадающих значений!')
        
        # Если такого ключа ещё нет, то вносим запись:
        else:
            
            # При этом отрицательные ключи заменяем на None:
            class_meaning2superclass_meaning[yolo_label if yolo_label >= 0 else None] = superclass_meaning
            # Это нужно для того, чтобы суперкласс неиспользуемых объектов имел None вместо своего номера
    
    return class_meaning2superclass_meaning


def make_class_meaning2superclass_meaning(superclasses):
    '''
    Строит словарь перехода от имени класса к имени соответствующего суперкласса.
    '''
    # Заполняемый словарь:
    class_meaning2superclass_meaning = {}
    
    # Перебираем все строки датафрейма суперклассов:
    for ind in range(len(superclasses)):
        
        # Берём из строки нужные данные:
        row = superclasses.iloc[ind]
        yolo_label         = row[scl_number_column]
        class_meaning      = row[scl_clsnme_column]
        superclass_meaning = row[superclass_column]
        
        # Если такой ключ уже внесён в словарь:
        if class_meaning in class_meaning2superclass_meaning:
            raise KeyError(f'В таблице суперклассов метка "{class_meaning}" встречается минимум дважды!')
        
        # Если такого ключа ещё нет, то вносим запись:
        else:
            
            # При этом отрицательные ключи заменяем на None:
            class_meaning2superclass_meaning[class_meaning.lower()] = superclass_meaning
            # Это нужно для того, чтобы суперкласс неиспользуемых объектов имел None вместо своего номера
    
    return class_meaning2superclass_meaning


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


class LabelsConvertor:
    '''
    Класс-утилита для работы с классами, метками, суперклассами и т.п.
    '''
    def __init__(self                                                                                                                                              ,
                 classes_info_file : 'xlsx-файл, содержащий таблицу классов с привязкой к  меткам в CVAT и GG' = 'Cписок_классов_для_разметчиков.xlsx'             ,
            superclasses_info_file : 'xlsx-файл, содержащий таблицу суперклассов с привязкой к классам'        = 'Список_классов_для_поиска_онлайн_08_классов.xlsx'):

        # Папка с файлами, содержащими информацию по всем классам объектов:
        defalut_classes_info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classes_info')

        # Если заданные пути не существуеют - берём файлы из defalut_classes_info_dir:
        if not os.path.isfile(     classes_info_file):      classes_info_file = os.path.join(defalut_classes_info_dir,      classes_info_file)
        if not os.path.isfile(superclasses_info_file): superclasses_info_file = os.path.join(defalut_classes_info_dir, superclasses_info_file)

        # Загружаем таблицы:
        self.     classes =      read_classes2df(     classes_info_file) # Загружаем таблицу классов
        self.superclasses = read_superclasses2df(superclasses_info_file) # Загружаем таблицу суперклассов

        # Строим дерево классов:
        self.tree = classes2tree(self.classes)
        #self.tree.show()

        # Создаём словари перехода от меток к их расшифровкам:
        self.cvat_label2class_meaning, self.gg_uuid2class_meaning = make_label2class_meaning_dicts(self.tree)

        # Строим словари номер_суперкласса <-> имя_суперкласса (используется в YOLO):
        self.yolo_label2superclass_meaning = make_yolo_label2superclass_meaning(self.superclasses)
        self.superclass_meaning2yolo_label = {v.lower(): k for k, v in self.yolo_label2superclass_meaning.items()}

        assert len(self.yolo_label2superclass_meaning) == len(self.superclass_meaning2yolo_label)

        # Словарь номер_класса -> расшифровка_класса (в отличие от yolo_label2superclass_meaning не включает не используемый класс):
        self.yolo_class_ind2superclass_meaning = {k:v for k, v in self.yolo_label2superclass_meaning.items() if k is not None}

        # Строим словарь перехода от имён классов к именам суперклассов
        self.class_meaning2superclass_meaning = make_class_meaning2superclass_meaning(self.superclasses)

        # Списки расшифровок классов, имеющих свои метки:
        self.cvat_meanings_list = [self.cvat_label2class_meaning[label.lower()] for label in self.classes[cvat_label_column] if pd.notna(label)]
        self.  gg_meanings_list = [self.   gg_uuid2class_meaning[label.lower()] for label in self.classes[uuid_label_column] if pd.notna(label)]

        # Общие списки расшифровок классов и суперклассов:
        self.superclass_meaning_list = list(self.yolo_label2superclass_meaning.values())
        self.     class_meaning_list = list(self.cvat_label2class_meaning     .values()) + list(self.gg_uuid2class_meaning.values())

        # Отбрасываем повторения в списках:
        self.     cvat_meanings_list = list(dict.fromkeys(self.     cvat_meanings_list))
        self.       gg_meanings_list = list(dict.fromkeys(self.       gg_meanings_list))
        self.superclass_meaning_list = list(dict.fromkeys(self.superclass_meaning_list))

        # Создаём счётчики классов каждого датасета и суперклассов:
        self.      cvat_counter = init_df_counter(self.     cvat_meanings_list)
        self.        gg_counter = init_df_counter(self.       gg_meanings_list)
        self.superclass_counter = init_df_counter(self.superclass_meaning_list)

    # Возвращает новые счётчики классов:
    def init_df_counter(self, source_type='superclasses', column_name='num'):

        # На всякий случай принудительно переводим тип датасета в нижний
        # регистр:
        source_type = source_type.lower()

        # Берём нужный уже инициированный датафрейм за основу:
        if source_type == 'cvat':
            df = self.cvat_counter
        elif source_type == 'cg':
            df = self.cvat_counter
        elif source_type == 'gg':
            df = self.gg_counter
        elif source_type == 'superclasses':
            df = self.superclass_counter
        else:
            raise ValueError(f'Неизвестный тип датасета: "{source_type}".')

        # Возвращаем копию датафрейма с заменой имени столбца на заданный:
        return df.rename({'num': column_name}, axis='columns')

    # Конвертация метки любого типа в её расшифровку:
    def any_label2meaning(self, label):

        # Переводим в нижный регистр:
        lower_label = label.lower()

        # Ищем подходящюю расшифровку по словарям:
        if lower_label in self.cvat_label2class_meaning:
            class_meaning = self.cvat_label2class_meaning[lower_label]

        elif lower_label in self.gg_uuid2class_meaning:
            class_meaning = self.gg_uuid2class_meaning[lower_label]

        else:
            raise KeyError(f'Неизвестная метка "{label}"!')

        return class_meaning

    # Переводит метку из CVAT или иного датасета в номер суперкласса:
    def __call__(self, label):

        # Переводим любую метку в её расшифровку:
        class_meaning = self.any_label2meaning(label)

        # Получаем расшифровку суперкласса:
        superclass_meaning = \
            self.class_meaning2superclass_meaning[class_meaning.lower()]

        # Возвращаем индекс суперкласса:
        return self.superclass_meaning2yolo_label[superclass_meaning.lower()]

    # Заменяет в датафрейме все метки на их номера.
    def apply2df(self, df):

        # Делаем копию исходного датафрейма, чтобы не менять оригинал:
        df = df.copy()

        # Замета меток на номера суперклассов:
        df['label'] = df['label'].apply(self)

        return df


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
            for label in df['label'].drop_duplicates():
                
                # Добавляем все незнакомые метки в множество:
                try:
                    labels_convertor(label)
                except KeyError as err:
                    if label not in unsupported_labels:
                        unsupported_labels.add(label)
                        print('Неизвестная метка {:>50} :'.format(label), err)
    
    return unsupported_labels

