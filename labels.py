"""labesl.py.

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
.
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pandas as pd
from treelib import Tree

from utils import rim2arabic

# Имена столбцов в labels.xlsx:
label_column = 'Метка в CVAT'  # Имя класса (метка)
synonym_column = 'Метка в другом источнике данных'  # Синоним класса

# Имена столбцов в superlabels.xlsx:
superlabel_column = 'Наименование суперкласса'  # Имена суперклассов
scl_clsnme_column = 'Классы (содержимое суперкласса)'  # Имена классов
scl_number_column = '№ п/п'  # Номера суперклассов
scl_prrity_column = 'Приоритет'  # Приоритет суперклассов


def _fix_string(lbl: str) -> str:
    """Чуть корректирует значения ячеек в xlsx-таблице.

    Заменяет неправильные пробелы на правильные и убирает пробелы в начале и
    конце.
    """
    return lbl.replace('\xa0', ' ').strip() if isinstance(lbl, str) else lbl


def _file2labels_df(file_path: str) -> pd.DataFrame:
    """Загружает xlsx-файл со списком классов."""
    # Загружаем полный список классов:
    df = pd.read_excel(file_path, engine='openpyxl')

    # Отбрасываем столбцы, чьи имена не заданы явно:
    df = df.drop(columns=[column for column in df.columns if 'Unnamed: ' in column])

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Задаём первый столбец в качестве индекса:
    df = df.set_index(df.columns[0])

    # Отбрасывание пустых строк:
    return df[~df.index.isna()]


def _file2superlabels_df(file_path: str) -> pd.DataFrame:
    """Загружает xlsx-файл со списком суперклассов."""
    # Читаем список суперклассов:
    df = pd.read_excel(file_path, engine='openpyxl')

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Заполняем пропуски:
    msgs = []  # Список ошибок при чтении
    for ind, dfrow in enumerate(df.iloc):
        # Считываем текущие значения в строке:
        cur_superlabel_name = dfrow[superlabel_column]  # Имя
        cur_scl_number = dfrow[scl_number_column]  # Номер
        cur_scl_priority = dfrow[scl_prrity_column]  # Приоритет

        # Если cur_superlabel_name не NaN, значит это новый класс:
        if pd.notna(cur_superlabel_name):
            # Остальные параметры тоже должны быть не NaN:
            if pd.isna(cur_scl_number):
                msg = f'[{ind}, {superlabel_column}] = NaN'
                msgs.append(msg)
            if pd.isna(cur_scl_priority):
                msg = f'[{ind}, {scl_prrity_column}] = NaN'
                msgs.append(msg)

            # Читаем из строки действительные значения для текущего
            # суперкласса:
            superlabel_name = cur_superlabel_name  # Имя
            scl_number = cur_scl_number  # Номер
            scl_priority = cur_scl_priority  # Приоритет

        # Если cur_superlabel_name = NaN, тострока не дозаполнена:
        else:
            # Остальные параметры тоже должны быть NaN:
            if pd.notna(cur_scl_number):
                msg = f'[{ind}, {superlabel_column}] = {cur_scl_number}'
                msgs.append(msg)
            if pd.notna(cur_scl_priority):
                msg = f'[{ind}, {scl_prrity_column}] = {cur_scl_priority}'
                msgs.append(msg)

            # Пишем в строку пропущенные значения для текущего суперкласса:
            df.loc[ind, superlabel_column] = superlabel_name  # Имя
            df.loc[ind, scl_number_column] = scl_number  # Номер
            df.loc[ind, scl_prrity_column] = scl_priority  # Приоритет

    # Если найдена хоть одна проблема - возвращаем ошибку:
    if msgs:
        raise ValueError(msgs.join('\n'))

    # Приводим номера суперклассов к целочислоенному типу и сдвигаем, чтобы
    # суперкласс неиспользуемых объектов был под номером -1, а остальные
    # начинались с 0:
    df[scl_number_column] = df[scl_number_column].apply(int) - 1
    # В результате исключённые объекты получат значение -2 !

    return df


def _any_file2df(file_path: str) -> (pd.DataFrame, str):
    """Читает файл классов или суперклассов.

    Возвращает датафейрм и тип сожержимого (labels / superlabels).
    """
    if not Path(file_path).is_file():
        msg = f'Файла "{file_path}" не существует!'
        raise FileNotFoundError(msg)

    exceptions = []
    for func, type_ in [
        (_file2superlabels_df, 'superlabels'),  # Суперклассы
        (_file2labels_df, 'labels'),  # Классы
    ]:
        try:
            return func(file_path), type_
        except Exception as exception:  # noqa: BLE001
            exceptions.append(exception)

    msg = f'"{file_path}" не является файлом классов или суперклассов!'
    raise ExceptionGroup(
        msg,
        exceptions,
    )


def _labels_df2tree(df: pd.DataFrame) -> Tree:
    """Парсит датафрейм суперклассов.

    Строит дерево данных из pandas-dataframe в индексах которого прописаны
    номера списков с вложенностью.
    """
    # Создаём дерево и указываем корень:
    tree = Tree()
    tree.create_node('Номер класса', 'Номер класса')

    # Перебор по всем строкам таблицы:
    for df_ind in df.index:
        # Имена класса для текущей строки:
        label = df[label_column][df_ind]
        synonym = df[synonym_column][df_ind]

        # Если параметр не заполнен, то в дерево надо будет вносить None:
        if pd.isna(label):
            label = None
        if pd.isna(synonym):
            synonym = None

        # Объединение меток
        label = (label, synonym)

        # Разделяем строку ind на индекс (позицию во вложенных списках) и
        # расшифровку класса:

        # Заменяем нетипичные пробелы на типичные:
        ind = df_ind.replace('\xa0', ' ')
        # Расщепляем строку на слова:
        words = [word for word in ind.strip().split(' ') if word]
        ind = words[0]  # Первое слово является позицией в списках
        name = ' '.join(words[1:])  # Остальные слова расшифровывают класс
        if ind[-1] == '.':  # В конце индекса должна стоять точка:
            ind = ind[:-1]  # Отбрасываем её
        else:
            msg = f'Нехватает точки в конце строки "{ind}"!'
            raise ValueError(msg)

        # Парсим строку индекса и вносим данные в нужную ветку дерева:

        # Если индекс записан римскими цифрами:
        if ind[0] not in '0123456789':
            # Определяем ветку в дереве:
            group = rim2arabic(ind)

            # Вносим в дерево новые данные:
            tree.create_node(name, (group,), parent='Номер класса', data=label)

        # Если индекс записан арабскими цифрами::
        else:
            # Определяем ветку и листок в дереве:
            ind = (group, *map(int, ind.split('.')))
            parent = tuple(ind[:-1])

            # Вносим в дерево новые данные:
            tree.create_node(name, ind, parent=parent, data=label)

    return tree


def _check_meanin_missmatch(label: str, meaning1: str, meaning2: str) -> None:
    """Возвращает ошибку, если расшифровки не совпадают.

    Используется в _make_labels2meanings.
    """
    if meaning1 != meaning2:
        error_str = (
            f'Для метки "{label}" встретились'
            'следующие несовпадающие расшифровки:\n'
            f'"{meaning1}" и "{meaning2}"!'
        )
        raise KeyError(error_str)


def _make_labels2meanings(tree: Tree) -> (dict, dict):
    """Формируем словари перехода от меток к их расшифровкам."""
    labels2meanings = {}  # Label   -> расшифровка
    synonyms2meanings = {}  # Synonym -> расшифровка

    # Перебираем все строки таблицы классов:
    for node in tree.expand_tree(mode=Tree.DEPTH):
        # Пропускаем классы, не имеющие меток:
        if tree[node].data in (None, (None, None)):
            continue

        # Считываем параметры класса
        meaning = tree[node].tag  # Расшифровка
        label, synonym = tree[node].data  # Метки

        # Вносим существующие метки в cvat-словарь:
        if label is not None:
            # Перевод в нижний регистр:
            label = label.lower()

            # Если такая метка уже встречалась:
            if label in labels2meanings:
                # Выводим ошибку, если текущая расщифровка не совпадает с
                # предыдущей:
                cur_meaning = labels2meanings[label]
                _check_meanin_missmatch(label, cur_meaning, meaning)

            # Добавляем метку, если она не встречалась:
            else:
                labels2meanings[label] = meaning

        # Вносим существующие метки в gg-словарь:
        if synonym is not None:
            # Перевод в нижний регистр:
            synonym = synonym.lower()

            # Если такая метка уже встречалась:
            if synonym in synonyms2meanings:
                # Выводим ошибку, если текущая расщифровка не совпадает с
                # предыдущей:
                cur_meaning = synonyms2meanings[synonym]
                _check_meanin_missmatch(synonym, cur_meaning, meaning)

            # Добавляем метку, если она не встречалась:
            else:
                synonyms2meanings[synonym] = meaning

    return labels2meanings, synonyms2meanings


def _make_meanings2superlabels2superinds(
    superlabels_df: pd.DataFrame,
) -> (dict, dict):
    """Парсит датафрейм суперклассов.

    Строит из датафрейма словарь перехода от имени класса к имени
    соответствующего суперкласса.
    """
    # Заполняемые словари:
    meanings2superlabels = {}
    superlabels2superinds = {}

    # Перебираем все строки датафрейма суперклассов:
    for row in superlabels_df.iloc:
        # Берём из строки нужные данные:
        meaning = row[scl_clsnme_column]
        superlabel = row[superlabel_column]
        superind = row[scl_number_column]

        # Если такая расшифровка уже внесена в словарь:
        if meaning in meanings2superlabels:
            msg = (
                f'В таблице суперклассов расшифровка "{meaning}"'
                ' встречается минимум дважды!'
            )
            raise KeyError(msg)

        # Если такой расшифровки ещё нет, то вносим запись:

        # При этом отрицательные расшифровки заменяем на None:
        meanings2superlabels[meaning] = superlabel
        # Нужно, чтобы суперкласс неиспользуемых объектов имел None вместо
        # своего имени.

        # Если такой индекс уже есть, проверяем совпадение:
        if superlabel in superlabels2superinds:
            old_superind = superlabels2superinds[superlabel]
            if old_superind != superind:
                msg = (
                    'Противоречивые записи в суперклассе '
                    f'"{superlabel}": {old_superind} != {superind}!'
                )
                raise KeyError(msg)

        # Если такго индекса ещё нет, то вносим запись:
        else:
            superlabels2superinds[superlabel] = superind

    return meanings2superlabels, superlabels2superinds


class CoreLabelsConvertor(dict):
    """Ядро класс-утилиты для замены имён меток в разметке.

    Наследует словарь, обеспечивая методами, позволяющими выполнять работу с
    метками. Концептуально предназначен только для интерфейсных функций
    главного словаря.Экземпляр класса представляет собой только главный
    словарь. Главным словарём называется тот, по которому производится замена
    меток.
    """

    def __init__(self, main_dict: dict) -> None:
        """Инициация словарём."""
        self |= main_dict

    def __call__(self, label: str | int | None) -> str | int | None:
        """Возвращает новую метку в соответствии с основным словарём."""
        return self[label]

    def apply2df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет конвертор ко всем меткам в датафрейме."""
        # Делаем копию исходного датафрейма, чтобы не менять оригинал:
        df = df.copy()

        # Замета меток на номера суперклассов:
        df['label'] = df['label'].map(self)

        return df

    def asdict(self) -> dict:
        """Возвращает сам словарь."""
        return dict(self)

    def get_unknown_labels(self, labels: list | tuple | set | pd.DataFrame) -> set:
        """Возвращает множество "неизвестных" словарю меток.

        Может исопльзоваться для проверки применимости словаря:
        ```
        unknown_labels = lc.get_unknown_labels(labels)
        if unknown_labels:
            raise KeyError(f"Неизвестные метки: {unknown_labels}!")
        ```
        """
        # Преобразуем набор меток лбого типа во множество:
        if isinstance(labels, pd.DataFrame):
            labels = set(labels['label'].unique())
        elif isinstance(labels, (list, tuple)):
            labels = set(labels)
        elif isinstance(labels, set):
            pass
        else:
            msg = f'Неподдерживаемый тип набора меток: {type(labels)}!'
            raise TypeError(msg)

        # Получаем множество неизвестных меток:
        return labels - set(self)


class ForbiddenLabelError(Exception):
    """Исключение, возвращаемое в случае обнаружения запретной метки."""

    def __init__(self, msg: str = 'обнаружены запретные метки') -> Exception:
        """Инициализация исключения."""
        self.msg = msg

    def __str__(self) -> str:
        """Сообщение об ошибке."""
        return f'ForbiddenLabelError: {self.msg}!'


class LabelsConvertor(CoreLabelsConvertor):
    """Класс-утилита для замены имён меток в разметке.

    Наследует CoreLabelsConvertor, позволяя формировать главный словарь более
    сложными способами.

    Используется следующая внутренняя терминология:
        label      - оригинальное название класа;
        meaning    - его расшифровка (может быть на русском языке);
        superlabel - название суперкласса (может включать в себя
                     несколько классов);
        superind   - номер суперкласса.

    Полная цепочка перехода: label -> meaning -> superlabel -> superind.
    Предпологается, что переход "label -> meaning" взаимнооднозначен (может
    быть обращён), а "meaning -> superlabel" - наоборот - позволяет
    "схлопывать" часть классов в суперклассы. По факту могут быть любые
    варианты.

    Главный словарь может быть выбран одним из нескольких вариантов:
        "auto"             - использует масимально возможную часть полной
                             цепочки перехода (многие варианты инициализации
                             объекта предологают лишь частичную определённость
                             полной цепочки);
        "label2superind"   - от класса к номеру суперкласса (YOLO-формат);
        "label2superlabel" - от класса к суперклассу (например, для замены
                             меток после авторазметки).

    Некоторые суперклассы могут быть:
        неиспользуемыми - их superind = -1, соотвествующие объекты должны
                          исключаться из разметки в итоговой выборке при
                          конвертации датасета;
        запретными      - их superind = -2, кадры, содержащие эти объекты,
                          подлежат исключению из итоговой выборки при
                          конвертации.
    """

    # "Тип перехода -> соответствующий словарь"
    # в порядке убывания приоритета:
    main_dict_name2attrib: ClassVar[dict] = {
        # label -> meaning -> superlabel -> superind:
        'label2superind': 'labels2superinds',
        # label -> meaning -> superlabel            :
        'label2superlabel': 'labels2superlabels',
        #          meaning -> superlabel -> superind:
        'meaning2superind': 'meaning2superinds',
        # label -> meaning                          :
        'label2meaning': 'labels2meanings',
        #          meaning -> superlabel            :
        'meaning2superlabel': 'meanings2superlabels',
        #                     superlabel -> superind:
        'superlabel2superind': 'superlabels2superinds',
    }

    def _read_files(
        self,
        labels2meanings: str | dict,
        meanings2superlabels: str | dict | None,
    ) -> None:
        """Пытается читать файлы."""
        if isinstance(labels2meanings, str):
            df, type_ = _any_file2df(labels2meanings)
            if type_ == 'labels':
                self.labels_df = df
            elif type_ == 'superlabels':
                self.superlabels_df = df
            else:
                msg = f'Неизвестный тип файла: {type_}!'
                raise NotImplementedError(msg)

        if isinstance(meanings2superlabels, str):
            df, type_ = _any_file2df(meanings2superlabels)
            if type_ == 'labels':
                if hasattr(self, 'labels_df'):
                    msg = 'Передано два файла классов!'
                    raise ValueError(msg)
                self.labels_df = df
            elif type_ == 'superlabels':
                if hasattr(self, 'superlabels_df'):
                    msg = 'Передано два файла суперклассов!'
                    raise ValueError(msg)
                self.superlabels_df = df
            else:
                msg = f'Неизвестный тип файла: {type_}!'
                raise NotImplementedError(msg)

    def _read_dicts(
        self,
        labels2meanings: str | dict,
        meanings2superlabels: str | dict | None,
    ) -> None:
        """Пытается читать словари."""
        if isinstance(labels2meanings, dict):
            if hasattr(self, 'labels_df'):
                self.meanings2superlabels = labels2meanings
            else:
                self.labels2meanings = labels2meanings

        if isinstance(meanings2superlabels, dict):
            if hasattr(self, 'superlabels_df'):
                self.labels2meanings = meanings2superlabels
            else:
                self.meanings2superlabels = meanings2superlabels

    def _parse_dfs(self) -> None:
        """Пытается парсить датафреймы."""
        if hasattr(self, 'labels_df'):
            tree = _labels_df2tree(self.labels_df)
            labels2meanings, synonyms2meanings = _make_labels2meanings(tree)

            self.tree = tree
            self._labels2meanings = labels2meanings
            self._synonyms2meanings = synonyms2meanings

            # Объединяем словари меток и их синонимов:
            self.labels2meanings = labels2meanings | synonyms2meanings

        if hasattr(self, 'superlabels_df'):
            meanings2superlabels, superlabels2superinds = (
                _make_meanings2superlabels2superinds(self.superlabels_df)
            )
            self.meanings2superlabels = meanings2superlabels
            self.superlabels2superinds = superlabels2superinds

    def _build_dicts(self) -> None:
        """Строит всевозможные словари перехода."""
        if hasattr(self, 'labels2meanings') and hasattr(self, 'meanings2superlabels'):
            self.labels2superlabels = {}
            for label, meaning in self.labels2meanings.items():
                meanings2slabels = self.meanings2superlabels  # Краткое имя
                if meaning in meanings2slabels:
                    superlabel = meanings2slabels[meaning]
                    self.labels2superlabels[label] = superlabel

            if hasattr(self, 'superlabels2superinds'):
                self.labels2superinds = {}
                for label, superlabel in self.labels2superlabels.items():
                    slabels2sinds = self.superlabels2superinds  # Краткое имя
                    if superlabel in slabels2sinds:
                        superind = slabels2sinds[superlabel]
                        self.labels2superinds[label] = superind

    def __init__(
        self,
        labels2meanings: str | dict,
        meanings2superlabels: str | dict | None = None,
        main_dict: str = 'auto',
    ) -> None:
        """Создание конвертора.

        файл/словарь labels2meanings должен представлять собой переход
        label -> meaning, а meanings2superlabels - переход
        meaning -> superlabel.
        Если meanings2superlabels не задан, то labels2meanings - это переход
        label -> superlabel.

        Варианты инициализации:
            # Схлопывает классы "человек" и "толпа" в суперкласс "Люди", а
            # "Машина" и "Самолёт" в "Транспорт":
            lc = LabelsConvertor(
                {
                    'person': 'people',
                    'crowd': 'people',
                    'car': 'transport',
                    'plane': 'transport'
                }
            )

            # Тот же вариант с промежуточной расшифровкой:
            lc = LabelsConvertor(
                {
                    'person': 'человек',
                    'crowd': 'толпа',
                    'car': 'машина',
                    'plane': 'самолёт'
                },
                {
                    'человек': 'people',
                    'толпа': 'people',
                    'машина': 'transport',
                    'самолёт': 'transport'
                },
            )

            # Загрузка данных из Excel-файлов:
            lc = LabelsConvertor('labels.xlsx', 'superlabels.xlsx')

            # или просто:
            lc = LabelsConvertor('superlabels.xlsx')

        # Примеры содержимого Excel-файлов представлены файлами
        # labels_template.xlsx и superlabels_template.xlsx.

        main_dict отвечает за поведение экземпляра класса при использовании его
        как функции от одного аргумента:
            'label2superind'   - переход от метки класса к номеру суперкласса
                                 (конвертация в YOLO-формат);
            'label2superlabel' - переход от метки класса к метке суперкласса
                                 (подмена одних меток другими, например, при
                                 обработке результатов авторазметки);
             'auto'            - переход, задейсвтующий все переданные данные.
                                 Начало и конец зависят от того, какие данные
                                 переданы. В пределе это - полная цепочка:
                                 label -> meaning -> superlabel -> superind.
        """
        # Сначала читаем указанные файлы:
        self._read_files(labels2meanings, meanings2superlabels)

        # Теперь разбираемся со словарями:
        self._read_dicts(labels2meanings, meanings2superlabels)

        # Наконец парсим все датафреймы, полученные из файлов:
        self._parse_dfs()

        # Пытаемся построить словари полного перехода
        # label/synonym -> meaning -> superlabel[ -> superind]:
        self._build_dicts()

        # Устанавливаем основной словарь:
        self.main_dict = main_dict

    def _get_auto_dict_name(self) -> str:
        """Определяет переход, захватывающий всю доступную часть цепочки."""
        for main_dict_name, attrib in self.main_dict_name2attrib.items():
            if hasattr(self, attrib):
                return main_dict_name
        msg = 'Не найдено подходящих методов!'
        raise NotImplementedError(msg)

    @property
    def main_dict(self) -> str:
        """Возвращает аттрибут main_dict.

        Указывает тип поведения экземпляра класса в случае вызова как функции.
        """
        return self._main_dict_name

    @main_dict.setter
    def main_dict(self, main_dict_name: str) -> None:
        """Ставит аттрибут main_dict.

        Меняет поведения экземпляра класса в случае вызова как функции.
        """
        # Определяем главный словарь автоматически, если надо:
        if main_dict_name == 'auto':
            main_dict_name = self._get_auto_dict_name()

        # Находим имя соответствующего словаря:
        attrib = self.main_dict_name2attrib[main_dict_name]

        # Очищаем главный словарь:
        self.clear()

        # Если нужный словарь имеется в арсенале, то копируем его в главный:
        if hasattr(self, attrib):
            self |= getattr(self, attrib)
            self._main_dict_name = main_dict_name

        else:
            msg = f'Неподдерживаемый переход: {main_dict_name}"!'
            raise NotImplementedError(
                msg,
            )

    @staticmethod
    def _iterable2set(obj: object) -> set | None:
        """Переводит любой объект во множество.

        Если объект - не список, кортеж или множество, то он берётся целиком.
        Используется в df_convertor.
        """
        if isinstance(obj, set):  # set[objs] -> set[objs]
            return obj
        if isinstance(obj, (list, tuple)):  # (list | tuple)[objs] -> set[objs]
            return set(obj)
        if obj is None:  # None -> None
            return None
        return {obj}  # obj -> set[obj}

    @staticmethod
    def _process_labels2del(df: pd.DataFrame, labels2del: set) -> pd.DataFrame:
        """Удаляет строки датафрема, содержащие неиспользуемые объекты."""
        return df[~df['label'].isin(labels2del)]

    @staticmethod
    def _process_labels2raise(df: pd.DataFrame, labels2raise: set) -> pd.DataFrame:
        """Возвращает исключение, если в датафрейме есть запрещённые объекты."""
        problem_frames = df[df['label'].isin(labels2raise)]['frame'].unique()
        if len(problem_frames):
            msg = (
                'В кадрах {'
                + ', '.join(map(str, sorted(problem_frames)))
                + '} найдены запрещённые объекты'
            )
            raise ForbiddenLabelError(msg)
        return df

    def df_convertor(
        self,
        labels2del: list | tuple | set | object | None = None,
        labels2raise: list | tuple | set | object | None = None,
    ) -> Callable:
        """Создаёт умный функтор обработки датафреймов.

        Отличается от apply2df тем, что, в случае меток из набора labels2del,
        соотвествующая запись из датафрейма выкидывается, а при обнаружении хоть
        одной метки из labels2raise - возвращается ошибка ForbiddenLabelError.
        """
        # Адаптируем типы входных данных:
        labels2del = self._iterable2set(labels2del)
        labels2raise = self._iterable2set(labels2raise)

        # Формируем нужную функцию:
        if labels2del:
            # Если надо обрабатывать и неиспользуемые, и запрещённые объекты:
            if labels2raise:

                def apply2df(df: pd.DataFrame) -> pd.DataFrame:
                    df = self.apply2df(df)
                    df = self._process_labels2raise(df, labels2raise)
                    return self._process_labels2del(df, labels2del)

            # Если надо обрабатывать только неиспользуемые объекты:
            else:

                def apply2df(df: pd.DataFrame) -> pd.DataFrame:
                    df = self.apply2df(df)
                    return self._process_labels2del(df, labels2del)

        # Если надо обрабатывать только запрещённые объекты:
        elif labels2raise:

            def apply2df(df: pd.DataFrame) -> pd.DataFrame:
                df = self.apply2df(df)
                return self._process_labels2raise(df, labels2raise)

        # Если нужна лишь сама конвертация меток, без постобработок:
        else:
            apply2df = self.apply2df

        return apply2df
