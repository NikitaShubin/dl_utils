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

from pathlib import Path
from typing import ClassVar, cast

import pandas as pd
from treelib import Tree

from utils import rim2arabic

# Задаём тип меток и их наборов:
Label = str | int | None
Labels = list[Label] | tuple[Label] | set[Label]

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
    return df[df.index.notna()]


def _file2superlabels_df(file_path: str) -> pd.DataFrame:
    """Загружает xlsx-файл со списком суперклассов."""
    # Читаем список суперклассов:
    df = pd.read_excel(file_path, engine='openpyxl')

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Заполняем пропуски:
    msgs: list[str] = []  # Список ошибок при чтении
    for ind, dfrow_tuple in enumerate(df.iterrows()):
        # Распаковываем кортеж:
        _, dfrow = dfrow_tuple

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
        raise ValueError('\n'.join(msgs))

    # Приводим номера суперклассов к целочислоенному типу и сдвигаем, чтобы
    # суперкласс неиспользуемых объектов был под номером -1, а остальные
    # начинались с 0:
    df[scl_number_column] = df[scl_number_column].apply(int) - 1
    # В результате исключённые объекты получат значение -2 !

    return df


def _any_file2df(file_path: str) -> tuple[pd.DataFrame, str]:
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


def _make_labels2meanings(
    tree: Tree,
) -> tuple[dict[Label, str], dict[Label, str]]:
    """Формируем словари перехода от меток к их расшифровкам."""
    labels2meanings: dict[Label, str] = {}  # Label   -> расшифровка
    synonyms2meanings: dict[Label, str] = {}  # Synonym -> расшифровка

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


def _make_meanings_2_superlabels2del_and_2raise(
    superlabels_df: pd.DataFrame,
) -> tuple[set[Label], set[Label]]:
    """Парсит датафрейм суперклассов.

    Строит множество неиспользуемых и игнорируемых суперклассов.
    """

    def scl_number2superlabels(scl_number: int) -> set:
        """Извлекает множество суперклассов с заданным индексом."""
        mask = superlabels_df[scl_number_column] == scl_number
        return set(superlabels_df[mask][superlabel_column].unique())

    # Множества неиспользуемых и запрещённых суперклассов:
    return scl_number2superlabels(-1), scl_number2superlabels(-2)


def _make_meanings2superlabels2superinds(
    superlabels_df: pd.DataFrame,
) -> tuple[dict[str, Label], dict[Label, int]]:
    """Парсит датафрейм суперклассов.

    Строит из датафрейма словарь перехода от имени класса к имени
    соответствующего суперкласса.
    """
    # Заполняемые словари:
    meanings2superlabels: dict[str, Label] = {}
    superlabels2superinds: dict[Label, int] = {}

    # Перебираем все строки датафрейма суперклассов:
    for dfrow_tuple in superlabels_df.iterrows():
        # Распаковываем кортеж:
        _, dfrow = dfrow_tuple

        # Берём из строки нужные данные:
        meaning = str(dfrow[scl_clsnme_column])
        superlabel = cast('Label', dfrow[superlabel_column])
        superind = int(dfrow[scl_number_column])

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


class ForbiddenLabelError(Exception):
    """Исключение, возвращаемое в случае обнаружения запретной метки."""

    def __init__(self, msg: str = 'обнаружены запретные метки') -> None:
        """Инициализация исключения."""
        self.msg = msg

    def __str__(self) -> str:
        """Сообщение об ошибке."""
        return f'ForbiddenLabelError: {self.msg}!'


class CoreLabelsConvertor(dict):
    """Ядро класс-утилиты для замены имён меток в разметке.

    Наследует словарь, обеспечивая методами, позволяющими выполнять работу с
    метками. Концептуально предназначен только для интерфейсных функций
    главного словаря.Экземпляр класса представляет собой только главный
    словарь. Главным словарём называется тот, по которому производится замена
    меток.
    """

    def __init__(
        self,
        main_dict: dict[Label, Label],
        values2del: set | None = None,
        values2raise: set | None = None,
    ) -> None:
        """Инициация словарём."""
        if values2raise is None:
            values2raise = set()
        if values2del is None:
            values2del = set()
        self |= main_dict
        self.values2del = values2del
        self.values2raise = values2raise

    def __call__(self, label: Label) -> Label:
        """Возвращает новую метку в соответствии с основным словарём."""
        return self[label]

    def apply2df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет конвертор ко всем меткам в датафрейме."""
        # Делаем копию исходного датафрейма, чтобы не менять оригинал:
        df = df.copy()

        # Замета меток на номера суперклассов:
        labels = df['label'].map(self)
        df['label'] = labels

        # Ищем запрещённые объекты, если надо:
        if self.values2raise:
            problem_frames = df[labels.isin(self.values2raise)]['frame'].unique()
            if problem_frames:
                msg = (
                    'В кадрах {'
                    + ', '.join(map(str, sorted(problem_frames)))
                    + '} найдены запрещённые объекты'
                )
                raise ForbiddenLabelError(msg)

        # Удаляем ненужные объекты, если надо:
        if self.values2del:
            df = df[~labels.isin(self.values2del)]

        return df

    def apply2objs(self, objs: list | tuple) -> list:
        """Выполняет замену меток в списках объектов.

        Применяется к спискам экземлпяров таких классов, как BBox и Mask из модуля
        cv_utils.py, но работает со списками и кортежами любых объектов, если метки
        хранятся у них в obj.attribs['label'], а сами они поддерживают метод copy().
        """
        # Проверка наличия нужного атрибута в объектах:
        no_attribs_inds = [
            ind for ind, obj in enumerate(objs) if not hasattr(obj, 'attribs')
        ]
        if no_attribs_inds:
            msg = (
                'Объекты с номерами {'
                + ', '.join(map(str, no_attribs_inds))
                + '} не имеют поля "attribs"!'
            )
            raise TypeError(msg)

        # Проверка наличия нужного ключа в атрибутах:
        no_label_inds = [
            ind for ind, obj in enumerate(objs) if 'label' not in obj.attribs
        ]
        if no_label_inds:
            msg = (
                'Объекты с номерами {'
                + ', '.join(map(str, no_label_inds))
                + '} не имеют ключа "label" в словаре attribs!'
            )
            raise KeyError(msg)

        new_objs = []  # Итоговый список объектов
        error_inds = []  # Индексы объектов с запрещённой меткой
        for ind, obj in enumerate(objs):
            # Новая метка:
            value = self(obj.attribs['label'])

            # Вносим индекс в список номеров запрещённых объектов, если надо:
            if value in self.values2raise:
                error_inds.append(ind)

            # Пополняем итоговый список, если метка не в списке неиспользуемых:
            elif value not in self.values2del:
                new_obj = obj.copy()
                new_obj.attribs['label'] = value
                new_objs.append(new_obj)

        if error_inds:
            msg = (
                'объекты с номерами {'
                + ', '.join(map(str, error_inds))
                + '} имеют запрещённые метки'
            )
            raise ForbiddenLabelError(msg)

        return new_objs

    def asdict(self) -> dict[Label, Label]:
        """Возвращает сам словарь."""
        return dict(self)

    def get_unknown_labels(self, labels: Labels | pd.DataFrame) -> set:
        """Возвращает множество "неизвестных" словарю меток.

        Может исопльзоваться для проверки применимости словаря:
            unknown_labels = lc.get_unknown_labels(labels)
            if unknown_labels:
                raise KeyError(f"Неизвестные метки: {unknown_labels}!")
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
        labels2meanings: str | dict[Label, str],
        meanings2superlabels: str | dict[str, Label] | None,
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
        labels2meanings: str | dict[Label, str] | dict[str, Label],
        meanings2superlabels: str | dict[Label, str] | dict[str, Label] | None,
    ) -> None:
        """Пытается читать словари."""
        if isinstance(labels2meanings, dict):
            if hasattr(self, 'labels_df'):
                self.meanings2superlabels = cast('dict[str, Label]', labels2meanings)
            else:
                self.labels2meanings = cast('dict[Label, str]', labels2meanings)

        if isinstance(meanings2superlabels, dict):
            if hasattr(self, 'superlabels_df'):
                self.labels2meanings = cast('dict[Label, str]', meanings2superlabels)
            else:
                self.meanings2superlabels = cast(
                    'dict[str, Label]', meanings2superlabels
                )

        # Приходится явно указывать то, какие типы будут у meanings2superlabels и
        # labels2meanings, т.к. mypy эту логику отследить не может.

    def _parse_dfs(self) -> None:
        """Пытается парсить датафреймы."""
        # Работаем с датафреймом меток:
        if hasattr(self, 'labels_df'):
            tree = _labels_df2tree(self.labels_df)
            labels2meanings, synonyms2meanings = _make_labels2meanings(tree)

            self.tree = tree
            self._labels2meanings = labels2meanings
            self._synonyms2meanings = synonyms2meanings

            # Объединяем словари меток и их синонимов:
            self.labels2meanings = labels2meanings | synonyms2meanings

        # Работаем с датафреймом суперметок:
        if hasattr(self, 'superlabels_df'):
            meanings2superlabels, superlabels2superinds = (
                _make_meanings2superlabels2superinds(self.superlabels_df)
            )
            self.meanings2superlabels = meanings2superlabels
            self.superlabels2superinds = superlabels2superinds

            # Определяем суперметки неиспользуемых и запрещённых объектов:
            superlabels2del, superlabels2raise = (
                _make_meanings_2_superlabels2del_and_2raise(self.superlabels_df)
            )
            self.superlabels2del = superlabels2del
            self.superlabels2raise = superlabels2raise

    def _build_dicts(self) -> None:
        """Строит всевозможные словари перехода."""
        if hasattr(self, 'labels2meanings') and hasattr(self, 'meanings2superlabels'):
            self.labels2superlabels = {}
            for label, meaning in self.labels2meanings.items():
                if meaning in self.meanings2superlabels:
                    superlabel = self.meanings2superlabels[meaning]
                    self.labels2superlabels[label] = superlabel

            if hasattr(self, 'superlabels2superinds'):
                self.labels2superinds = {}
                for label, superlabel in self.labels2superlabels.items():
                    if superlabel in self.superlabels2superinds:
                        superind = self.superlabels2superinds[superlabel]
                        self.labels2superinds[label] = superind

    @staticmethod
    def _iterable2set(obj: Label | Labels) -> Labels | None:
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

    def _set_values2del_and_2raise(
        self,
        values2del: Label | Labels = None,
        values2raise: Label | Labels = None,
    ) -> None:
        """Установка значений values2del и values2raise."""
        # Принудительно превращаем наборы во множества:
        values2del = self._iterable2set(values2del)
        values2raise = self._iterable2set(values2raise)

        # Доопределяем набор неиспользуемых значений:
        if isinstance(values2del, set):
            self.values2del = values2del
        elif '2superind' in self._main_dict_name:
            self.values2del = {-1}
        elif '2superlabel' in self._main_dict_name and hasattr(self, 'superlabels2del'):
            self.values2del = self.superlabels2del
        else:
            self.values2del = set()

        # Доопределяем набор запрещённых значений:
        if isinstance(values2raise, set):
            self.values2raise = values2raise
        elif '2superind' in self._main_dict_name:
            self.values2raise = {-2}
        elif '2superlabel' in self._main_dict_name and hasattr(
            self, 'superlabels2raise'
        ):
            self.values2raise = self.superlabels2raise
        else:
            self.values2raise = set()

    def __init__(
        self,
        labels2meanings: str | dict[Label, str],
        meanings2superlabels: str | dict[str, Label] | None = None,
        main_dict: str = 'auto',
        values2del: Label | Labels = None,
        values2raise: Label | Labels = None,
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

        values2del - одно или несколько значений итоговой метки неиспользуемого
        объекта. Т.е. такие объекты пропускаются в случае вызова методов apply2df
        и apply2obj.
        values2raise - одно или несколько значений итоговой метки запрещённого
        объекта. Т.е. при обнаружении таких объектов в случае вызова методов apply2df
        и apply2obj возникает ошибка ForbiddenLabelError.
        values2del и values2raise ИГНОРИРУЮТСЯ в случае использования объекта как
        функтора:
            lc = LabelsConvertor(...)
            new_label = lc(old_label)
            # Новая метка будет возвращена без ошибок, даже если она в списке
            # values2del или values2raise!
        Если значения values2del и values2raise не указаны явно (= None), но при
        инициализации передан xlsx-файл суперметок, то значения читаются оттуда, в
        противном случае списки будут пусты. Чтобы очистить списки даже при заданном
        файле суперклассов, нужно явно указать пустой список:
            lc = LabelsConvertor('superlabels.xlsx', values2del=[], values2raise=[])
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

        # Фиксация наборов нениспользуемых и запрещённых меток:
        self._set_values2del_and_2raise(
            self._iterable2set(values2del), self._iterable2set(values2raise)
        )

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
            raise NotImplementedError(msg)
