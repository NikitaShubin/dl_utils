"""labesl.py
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
"""


import os

from typing import Any

import pandas as pd

from treelib import Tree

from utils import rim2arabic

# Имена столбцов в labels.xlsx:
label_column = "Метка в CVAT"                       # Имя класса (метка)
synonym_column = "Метка в другом источнике данных"  # Синоним класса

# Имена столбцов в superlabels.xlsx:
superlabel_column = "Наименование суперкласса"         # Имена суперклассов
scl_clsnme_column = "Классы (содержимое суперкласса)"  # Имена классов
scl_number_column = "№ п/п"      # Номера суперклассов
scl_prrity_column = "Приоритет"  # Приоритет суперклассов


def _fix_string(s: str) -> str:
    """Заменяет неправильные пробелы на правильные и убирает пробелы в начале и
    конце.
    """
    return s.replace("\xa0", " ").strip() if isinstance(s, str) else s


def _file2labels_df(file: str) -> pd.DataFrame:
    """Загружает xlsx-файл со списком классов."""
    # Загружаем полный список классов:
    df = pd.read_excel(file, engine="openpyxl")

    # Отбрасываем столбцы, чьи имена не заданы явно:
    df = df.drop(columns=[column for column in df.columns
                          if "Unnamed: " in column])

    # Подчищаем данные в таблице:
    for column in df.columns:
        df[column] = df[column].apply(_fix_string)

    # Задаём первый столбец в качестве индекса:
    df = df.set_index(df.columns[0])

    # Отбрасывание пустых строк:
    df = df[~df.index.isna()]

    return df


def _file2superlabels_df(file: str) -> pd.DataFrame:
    """Загружает xlsx-файл со списком суперклассов."""
    # Читаем список суперклассов:
    df = pd.read_excel(file, engine="openpyxl")

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
        if pd.notna(cur_superlabel_name):

            # Остальные параметры тоже должны быть не NaN:
            if pd.isna(cur_scl_number):
                raise ValueError(cur_scl_number)
            if pd.isna(cur_scl_priority):
                raise ValueError(cur_scl_priority)

            # Читаем из строки действительные значения для текущего
            # суперкласса:
            superlabel_name = cur_superlabel_name  # Имя
            scl_number = cur_scl_number            # Номер
            scl_priority = cur_scl_priority        # Приоритет

        # Если cur_superlabel_name = NaN, тострока не дозаполнена:
        else:
            # Остальные параметры тоже должны быть NaN:
            if pd.notna(cur_scl_number):
                raise ValueError(cur_scl_number)
            if pd.notna(cur_scl_priority):
                raise ValueError(cur_scl_priority)

            # Пишем в строку пропущенные значения для текущего суперкласса:
            df.loc[ind, superlabel_column] = cur_superlabel_name  # Имя
            df.loc[ind, scl_number_column] = cur_scl_number       # Номер
            df.loc[ind, scl_prrity_column] = cur_scl_priority     # Приоритет

    # Приводим номера суперклассов к целочислоенному типу и сдвигаем, чтобы
    # суперкласс неиспользуемых объектов был под номером -1, а остальные
    # начинались с 0:
    df[scl_number_column] = df[scl_number_column].apply(int) - 1
    # В результате исключённые объекты получат значение -2 !

    return df


def _any_file2df(file: str) -> (pd.DataFrame, str):
    """Читает файл классов или суперклассов.
    Возвращает датафейрм и тип сожержимого (labels / superlabels).
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f'Файла "{file}" не существует!')

    exceptions = []
    for func, type_ in [
        (_file2superlabels_df, "superlabels"),  # Суперклассы
        (_file2labels_df, "labels"),             # Классы
    ]:
        try:
            return func(file), type_
        except Exception as e:
            exceptions.append(e)

    raise ExceptionGroup(
        f'"{file}" не является файлом классов или суперклассов!',
        exceptions,
    )


def _labels_df2tree(df: pd.DataFrame) -> Tree:
    """Строит дерево данных из pandas-dataframe в индексах которого прописаны
    номера списков с вложенностью.
    """
    # Создаём дерево и указываем корень:
    tree = Tree()
    tree.create_node("Номер класса", "Номер класса")

    # Перебор по всем строкам таблицы:
    for ind in df.index:

        # Имена класса для текущей строки:
        label = df[label_column][ind]
        synonym = df[synonym_column][ind]

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
        ind = ind.replace("\xa0", " ")
        # Расщепляем строку на слова:
        words = [word for word in ind.strip().split(" ") if word]
        ind = words[0]              # Первое слово является позицией в списках
        name = " ".join(words[1:])  # Остальные слова расшифровывают класс
        assert (ind[-1] == ".")     # В конце индекса должна стоять точка
        ind = ind[:-1]              # Её отбрасываем

        # Парсим строку индекса и вносим данные в нужную ветку дерева:

        # Если индекс с арабскими цифрами:
        if ind[0] in "0123456789":
            # Определяем ветку и листок в дереве:
            ind = (group, *map(int, ind.split(".")))
            parent = tuple(ind[:-1])

            # Вносим в дерево новые данные:
            tree.create_node(name, ind, parent=parent, data=label)

        # Если индекс записан римскими цифрами:
        else:
            # Определяем ветку в дереве:
            group = rim2arabic(ind)

            # Вносим в дерево новые данные:
            tree.create_node(name, (group,),
                             parent="Номер класса", data=label)

    return tree


def _make_labels2meanings(tree: Tree) -> (dict, dict):
    """Формируем словари перехода от меток к их расшифровкам."""
    labels2meanings = {}    #   Label -> расшифровка
    synonyms2meanings = {}  # Synonym -> расшифровка

    # Перебираем все строки таблицы классов:
    for node in tree.expand_tree(mode=Tree.DEPTH):

        # Пропускаем классы, не имеющие меток:
        if tree[node].data in (None, (None, None)):
            continue

        # Считываем параметры класса
        meaning = tree[node].tag          # Расшифровка
        label, synonym = tree[node].data  # Метки

        # Вносим существующие метки в cvat-словарь:
        if label is not None:

            # Перевод в нижний регистр:
            label = label.lower()

            # Если такая метка уже встречалась:
            if label in labels2meanings:

                # Выводим ошибку, если текущая расщифровка не совпадает с
                # предыдущей:
                if labels2meanings[label] != meaning:
                    error_str = \
                        f'Для метки "{label}" встретились' \
                        'следующие несовпадающие расшифровки:\n' \
                        f'"{labels2meanings[label]}" и "{meaning}"!'
                    raise KeyError(error_str)

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
                if synonyms2meanings[synonym] != meaning:
                    error_str = f'Для метки "{synonym}" встретились ' \
                        'следующие несовпадающие расшифровки:\n' \
                        f'"{synonyms2meanings[synonym]}" и "{meaning}"!'
                    raise KeyError(error_str)

            # Добавляем метку, если она не встречалась:
            else:
                synonyms2meanings[synonym] = meaning

    return labels2meanings, synonyms2meanings


def _make_meanings2superlabels2superinds(superlabels_df: pd.DataFrame) -> \
        (dict, dict):
    """Строит из датафрейма словарь перехода от имени класса к имени
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
            raise KeyError(f'В таблице суперклассов расшифровка "{meaning}"'
                           ' встречается минимум дважды!')

        # Если такой расшифровки ещё нет, то вносим запись:

        # При этом отрицательные расшифровки заменяем на None:
        meanings2superlabels[meaning] = superlabel
        # Нужно, чтобы суперкласс неиспользуемых объектов имел None вместо
        # своего имени.

        # Если такой индекс уже есть, проверяем совпадение:
        if superlabel in superlabels2superinds:
            old_superind = superlabels2superinds[superlabel]
            if old_superind != superind:
                raise KeyError("Противоречивые записи в суперклассе "
                               f'"{superlabel}": '
                               f"{old_superind} != {superind}!")

        # Если такго индекса ещё нет, то вносим запись:
        else:
            superlabels2superinds[superlabel] = superind

    return meanings2superlabels, superlabels2superinds


class LabelsConvertor:
    """Класс-утилита для работы с классами (labels),
    суперклассами (superlabels) и т.п.

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

    Чаще всего экземпляр класса используется как переход:
        "end2end"          - использует масимально возможную часть полной
                             цепочки перехода (многие варианты инициализации
                             объекта предологают лишь частичную определённость
                             полной цепочки);
        "label2superind"   - от класса к номеру суперкласса (YOLO-формат);
        "label2superlabel" - от класса к суперклассу (например, для замены
                             меток после авторазметки).

    Некоторые суперклассы могут быть:
        неиспользуемыми - их superind = -1, соотвествующие объекты должны
                          исклюаться из разметки в итоговой выборке при
                          конвертации датасета;
        запретными      - их superind = -2, кадры, содержащие эти объекты,
                          подлежат исключению из итоговой выборки при
                          конвертации.
    """

    # "Тип перехода -> соответствующий словарь"
    # в порядке убывания приоритета:
    on_call2method = {
        # label -> meaning -> superlabel -> superind:
        "label2superind": "labels2superinds",
        # label -> meaning -> superlabel            :
        "label2superlabel": "labels2superlabels",
        #          meaning -> superlabel -> superind:
        "meaning2superind": "meaning2superinds",
        # label -> meaning                          :
        "label2meaning": "labels2meanings",
        #          meaning -> superlabel            :
        "meaning2superlabel": "meanings2superlabels",
        #                     superlabel -> superind:
        "superlabel2superind": "superlabels2superinds",
    }

    def _read_files(self,
                    labels2meanings: str | dict,
                    meanings2superlabels: str | dict | None):
        """Пытается читать файлы"""
        if isinstance(labels2meanings, str):
            df, type_ = _any_file2df(labels2meanings)
            if type_ == "labels":
                self.labels_df = df
            elif type_ == "superlabels":
                self.superlabels_df = df
            else:
                raise NotImplementedError(f"Неизвестный тип файла: {type_}!")

        if isinstance(meanings2superlabels, str):
            df, type_ = _any_file2df(meanings2superlabels)
            if type_ == "labels":
                if hasattr(self, "labels_df"):
                    raise ValueError("Передано два файла классов!")
                self.labels_df = df
            elif type_ == "superlabels":
                if hasattr(self, "superlabels_df"):
                    raise ValueError("Передано два файла суперклассов!")
                self.superlabels_df = df
            else:
                raise NotImplementedError(f"Неизвестный тип файла: {type_}!")

    def _read_dicts(self,
                    labels2meanings: str | dict,
                    meanings2superlabels: str | dict | None):
        """Пытается читать словари."""
        if isinstance(labels2meanings, dict):
            if hasattr(self, "labels_df"):
                self.meanings2superlabels = labels2meanings
            else:
                self.labels2meanings = labels2meanings

        if isinstance(meanings2superlabels, dict):
            if hasattr(self, "superlabels_df"):
                self.labels2meanings = meanings2superlabels
            else:
                self.meanings2superlabels = meanings2superlabels

    def _parse_dfs(self):
        """Пытается парсить датафреймы."""
        if hasattr(self, "labels_df"):
            tree = _labels_df2tree(self.labels_df)
            _labels2meanings, _synonyms2meanings = _make_labels2meanings(tree)

            # Объединяем словари меток и их синонимов:
            labels2meanings = _labels2meanings | _synonyms2meanings

            self.tree = tree
            self._labels2meanings = _labels2meanings
            self._synonyms2meanings = _synonyms2meanings
            self.labels2meanings = labels2meanings

        if hasattr(self, "superlabels_df"):
            self.meanings2superlabels, self.superlabels2superinds = \
                _make_meanings2superlabels2superinds(self.superlabels_df)

    def _build_dicts(self):
        """Строит всевозможные словари перехода."""
        if hasattr(self, "labels2meanings") and \
                hasattr(self, "meanings2superlabels"):
            self.labels2superlabels = {}
            for label, meaning in self.labels2meanings.items():
                if meaning in self.meanings2superlabels:
                    superlabel = self.meanings2superlabels[meaning]
                    self.labels2superlabels[label] = superlabel

            if hasattr(self, "superlabels2superinds"):
                self.labels2superinds = {}
                for label, superlabel in self.labels2superlabels.items():
                    if superlabel in self.superlabels2superinds:
                        superind = self.superlabels2superinds[superlabel]
                        self.labels2superinds[label] = superind

    def __init__(self,
                 labels2meanings: str | dict,
                 meanings2superlabels: str | dict | None = None,
                 on_call: str = "end2end"):
        """файл/словарь labels2meanings должен представлять собой переход
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

        on_call отвечает за поведение экземпляра класса при использовании его
        как функции от одного аргумента:
            'label2superind'   - переход от метки класса к номеру суперкласса
                                 (конвертация в YOLO-формат);
            'label2superlabel' - переход от метки класса к метке суперкласса
                                 (подмена одних меток другими, например, при
                                 обработке результатов авторазметки);
             'end2end'         - переход, задейсвтующий все переданные данные.
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

        # Устанавливаем функцию вызова:
        self.set_call(on_call)

    def _get_end2end(self) -> str:
        """Определяет переход, захватывающий всю доступную часть цепочки."""
        for on_call, method in self.on_call2method.items():
            if hasattr(self, method):
                return on_call
        raise Exception("Не найдено подходящих методов!")

    def set_call(self, on_call: str):
        """Меняет метод поведения экземпляра класса в случае вызова как
        функции.
        """
        if on_call == "end2end":
            on_call = self._get_end2end()

        method = self.on_call2method[on_call]

        if hasattr(self, method):
            self.call = getattr(self, method).__getitem__
        else:
            raise NotImplementedError("Неподдерживаемый переход: "
                                      f'"{on_call}"!')

        # Фиксируем текущий тип поведения:
        self.on_call = on_call

    def __call__(self, label: Any) -> Any:
        """Возвращает новую метку в соответствии с on_call."""
        return self.call(label)

    def apply2df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применяет конвертор ко всем меткам в датафрейме."""
        # Делаем копию исходного датафрейма, чтобы не менять оригинал:
        df = df.copy()

        # Замета меток на номера суперклассов:
        df["label"] = df["label"].apply(self)

        return df
