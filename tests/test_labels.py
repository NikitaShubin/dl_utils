"""Тесты для labels.py.

Модуль содержит тесты для классов LabelsConvertor и ForbiddenLabelError.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from labels import ForbiddenLabelError, LabelsConvertor

# ============================================================================
# Тесты для вспомогательных функций (через публичный интерфейс)
# ============================================================================


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    @pytest.fixture
    def labels_csv_with_commas(self) -> Iterator[str]:
        """Создаёт временный CSV-файл с запятыми в значениях."""
        csv_content = (
            'Класс объекта,Метка в CVAT,Метка в другом источнике данных,Признаки\n'
            'I. Транспорт,,,\n'
            '1. Машина,car,"машина, автомобиль",\n'
            'II. Люди,,,\n'
            '1. Человек,person,"человек, люди",'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    @pytest.fixture
    def labels_tsv_file(self) -> Iterator[str]:
        """Создаёт временный TSV-файл с метками."""
        tsv_content = (
            'Класс объекта\tМетка в CVAT\tМетка в другом источнике данных\t'
            'Признаки\tURL\n'
            'I. Транспорт\t\t\t\t\n'
            '1. Машина\tcar\t\t\t\n'
            'II. Люди\t\t\t\t\n'
            '1. Человек\tperson\t\t\t'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.tsv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(tsv_content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    @pytest.fixture
    def labels_csv_cp1251(self) -> Iterator[str]:
        """Создаёт временный CSV-файл в кодировке cp1251."""
        content = (
            'Класс объекта,Метка в CVAT,Метка в другом источнике данных,Признаки\n'
            'I. Транспорт,,,\n'
            '1. Машина,car,автомобиль,\n'
            'II. Люди,,,\n'
            '1. Человек,person,человек,'
        )

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            f.write(content.encode('cp1251'))
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    def test_csv_with_commas_in_values(self, labels_csv_with_commas: str) -> None:
        """Тест чтения CSV-файла с запятыми в значениях ячеек."""
        lc = LabelsConvertor(labels_csv_with_commas)
        assert lc.main_dict == 'label2meaning'
        assert lc('person') == 'Человек'
        assert lc('car') == 'Машина'

    def test_tsv_reading(self, labels_tsv_file: str) -> None:
        """Тест чтения TSV-файла."""
        lc = LabelsConvertor(labels_tsv_file)
        assert lc.main_dict == 'label2meaning'
        assert lc('person') == 'Человек'
        assert lc('car') == 'Машина'

    def test_csv_different_encoding(self, labels_csv_cp1251: str) -> None:
        """Тест чтения CSV-файла в кодировке cp1251."""
        lc = LabelsConvertor(labels_csv_cp1251)
        assert lc.main_dict == 'label2meaning'
        assert lc('person') == 'Человек'
        assert lc('car') == 'Машина'

    def test_unsupported_format_error(self) -> None:
        """Тест ошибки при неподдерживаемом формате файла."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name

        try:
            with pytest.raises(
                ExceptionGroup,
                match='не является файлом меток или суперметок',
            ):
                LabelsConvertor(temp_path)
        finally:
            Path(temp_path).unlink()


# ============================================================================
# Тесты для ForbiddenLabelError
# ============================================================================


class TestForbiddenLabelError:
    """Тесты для класса ForbiddenLabelError."""

    def test_forbidden_label_error_init(self) -> None:
        """Тест инициализации ForbiddenLabelError."""
        # С пользовательским сообщением
        error = ForbiddenLabelError('обнаружены запретные метки в кадрах 1, 2, 3')
        assert error.msg == 'обнаружены запретные метки в кадрах 1, 2, 3'

        # С сообщением по умолчанию
        error = ForbiddenLabelError()
        assert error.msg == 'обнаружены запретные метки'

    def test_forbidden_label_error_str(self) -> None:
        """Тест строкового представления ForbiddenLabelError."""
        error = ForbiddenLabelError('тестовое сообщение')
        assert str(error) == 'ForbiddenLabelError: тестовое сообщение!'

        error = ForbiddenLabelError()
        assert str(error) == 'ForbiddenLabelError: обнаружены запретные метки!'

    def test_forbidden_label_error_raise(self) -> None:
        """Тест возбуждения ForbiddenLabelError."""
        error_message = 'кастомное сообщение'

        with pytest.raises(ForbiddenLabelError) as exc_info:
            raise ForbiddenLabelError(error_message)

        assert str(exc_info.value) == f'ForbiddenLabelError: {error_message}!'


# ============================================================================
# Тесты для CoreLabelsConvertor
# ============================================================================


class TestCoreLabelsConvertor:
    """Тесты для базового класса CoreLabelsConvertor."""

    @pytest.fixture
    def simple_dict(self) -> dict[str, str]:
        """Фикстура с простым словарём преобразования."""
        return {'a': 'A', 'b': 'B', 'c': 'C'}

    @pytest.fixture
    def core_converter(self, simple_dict: dict[str, str]) -> LabelsConvertor:
        """Фикстура с экземпляром CoreLabelsConvertor."""
        # Используем LabelsConvertor для создания CoreLabelsConvertor
        return LabelsConvertor(simple_dict, main_dict='label2meaning')

    def test_call_method(
        self,
        core_converter: LabelsConvertor,
        simple_dict: dict[str, str],
    ) -> None:
        """Тест вызова конвертора как функции."""
        for key, value in simple_dict.items():
            assert core_converter(key) == value

    def test_apply2df(self, core_converter: LabelsConvertor) -> None:
        """Тест применения конвертора к DataFrame."""
        df = pd.DataFrame(
            {
                'label': ['a', 'b', 'c', 'a'],
                'frame': [1, 2, 3, 4],
                'value': [10, 20, 30, 40],
            },
        )

        result = core_converter.apply2df(df)
        assert result['label'].tolist() == ['A', 'B', 'C', 'A']
        assert result['frame'].tolist() == [1, 2, 3, 4]
        assert result['value'].tolist() == [10, 20, 30, 40]

    def test_apply2df_with_values2del(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к DataFrame с удалением меток."""
        converter = LabelsConvertor(
            simple_dict,
            main_dict='label2meaning',
            values2del='A',
        )

        df = pd.DataFrame(
            {
                'label': ['a', 'b', 'c', 'a'],
                'frame': [1, 2, 3, 4],
            },
        )

        result = converter.apply2df(df)
        assert len(result) == 2  # 'a' -> 'A' удалено (2 раза)
        assert 'A' not in result['label'].to_numpy()
        assert set(result['label'].unique()) == {'B', 'C'}

    def test_apply2df_with_values2raise(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к DataFrame с проверкой запрещённых меток."""
        converter = LabelsConvertor(
            simple_dict,
            main_dict='label2meaning',
            values2raise='A',
        )

        df = pd.DataFrame(
            {
                'label': ['a', 'b', 'c'],
                'frame': [1, 2, 3],
            },
        )

        with pytest.raises(ForbiddenLabelError) as exc_info:
            converter.apply2df(df)

        assert 'В кадрах {1} найдены запрещённые объекты' in str(exc_info.value)

    def test_apply2objs(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к списку объектов."""

        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        converter = LabelsConvertor(simple_dict, main_dict='label2meaning')

        objs = [MockObject('a'), MockObject('b'), MockObject('c')]
        result = converter.apply2objs(objs)

        assert [obj.attribs['label'] for obj in result] == ['A', 'B', 'C']

    def test_apply2objs_with_values2del(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к объектам с удалением меток."""

        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        converter = LabelsConvertor(
            simple_dict,
            main_dict='label2meaning',
            values2del='A',
        )

        objs = [MockObject('a'), MockObject('b'), MockObject('c'), MockObject('a')]
        result = converter.apply2objs(objs)

        assert len(result) == 2  # 'a' -> 'A' удалено (2 раза)
        assert all(obj.attribs['label'] in ['B', 'C'] for obj in result)

    def test_apply2objs_with_values2raise(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к объектам с проверкой запрещённых меток."""

        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        converter = LabelsConvertor(
            simple_dict,
            main_dict='label2meaning',
            values2raise='A',
        )

        objs = [MockObject('a'), MockObject('b'), MockObject('c')]

        with pytest.raises(ForbiddenLabelError) as exc_info:
            converter.apply2objs(objs)

        assert 'объекты с номерами {0} имеют запрещённые метки' in str(exc_info.value)

    def test_apply2objs_invalid_objects(self, simple_dict: dict[str, str]) -> None:
        """Тест применения конвертора к некорректным объектам."""
        converter = LabelsConvertor(simple_dict, main_dict='label2meaning')

        # Объект без атрибута attribs
        class BadObject1:
            pass

        # Объект без ключа 'label' в attribs
        class BadObject2:
            def __init__(self) -> None:
                self.attribs = {'not_label': 'value'}

        objs1 = [BadObject1()]
        with pytest.raises(TypeError) as exc_info:
            converter.apply2objs(objs1)
        assert 'не имеют поля "attribs"' in str(exc_info.value)

        objs2 = [BadObject2()]
        with pytest.raises(KeyError) as exc_info:
            converter.apply2objs(objs2)
        assert 'не имеют ключа "label" в словаре attribs' in str(exc_info.value)

    def test_asdict(self, simple_dict: dict[str, str]) -> None:
        """Тест получения словаря."""
        converter = LabelsConvertor(simple_dict, main_dict='label2meaning')
        result_dict = converter.asdict()

        assert isinstance(result_dict, dict)
        assert result_dict == simple_dict

    def test_get_unknown_labels(self, simple_dict: dict[str, str]) -> None:
        """Тест получения неизвестных меток."""
        converter = LabelsConvertor(simple_dict, main_dict='label2meaning')

        # Тест с списком меток:
        unknown = converter.get_unknown_labels(['a', 'unknown', 'b'])
        assert unknown == {'unknown'}

        # Тест с множеством меток:
        unknown = converter.get_unknown_labels({'unknown1', 'unknown2'})
        assert unknown == {'unknown1', 'unknown2'}

        # Тест с DataFrame:
        df = pd.DataFrame({'label': ['a', 'unknown']})
        unknown = converter.get_unknown_labels(df)
        assert unknown == {'unknown'}

        # Тест с неподдерживаемым типом:
        with pytest.raises(TypeError):
            converter.get_unknown_labels(123)


# ============================================================================
# Тесты для LabelsConvertor
# ============================================================================


class TestLabelsConvertor:
    """Тесты для класса LabelsConvertor."""

    @pytest.fixture
    def labels2meanings(self) -> dict[str, str]:
        """Фикстура с метками и их расшифровками."""
        return {
            'domain': 'Домен',
            'kingdom': 'Царство',
            'phylum': 'Отдел',
            'class': 'Класс',
            'order': 'Порядок (отряд)',
            'family': 'Семейство',
            'genus': 'Род',
            'species': 'Вид',
        }

    @pytest.fixture
    def meanings2superlabels(self) -> dict[str, str]:
        """Фикстура с расшифровки и суперклассами."""
        return {
            'Род': 'Виды и рода',
            'Вид': 'Виды и рода',
            'Класс': 'Классы, отряды и семейства',
            'Порядок (отряд)': 'Классы, отряды и семейства',
            'Семейство': 'Классы, отряды и семейства',
            'Отдел': 'Неиспользуемые объекты',
            'Домен': 'Исключаемые объекты',
            'Царство': 'Другая категория',
        }

    @pytest.fixture
    def labels_csv_file(self) -> Iterator[str]:
        """Создаёт временный CSV-файл с метками."""
        csv_content = (
            'Класс объекта,Метка в CVAT,Метка в другом источнике данных,'
            'Признаки,URL\n'
            'I. Транспорт,,,,\n'
            '1. Машина,car,,,\n'
            'II. Люди,,,,\n'
            '1. Человек,person,,,'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    @pytest.fixture
    def superlabels_csv_file(self) -> Iterator[str]:
        """Создаёт временный CSV-файл с суперметками."""
        # Простой файл суперметок для теста
        csv_content = (
            '№ п/п,Наименование суперкласса,Классы (содержимое суперкласса),'
            'Приоритет\n'
            '1,Человек,Человек,Высокий\n'
            '2,Транспорт,Машина,Высокий'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(csv_content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    @pytest.fixture
    def superlabels_tsv_file(self) -> Iterator[str]:
        """Создаёт временный TSV-файл с суперметками."""
        tsv_content = (
            '№ п/п\tНаименование суперкласса\tКлассы (содержимое суперкласса)\t'
            'Приоритет\n'
            '1\tЧеловек\tЧеловек\tВысокий\n'
            '2\tНаземный транспорт\tМашина\tВысокий\n'
            '\t\tГрузовик\t\n'
            '\t\tАвтобус\t\n'
            '\t\tМотоцикл\t\n'
            '\t\tВелосипед\t\n'
            '3\tВоздушный транспорт\tСамолёт\tВысокий\n'
            '4\tВодный транспорт\tЛодка\tВысокий\n'
            '0\tНеиспользуемые объекты\tРюкзак\tНет\n'
            '\t\tСумка\t\n'
            '\t\tЧемодан\t'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.tsv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(tsv_content)
            temp_path = f.name

        yield temp_path
        Path(temp_path).unlink()

    def test_init_with_both_files(self) -> None:
        """Тест инициализации с обоими файлами (Excel)."""
        lc = LabelsConvertor('labels_template.xlsx', 'superlabels_template.xlsx')
        assert lc.main_dict == 'label2superind'

    def test_init_with_superlabels_file_only(self) -> None:
        """Тест инициализации только с файлом суперклассов (Excel)."""
        lc = LabelsConvertor('superlabels_template.xlsx')
        assert lc.main_dict == 'meaning2superlabel'

    def test_init_with_labels_file_only(self) -> None:
        """Тест инициализации только с файлом классов (Excel)."""
        lc = LabelsConvertor('labels_template.xlsx')
        assert lc.main_dict == 'label2meaning'

    def test_init_with_single_dict(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест инициализации с одним словарём."""
        lc = LabelsConvertor(labels2meanings)
        assert lc.main_dict == 'label2meaning'

    def test_init_with_two_dicts(
        self,
        labels2meanings: dict[str, str],
        meanings2superlabels: dict[str, str],
    ) -> None:
        """Тест инициализации с двумя словарями."""
        lc = LabelsConvertor(labels2meanings, meanings2superlabels)
        assert lc.main_dict == 'label2superlabel'

    def test_init_with_second_dict_only(
        self,
        meanings2superlabels: dict[str, str],
    ) -> None:
        """Тест инициализации только со вторым словарём."""
        lc = LabelsConvertor(meanings2superlabels)
        assert lc.main_dict == 'label2meaning'

    def test_call_behavior(
        self,
        labels2meanings: dict[str, str],
        meanings2superlabels: dict[str, str],
    ) -> None:
        """Тест поведения вызова для разных конфигураций."""
        # Тест 1: label2superind
        lc1 = LabelsConvertor('labels_template.xlsx', 'superlabels_template.xlsx')
        assert lc1('class') == 1

        # Тест 2: meaning2superlabel
        lc2 = LabelsConvertor('superlabels_template.xlsx')
        assert lc2('Класс') == 'Классы, отряды и семейства'

        # Тест 3: label2meaning
        lc3 = LabelsConvertor('labels_template.xlsx')
        assert lc3('class') == 'Класс'
        assert lc3('kingdom') == 'Царство'

        # Тест 4: label2superlabel
        lc4 = LabelsConvertor(labels2meanings, meanings2superlabels)
        assert lc4('class') == 'Классы, отряды и семейства'

    def test_iteration(self) -> None:
        """Тест итерации по конвертору."""
        lc = LabelsConvertor('labels_template.xlsx')

        # Конвертор должен быть итерируемым (как словарь):
        labels = list(lc)
        assert len(labels) > 0
        assert all(isinstance(label, str) for label in labels)

    def test_unknown_label_handling(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест обработки неизвестных меток."""
        lc = LabelsConvertor(labels2meanings)

        # Неизвестная метка должна вызывать KeyError при вызове:
        with pytest.raises(KeyError):
            lc('unknown_label')

    @pytest.mark.parametrize(
        ('main_dict_type', 'expected_type'),
        [
            ('auto', 'label2superlabel'),
            ('label2superlabel', 'label2superlabel'),
            ('label2meaning', 'label2meaning'),
        ],
    )
    def test_main_dict_setter_supported(
        self,
        labels2meanings: dict[str, str],
        meanings2superlabels: dict[str, str],
        main_dict_type: str,
        expected_type: str,
    ) -> None:
        """Тест установки поддерживаемых типов основного словаря."""
        lc = LabelsConvertor(labels2meanings, meanings2superlabels)

        lc.main_dict = main_dict_type
        assert lc.main_dict == expected_type

    def test_main_dict_setter_unsupported(
        self,
        labels2meanings: dict[str, str],
        meanings2superlabels: dict[str, str],
    ) -> None:
        """Тест установки неподдерживаемых типов основного словаря."""
        lc = LabelsConvertor(labels2meanings, meanings2superlabels)

        # label2superind не поддерживается в этой конфигурации:
        with pytest.raises(NotImplementedError):
            lc.main_dict = 'label2superind'

    def test_invalid_main_dict_type(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест установки неподдерживаемого типа основного словаря."""
        lc = LabelsConvertor(labels2meanings)

        with pytest.raises(KeyError):
            lc.main_dict = 'invalid_type'

    def test_init_with_csv_files(
        self,
        labels_csv_file: str,
    ) -> None:
        """Тест инициализации с CSV-файлами."""
        # Создаем упрощенный файл суперметок для теста
        simple_superlabels_csv = (
            '№ п/п,Наименование суперкласса,Классы (содержимое суперкласса),'
            'Приоритет\n'
            '1,Человек,Человек,Высокий\n'
            '2,Транспорт,Машина,Высокий'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(simple_superlabels_csv)
            temp_path = f.name

        try:
            lc = LabelsConvertor(labels_csv_file, temp_path)
            assert lc.main_dict == 'label2superind'
            # Проверяем преобразование меток
            # Человек -> суперкласс 1, но после сдвига -1 = 0
            assert lc('person') == 0
            # Транспорт -> суперкласс 2, после сдвига -1 = 1
            assert lc('car') == 1
        finally:
            Path(temp_path).unlink()

    def test_init_with_csv_labels_only(self, labels_csv_file: str) -> None:
        """Тест инициализации только с CSV-файлом меток."""
        lc = LabelsConvertor(labels_csv_file)
        assert lc.main_dict == 'label2meaning'

        # Проверяем преобразование меток в расшифровки
        assert lc('person') == 'Человек'
        assert lc('car') == 'Машина'

    def test_init_with_csv_superlabels_only(self, superlabels_csv_file: str) -> None:
        """Тест инициализации только с CSV-файлом суперметок."""
        lc = LabelsConvertor(superlabels_csv_file)
        assert lc.main_dict == 'meaning2superlabel'

        # Проверяем преобразование расшифрок в суперметки
        assert lc('Человек') == 'Человек'
        assert lc('Машина') == 'Транспорт'

    def test_init_with_mixed_formats_csv_tsv(
        self,
        labels_csv_file: str,
    ) -> None:
        """Тест инициализации с файлами разных форматов (CSV + TSV)."""
        # Создаем упрощенный файл суперметок для теста
        simple_superlabels_tsv = (
            '№ п/п\tНаименование суперкласса\tКлассы (содержимое суперкласса)\t'
            'Приоритет\n'
            '1\tЧеловек\tЧеловек\tВысокий\n'
            '2\tТранспорт\tМашина\tВысокий'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.tsv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(simple_superlabels_tsv)
            temp_path = f.name

        try:
            lc = LabelsConvertor(labels_csv_file, temp_path)
            assert lc.main_dict == 'label2superind'
            # Проверяем преобразование меток
            assert lc('person') == 0
            assert lc('car') == 1
        finally:
            Path(temp_path).unlink()

    def test_init_with_mixed_formats_excel_csv(self) -> None:
        """Тест инициализации с Excel и CSV файлами."""
        # Используем существующий Excel-шаблон и создаём CSV
        csv_content = (
            '№ п/п,Наименование суперкласса,Классы (содержимое суперкласса),'
            'Приоритет\n'
            '1,Человек,Человек,Высокий\n'
            '2,Транспорт,Машина,Высокий'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            lc = LabelsConvertor('labels_template.xlsx', temp_path)
            assert lc.main_dict == 'label2superind'
        finally:
            Path(temp_path).unlink()

    @pytest.mark.parametrize('format_suffix', ['.xlsx', '.csv', '.tsv'])
    def test_multiple_formats_support(self, format_suffix: str, tmp_path: Path) -> None:
        """Тест поддержки нескольких форматов файлов."""
        # Создаём тестовые файлы в разных форматах
        labels_data = pd.DataFrame(
            {
                'Класс объекта': ['I. Транспорт', '1. Машина'],
                'Метка в CVAT': ['', 'car'],
                'Метка в другом источнике данных': ['', ''],
                'Признаки': ['', ''],
                'URL': ['', ''],
            },
        )

        labels_file = tmp_path / f'labels{format_suffix}'

        if format_suffix == '.xlsx':
            labels_data.to_excel(labels_file, index=False)
        elif format_suffix in ['.csv', '.tsv']:
            sep = '\t' if format_suffix == '.tsv' else ','
            labels_data.to_csv(labels_file, index=False, sep=sep)

        # Проверяем что файл читается в любом формате
        lc = LabelsConvertor(str(labels_file))
        assert lc.main_dict == 'label2meaning'
        assert lc('car') == 'Машина'

    def test_file_not_found_error(self) -> None:
        """Тест ошибки при отсутствии файла."""
        with pytest.raises(FileNotFoundError, match='не существует'):
            LabelsConvertor('non_existent_file.xlsx')

    def test_invalid_file_content(self) -> None:
        """Тест ошибки при некорректном содержимом файла."""
        # Создаем файл с правильными столбцами, но некорректными данными
        # В данном случае - без точки в индексе, что вызовет ValueError
        invalid_content = (
            'Класс объекта,Метка в CVAT,Метка в другом источнике данных,'
            'Признаки\n'
            'некорректные,данные,без,индексов'
        )

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False,
            encoding='utf-8',
        ) as f:
            f.write(invalid_content)
            temp_path = f.name

        try:
            # Файл будет прочитан как labels, но при построении дерева будет ошибка
            # из-за отсутствия точки в индексе, поэтому ожидаем ValueError
            with pytest.raises(ValueError, match='Нехватает точки в конце строки'):
                LabelsConvertor(temp_path)
        finally:
            Path(temp_path).unlink()
