"""Тесты для labels.py.

Модуль содержит тесты для классов LabelsConvertor и ForbiddenLabelError.
"""

import pandas as pd
import pytest

from labels import ForbiddenLabelError, LabelsConvertor


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

    def test_init_with_both_files(self) -> None:
        """Тест инициализации с обоими файлами."""
        lc = LabelsConvertor('labels_template.xlsx', 'superlabels_template.xlsx')
        assert lc.main_dict == 'label2superind'

    def test_init_with_superlabels_file_only(self) -> None:
        """Тест инициализации только с файлом суперклассов."""
        lc = LabelsConvertor('superlabels_template.xlsx')
        assert lc.main_dict == 'meaning2superlabel'

    def test_init_with_labels_file_only(self) -> None:
        """Тест инициализации только с файлом классов."""
        lc = LabelsConvertor('labels_template.xlsx')
        assert lc.main_dict == 'label2meaning'

    def test_init_with_single_dict(self, labels2meanings: dict[str, str]) -> None:
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
        self, meanings2superlabels: dict[str, str]
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

    def test_get_unknown_labels(self, labels2meanings: dict[str, str]) -> None:
        """Тест получения неизвестных меток."""
        lc = LabelsConvertor(labels2meanings)

        # Тест с списком меток:
        unknown = lc.get_unknown_labels(['domain', 'unknown_label', 'class'])
        assert unknown == {'unknown_label'}

        # Тест с множеством меток:
        unknown = lc.get_unknown_labels({'unknown1', 'unknown2'})
        assert unknown == {'unknown1', 'unknown2'}

        # Тест с DataFrame:
        df = pd.DataFrame({'label': ['domain', 'unknown_label']})
        unknown = lc.get_unknown_labels(df)
        assert unknown == {'unknown_label'}

    def test_apply2df(self, labels2meanings: dict[str, str]) -> None:
        """Тест применения конвертора к DataFrame."""
        lc = LabelsConvertor(labels2meanings)

        df = pd.DataFrame(
            {'label': ['domain', 'class', 'kingdom'], 'other_column': [1, 2, 3]}
        )

        result_df = lc.apply2df(df)

        # Проверяем, что метки были преобразованы:
        expected_labels = ['Домен', 'Класс', 'Царство']
        assert result_df['label'].tolist() == expected_labels

        # Другие колонки не должны меняться:
        assert result_df['other_column'].tolist() == [1, 2, 3]

    def test_asdict(self, labels2meanings: dict[str, str]) -> None:
        """Тест получения словаря."""
        lc = LabelsConvertor(labels2meanings)
        result_dict = lc.asdict()

        assert isinstance(result_dict, dict)
        assert result_dict == dict(lc)

    def test_iteration(self) -> None:
        """Тест итерации по конвертору."""
        lc = LabelsConvertor('labels_template.xlsx')

        # Конвертор должен быть итерируемым (как словарь):
        labels = list(lc)
        assert len(labels) > 0
        assert all(isinstance(label, str) for label in labels)

    def test_unknown_label_handling(self, labels2meanings: dict[str, str]) -> None:
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

    def test_invalid_main_dict_type(self, labels2meanings: dict[str, str]) -> None:
        """Тест установки неподдерживаемого типа основного словаря."""
        lc = LabelsConvertor(labels2meanings)

        with pytest.raises(KeyError):
            lc.main_dict = 'invalid_type'

    def test_apply2df_with_values2del(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест применения конвертора к DataFrame с удалением меток."""
        lc = LabelsConvertor(
            labels2meanings,
            main_dict='label2meaning',
            values2del='Отдел',
        )

        df = pd.DataFrame(
            {
                'label': ['domain', 'class', 'kingdom', 'phylum'],
                'frame': [1, 2, 3, 4],
                'other_column': [1, 2, 3, 4],
            }
        )

        result_df = lc.apply2df(df)

        # 'phylum' -> 'Отдел' должно быть удалено
        assert len(result_df) == 3
        assert 'Отдел' not in result_df['label'].to_numpy()
        assert set(result_df['label'].unique()) == {'Домен', 'Класс', 'Царство'}

    def test_apply2df_with_values2raise(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест применения конвертора к DataFrame с проверкой запрещенных меток."""
        lc = LabelsConvertor(
            labels2meanings,
            main_dict='label2meaning',
            values2raise='Домен',
        )

        df = pd.DataFrame(
            {
                'label': ['domain', 'class', 'kingdom'],
                'frame': [1, 2, 3],
                'other_column': [1, 2, 3],
            }
        )

        with pytest.raises(ForbiddenLabelError) as exc_info:
            lc.apply2df(df)

        assert 'В кадрах {1} найдены запрещённые объекты' in str(exc_info.value)

    def test_apply2df_with_both_values2del_and_values2raise(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест применения конвертора с удалением и проверкой меток."""
        lc = LabelsConvertor(
            labels2meanings,
            main_dict='label2meaning',
            values2del='Отдел',
            values2raise='Домен',
        )

        df = pd.DataFrame(
            {
                'label': ['domain', 'class', 'kingdom', 'phylum'],
                'frame': [1, 2, 3, 4],
                'other_column': [1, 2, 3, 4],
            }
        )

        with pytest.raises(ForbiddenLabelError) as exc_info:
            lc.apply2df(df)

        assert 'В кадрах {1} найдены запрещённые объекты' in str(exc_info.value)

    def test_apply2objs(self, labels2meanings: dict[str, str]) -> None:
        """Тест применения конвертора к списку объектов."""

        # Создаем простой класс для тестирования
        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        lc = LabelsConvertor(labels2meanings)

        objs = [MockObject('domain'), MockObject('class'), MockObject('kingdom')]

        result_objs = lc.apply2objs(objs)

        # Проверяем, что метки были преобразованы:
        expected_labels = ['Домен', 'Класс', 'Царство']
        assert [obj.attribs['label'] for obj in result_objs] == expected_labels

    def test_apply2objs_with_values2del(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест применения конвертора к объектам с удалением меток."""

        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        lc = LabelsConvertor(labels2meanings, values2del='Отдел')

        objs = [
            MockObject('domain'),
            MockObject('class'),
            MockObject('kingdom'),
            MockObject('phylum'),
        ]

        result_objs = lc.apply2objs(objs)

        # 'phylum' -> 'Отдел' должно быть удалено
        assert len(result_objs) == 3
        result_labels = [obj.attribs['label'] for obj in result_objs]
        assert set(result_labels) == {'Домен', 'Класс', 'Царство'}

    def test_apply2objs_with_values2raise(
        self,
        labels2meanings: dict[str, str],
    ) -> None:
        """Тест применения конвертора к объектам с проверкой запрещенных меток."""

        class MockObject:
            def __init__(self, label: str) -> None:
                self.attribs = {'label': label}

            def copy(self) -> 'MockObject':
                return MockObject(self.attribs['label'])

        lc = LabelsConvertor(labels2meanings, values2raise='Домен')

        objs = [MockObject('domain'), MockObject('class'), MockObject('kingdom')]

        with pytest.raises(ForbiddenLabelError) as exc_info:
            lc.apply2objs(objs)

        assert 'объекты с номерами {0} имеют запрещённые метки' in str(exc_info.value)

    def test_apply2objs_invalid_objects(self, labels2meanings: dict[str, str]) -> None:
        """Тест применения конвертора к некорректным объектам."""
        lc = LabelsConvertor(labels2meanings)

        # Объект без атрибута attribs
        class BadObject1:
            pass

        # Объект без ключа 'label' в attribs
        class BadObject2:
            def __init__(self) -> None:
                self.attribs = {'not_label': 'value'}

        objs1 = [BadObject1()]
        with pytest.raises(TypeError) as exc_info:
            lc.apply2objs(objs1)
        assert 'не имеют поля "attribs"' in str(exc_info.value)

        objs2 = [BadObject2()]
        with pytest.raises(KeyError) as exc_info:
            lc.apply2objs(objs2)
        assert 'не имеют ключа "label" в словаре attribs' in str(exc_info.value)


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
