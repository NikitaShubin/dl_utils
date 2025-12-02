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
        """Фикстура с расшифровками и суперклассами."""
        return {
            'Род': 'Виды и рода',
            'Вид': 'Виды и рода',
            'Класс': 'Классы, отряды и семейства',
            'Порядок (отряд)': 'Классы, отряды и семейства',
            'Семейство': 'Классы, отряды и семейства',
            'Отдел': 'Неиспользуемые объекты',
            'Домен': 'Исключаемые объекты',
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
        test_cases = [
            # (конвертор, ожидаемые преобразования):
            (
                LabelsConvertor('labels_template.xlsx', 'superlabels_template.xlsx'),
                {'class': 1},  # label2superind
            ),
            (
                LabelsConvertor('superlabels_template.xlsx'),
                {'Класс': 'Классы, отряды и семейства'},  # meaning2superlabel
            ),
            (
                LabelsConvertor('labels_template.xlsx'),
                {'class': 'Класс', 'kingdom': 'Царство'},  # label2meaning
            ),
            (
                LabelsConvertor(labels2meanings, meanings2superlabels),
                {'class': 'Классы, отряды и семейства'},  # label2superlabel
            ),
        ]

        for lc, expected in test_cases:
            for label, expected_result in expected.items():
                assert lc(label) == expected_result

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
        assert result_dict == dict(lc)  # Должен возвращать тот же словарь

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

    # Убираем тесты приватных методов, так как они не должны быть доступны извне
    # Вместо этого тестируем их через публичный интерфейс df_convertor

    def test_df_convertor_no_postprocessing(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor без постобработки (только конвертация)."""
        lc = LabelsConvertor(labels2meanings)

        # Создаем тестовый DataFrame
        df = pd.DataFrame(
            {
                'frame': [1, 2, 3],
                'label': ['domain', 'class', 'kingdom'],
                'other_column': [10, 20, 30],
            }
        )

        # Получаем функтор без обработки удаления/исключений
        convertor_func = lc.df_convertor()

        # Применяем конвертацию
        result = convertor_func(df)

        # Проверяем, что метки преобразованы
        expected_labels = ['Домен', 'Класс', 'Царство']
        assert result['label'].tolist() == expected_labels
        assert result['other_column'].tolist() == [10, 20, 30]

    def test_df_convertor_with_labels2del_only(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor только с удалением меток."""
        lc = LabelsConvertor(labels2meanings)

        # Создаем тестовый DataFrame
        df = pd.DataFrame(
            {
                'frame': [1, 2, 3, 4],
                'label': ['domain', 'class', 'kingdom', 'phylum'],
                'other_column': [10, 20, 30, 40],
            }
        )

        # Получаем функтор с удалением 'Домен' (после конвертации)
        convertor_func = lc.df_convertor(labels2del={'Домен'})

        # Применяем конвертацию
        result = convertor_func(df)

        # Проверяем, что 'domain' (который станет 'Домен') удален
        assert len(result) == 3
        assert set(result['label'].unique()) == {'Класс', 'Царство', 'Отдел'}
        assert result['other_column'].tolist() == [20, 30, 40]

    def test_df_convertor_with_labels2raise_only(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor только с проверкой запрещенных меток."""
        lc = LabelsConvertor(labels2meanings)

        # Создаем тестовый DataFrame
        df = pd.DataFrame(
            {
                'frame': [1, 2, 3],
                'label': ['domain', 'class', 'kingdom'],
                'other_column': [10, 20, 30],
            }
        )

        # Получаем функтор с проверкой на 'Домен'
        convertor_func = lc.df_convertor(labels2raise={'Домен'})

        # Применяем конвертацию - должно вызвать исключение
        with pytest.raises(ForbiddenLabelError) as exc_info:
            convertor_func(df)

        assert 'В кадрах {1} найдены запрещённые объекты' in str(exc_info.value)

        # Тест без запрещенных меток
        df_safe = pd.DataFrame(
            {'frame': [1, 2], 'label': ['class', 'kingdom'], 'other_column': [20, 30]}
        )

        result = convertor_func(df_safe)
        expected_labels = ['Класс', 'Царство']
        assert result['label'].tolist() == expected_labels

    def test_df_convertor_with_both_labels2del_and_labels2raise(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor с удалением одних меток и проверкой других."""
        lc = LabelsConvertor(labels2meanings)

        # Создаем тестовый DataFrame
        df = pd.DataFrame(
            {
                'frame': [1, 2, 3, 4],
                'label': ['domain', 'class', 'kingdom', 'phylum'],
                'other_column': [10, 20, 30, 40],
            }
        )

        # Удаляем 'Отдел', проверяем на 'Домен'
        convertor_func = lc.df_convertor(labels2del={'Отдел'}, labels2raise={'Домен'})

        # Применяем - должно вызвать исключение из-за 'Домен'
        with pytest.raises(ForbiddenLabelError) as exc_info:
            convertor_func(df)

        assert 'В кадрах {1} найдены запрещённые объекты' in str(exc_info.value)

        # Тест без запрещенных меток, но с удалением
        df_safe = pd.DataFrame(
            {
                'frame': [2, 3, 4],
                'label': ['class', 'kingdom', 'phylum'],
                'other_column': [20, 30, 40],
            }
        )

        result = convertor_func(df_safe)
        # 'phylum' -> 'Отдел' должно быть удалено
        assert len(result) == 2
        assert set(result['label'].unique()) == {'Класс', 'Царство'}
        assert result['other_column'].tolist() == [20, 30]

    def test_df_convertor_with_none_values(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor с None значениями в параметрах."""
        lc = LabelsConvertor(labels2meanings)

        df = pd.DataFrame(
            {'frame': [1, 2], 'label': ['domain', 'class'], 'other_column': [10, 20]}
        )

        # Все параметры None - просто конвертация
        convertor_func = lc.df_convertor(None, None)
        result = convertor_func(df)
        assert result['label'].tolist() == ['Домен', 'Класс']

        # Только labels2del = None
        convertor_func = lc.df_convertor(labels2del=None, labels2raise={'Домен'})
        with pytest.raises(ForbiddenLabelError):
            convertor_func(df)

        # Только labels2raise = None
        convertor_func = lc.df_convertor(labels2del={'Домен'}, labels2raise=None)
        result = convertor_func(df)
        assert len(result) == 1
        assert result['label'].tolist() == ['Класс']

    def test_df_convertor_with_various_input_types(
        self, labels2meanings: dict[str, str]
    ) -> None:
        """Тест df_convertor с различными типами входных данных."""
        lc = LabelsConvertor(labels2meanings)

        df = pd.DataFrame(
            {
                'frame': [1, 2, 3],
                'label': ['domain', 'class', 'kingdom'],
                'other_column': [10, 20, 30],
            }
        )

        # Тест с list для labels2del
        convertor_func = lc.df_convertor(labels2del=['Домен', 'Царство'])
        result = convertor_func(df)
        assert len(result) == 1
        assert result['label'].tolist() == ['Класс']

        # Тест с tuple для labels2raise
        convertor_func = lc.df_convertor(labels2raise=('Домен', 'Царство'))
        with pytest.raises(ForbiddenLabelError):
            convertor_func(df)

        # Тест с одиночным значением
        convertor_func = lc.df_convertor(labels2del='Домен')
        result = convertor_func(df)
        assert len(result) == 2
        assert set(result['label'].unique()) == {'Класс', 'Царство'}


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
        msg = 'кастомное сообщение'
        with pytest.raises(ForbiddenLabelError) as exc_info:
            raise ForbiddenLabelError(msg)

        assert str(exc_info.value) == 'ForbiddenLabelError: кастомное сообщение!'

        assert str(exc_info.value) == 'ForbiddenLabelError: кастомное сообщение!'
