"""Тесты для labels.py."""

import pandas as pd
import pytest

from labels import LabelsConvertor


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
