# tests/test_yolo.py
"""Тесты для модуля yolo."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from yolo import (  # noqa: E402
    YOLOLabels,
    class_statistic2superclass_statistic,
    df2statistic,
    fill_skipped_rows_in_statistic,
    gen_yaml,
    sources2statistic_and_train_val_test_tasks,
    task2yolo,
    tasks2statistic,
    tasks2yolo,
    yolo_img_exts,
    yolo_vid_exts,
)


# Фикстура для временного файла изображения
@pytest.fixture
def temp_image_file(tmp_path: Path) -> str:
    """Создает временный файл изображения для тестов."""
    img_path = tmp_path / 'test_image.jpg'
    # Создаем простое изображение
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img_array)
    return str(img_path)


class TestYOLOLabels:
    """Тесты для класса YOLOLabels."""

    def test_init_box_mode(self) -> None:
        """Тест инициализации в режиме box."""
        # Создаем тестовый датафрейм с правильным форматом points
        df = pd.DataFrame(
            {
                'label': ['cat', 'dog'],
                'type': ['rectangle', 'rectangle'],
                'points': [
                    [100, 200, 300, 400],
                    [150, 250, 350, 450],
                ],  # Список координат
                'rotation': [0, 0],
                'outside': [False, False],
            }
        )

        # Мокаем CVATPoints
        with patch('yolo.CVATPoints') as mock_cvat:
            mock_instance = Mock()
            mock_instance.yolobbox.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_cvat.return_value = mock_instance

            labels = YOLOLabels(df, mode='box', imsize=(600, 800))
            assert labels.mode == 'box'
            assert labels.imsize == (600, 800)
            assert len(labels.yolo_labels) == 2

    def test_init_seg_mode(self) -> None:
        """Тест инициализации в режиме seg."""
        df = pd.DataFrame(
            {
                'label': ['cat'],
                'type': ['polygon'],
                'points': [[100, 200, 150, 250, 200, 300]],  # Список координат
                'rotation': [0],
                'outside': [False],
            }
        )

        # Мокаем CVATPoints
        with patch('yolo.CVATPoints') as mock_cvat:
            mock_instance = Mock()
            mock_instance.yoloseg.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            mock_cvat.return_value = mock_instance

            labels = YOLOLabels(df, mode='seg', imsize=(600, 800))
            assert labels.mode == 'seg'

    def test_apply_label_func(self) -> None:
        """Тест применения функции к меткам."""
        df = pd.DataFrame(
            {
                'label': ['cat', 'dog'],
                'type': ['rectangle', 'rectangle'],
                'points': [[100, 200, 300, 400], [150, 250, 350, 450]],
                'rotation': [0, 0],
                'outside': [False, False],
            }
        )

        # Мокаем CVATPoints
        with patch('yolo.CVATPoints') as mock_cvat:
            mock_instance = Mock()
            mock_instance.yolobbox.return_value = [0.1, 0.2, 0.3, 0.4]
            mock_cvat.return_value = mock_instance

            labels = YOLOLabels(df, mode='box', imsize=(600, 800))

        # Функция преобразования меток
        def label_func(label: str) -> int:
            return 0 if label == 'cat' else 1

        labels.apply_label_func(label_func)

        # Проверяем преобразованные метки
        label_values = [label for label, _ in labels.yolo_labels]
        assert label_values == [0, 1]

    def test_save_successful(self, tmp_path: Path) -> None:
        """Тест успешного сохранения меток."""
        # Создаем тестовые метки напрямую
        labels = YOLOLabels.__new__(YOLOLabels)  # Создаем экземпляр без вызова __init__
        labels.yolo_labels = [(0, [0.1, 0.2, 0.3, 0.4]), (1, [0.2, 0.3, 0.4, 0.5])]

        file_path = tmp_path / 'test_labels.txt'

        # Используем реальный метод save
        result = labels.save(str(file_path))

        assert result is True
        assert file_path.exists()

        # Проверяем содержимое файла
        content = file_path.read_text()
        assert '0 0.1 0.2 0.3 0.4' in content
        assert '1 0.2 0.3 0.4 0.5' in content

    def test_save_with_excluded_label(self, tmp_path: Path) -> None:
        """Тест сохранения с исключенной меткой (label < -1)."""
        # Создаем тестовые метки напрямую
        labels = YOLOLabels.__new__(YOLOLabels)  # Создаем экземпляр без вызова __init__
        labels.yolo_labels = [(-2, [0.1, 0.2, 0.3, 0.4])]

        file_path = tmp_path / 'test_labels.txt'

        # Используем реальный метод save
        result = labels.save(str(file_path))

        assert result is False
        assert not file_path.exists()

    def test_draw_labels(self) -> None:
        """Тест отрисовки меток на изображении."""
        # Создаем тестовое изображение
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Создаем тестовый объект YOLOLabels с моками
        labels = YOLOLabels.__new__(YOLOLabels)
        labels.mode = 'box'
        labels.yolo_labels = [(0, [0.1, 0.2, 0.3, 0.4]), (1, [0.2, 0.3, 0.4, 0.5])]

        # Мокаем CVATPoints и другие зависимости
        with patch('yolo.CVATPoints') as mock_cvat:
            # Создаем мок для points2draw с методом draw, который возвращает изображение
            mock_points2draw = Mock()
            mock_points2draw.draw.side_effect = lambda img, *_, **__: img

            mock_instance = Mock()
            mock_instance.yolo2cvat.return_value = mock_points2draw
            mock_cvat.return_value = mock_instance

            # Вызываем метод draw_labels с изображением
            result = labels.draw_labels(image)

            assert result.shape == image.shape

            # Проверяем, что draw вызывался несколько раз
            assert mock_points2draw.draw.call_count > 0

    def test_draw_labels_with_args(self) -> None:
        """Тест отрисовки меток с проверкой аргументов."""
        # Создаем тестовое изображение
        image = np.zeros((600, 800, 3), dtype=np.uint8)

        # Создаем тестовый объект YOLOLabels с моками
        labels = YOLOLabels.__new__(YOLOLabels)
        labels.mode = 'box'
        labels.yolo_labels = [(0, [0.1, 0.2, 0.3, 0.4])]

        # Создаем список для отслеживания вызовов
        draw_calls: list[tuple[tuple, dict[str, object]]] = []

        def track_draw_calls(*args: object, **kwargs: object) -> np.ndarray:
            draw_calls.append((args, kwargs))
            return image

        # Мокаем CVATPoints
        with patch('yolo.CVATPoints') as mock_cvat:
            mock_points2draw = Mock()
            mock_points2draw.draw.side_effect = track_draw_calls

            mock_instance = Mock()
            mock_instance.yolo2cvat.return_value = mock_points2draw
            mock_cvat.return_value = mock_instance

            result = labels.draw_labels(image, edge_size=2, alpha=0.3)

            assert result.shape == image.shape

            # Проверяем, что draw вызывался
            assert len(draw_calls) >= 1

            # Первый вызов должен быть для отрисовки контура
            if draw_calls:
                args, kwargs = draw_calls[0]
                # Проверяем, что передан цвет
                assert 'color' in kwargs or len(args) > 1


class TestStatisticFunctions:
    """Тесты для функций статистики."""

    def test_df2statistic(self) -> None:
        """Тест подсчета статистики для датафрейма."""
        # Создаем тестовый датафрейм
        df = pd.DataFrame(
            {
                'label': ['cat', 'cat', 'dog', 'cat'],
                'track_id': [1, 1, 2, 3],  # cat появляется в track_id 1 дважды
                'outside': [False, False, False, False],
            }
        )

        # Создаем мок labels_convertor с правильной структурой
        mock_convertor = Mock()

        # init_df_counter должен возвращать DataFrame с одним столбцом
        img_stat = pd.DataFrame({'shapes': 0}, index=['cat', 'dog'])
        vid_stat = pd.DataFrame({'tracks': 0}, index=['cat', 'dog'])

        # Настраиваем side_effect для двух вызовов
        mock_convertor.init_df_counter.side_effect = [img_stat, vid_stat]
        mock_convertor.any_label2meaning.side_effect = lambda x: x

        result = df2statistic(
            df=df,
            source_type='test',
            labels_convertor=mock_convertor,
            shapes_col_name='shapes',
            tracks_col_name='tracks',
        )

        # Проверяем результаты - используем .item() для получения скалярных значений
        assert result.loc['cat', 'shapes'].item() == 3
        assert result.loc['dog', 'shapes'].item() == 1
        assert result.loc['cat', 'tracks'].item() == 2  # track_id 1 и 3
        assert result.loc['dog', 'tracks'].item() == 1  # track_id 2

    def test_tasks2statistic(self) -> None:
        """Тест подсчета статистики для списка задач."""
        # Создаем тестовые задачи
        tasks = [
            [  # Задача 1
                (
                    pd.DataFrame(
                        {
                            'label': ['cat', 'dog'],
                            'track_id': [1, 2],
                            'outside': [False, False],
                        }
                    ),
                    'file1.jpg',
                    {0: 0},
                )
            ],
            [  # Задача 2
                (
                    pd.DataFrame(
                        {
                            'label': ['cat', 'cat'],
                            'track_id': [3, 3],
                            'outside': [False, False],
                        }
                    ),
                    'file2.jpg',
                    {0: 0},
                )
            ],
        ]

        # Мокаем labels_convertor
        mock_convertor = Mock()

        # init_df_counter должен возвращать DataFrame с одним столбцом
        img_stat = pd.DataFrame({'shapes': 0}, index=['cat', 'dog'])
        vid_stat = pd.DataFrame({'tracks': 0}, index=['cat', 'dog'])

        # Настраиваем side_effect для двух вызовов в каждом df2statistic
        mock_convertor.init_df_counter.side_effect = [img_stat, vid_stat] * 2
        mock_convertor.any_label2meaning.side_effect = lambda x: x

        # Мокаем mpmap для последовательного выполнения
        with patch('yolo.mpmap') as mock_mpmap:
            # Настраиваем результаты для каждой задачи
            result1 = pd.DataFrame(
                {'shapes': [1, 1], 'tracks': [1, 1]}, index=['cat', 'dog']
            )
            result2 = pd.DataFrame(
                {'shapes': [2, 0], 'tracks': [1, 0]}, index=['cat', 'dog']
            )
            mock_mpmap.return_value = [result1, result2]

            result = tasks2statistic(
                tasks=tasks, source_type='test', labels_convertor=mock_convertor
            )

        # Проверяем суммарные результаты
        assert result.loc['cat', 'shapes'].item() == 3
        assert result.loc['dog', 'shapes'].item() == 1
        assert result.loc['cat', 'tracks'].item() == 2
        assert result.loc['dog', 'tracks'].item() == 1

    def test_class_statistic2superclass_statistic(self) -> None:
        """Тест преобразования статистики классов в суперклассы."""
        # Создаем тестовую статистику классов
        stat = pd.DataFrame(
            {'shapes': [10, 20, 30], 'tracks': [5, 10, 15]},
            index=['cat', 'dog', 'bird'],
        )

        # Мокаем labels_convertor
        mock_convertor = Mock()

        # init_df_counter должен возвращать DataFrame с одним столбцом
        # Будет вызван дважды: для 'shapes' и 'tracks'
        shapes_counter = pd.DataFrame({'shapes': 0}, index=['animals', 'birds'])
        tracks_counter = pd.DataFrame({'tracks': 0}, index=['animals', 'birds'])

        mock_convertor.init_df_counter.side_effect = [shapes_counter, tracks_counter]
        mock_convertor.class_meaning2superclass_meaning = {
            'cat': 'animals',
            'dog': 'animals',
            'bird': 'birds',
        }

        result = class_statistic2superclass_statistic(stat, mock_convertor)

        # Проверяем результаты
        assert result.loc['animals', 'shapes'].item() == 30  # cat + dog
        assert result.loc['birds', 'shapes'].item() == 30  # bird
        assert result.loc['animals', 'tracks'].item() == 15  # cat + dog
        assert result.loc['birds', 'tracks'].item() == 15  # bird

    def test_fill_skipped_rows_in_statistic(self) -> None:
        """Тест заполнения пропущенных строк в статистике."""
        # Создаем тестовую статистику с пропусками
        stat = pd.DataFrame({'count': [10, 20]}, index=['cat', 'dog'])

        # Полный индекс
        full_index = ['cat', 'dog', 'bird', 'fish']

        result = fill_skipped_rows_in_statistic(stat, full_index)

        # Проверяем результаты
        assert 'bird' in result.index
        assert 'fish' in result.index
        assert result.loc['bird', 'count'] == 0
        assert result.loc['fish', 'count'] == 0
        assert result.loc['cat', 'count'] == 10
        assert result.loc['dog', 'count'] == 20


class TestSources2Statistic:
    """Тесты для функции sources2statistic_and_train_val_test_tasks."""

    def test_cvat_source_split(self, tmp_path: Path) -> None:
        """Тест разделения CVAT источника."""
        # Создаем тестовые задачи для CVAT
        test_tasks: list = [
            [  # Задача с тестовым бекапом
                (pd.DataFrame(), '/path/to/backup_test_001/task.xml', {})
            ]
        ]

        train_tasks: list = [
            [  # Задача с тренировочным бекапом
                (pd.DataFrame(), '/path/to/backup_train_001/task.xml', {})
            ]
        ]

        source_name2tasks = {'cvat': test_tasks + train_tasks}

        # Мокаем labels_convertor
        mock_convertor = Mock()
        # init_df_counter должен возвращать DataFrame с правильной структурой
        mock_counter = pd.DataFrame({'shapes': [0], 'tracks': [0]}, index=['test'])
        mock_convertor.init_df_counter.return_value = mock_counter

        # Мокаем другие функции
        with (
            patch('yolo.mkdirs'),
            patch('yolo.tasks2statistic') as mock_stat,
            patch('yolo.class_statistic2superclass_statistic') as mock_super,
            patch('yolo.pd.DataFrame.to_csv'),
            patch('yolo.pd.DataFrame.to_excel'),
            patch('yolo.df2img'),
        ):
            mock_stat.return_value = pd.DataFrame({'shapes': [0], 'tracks': [0]})
            mock_super.return_value = pd.DataFrame({'shapes': [0], 'tracks': [0]})

            _train, _val, _test = sources2statistic_and_train_val_test_tasks(
                source_name2tasks=source_name2tasks,
                yolo_ds_dir=str(tmp_path / 'test'),
                labels_convertor=mock_convertor,
                val_size=0.2,
                test_size=0,
            )

        # Проверяем, что функции вызывались
        mock_stat.assert_called()

    def test_gg_source_split(self, tmp_path: Path) -> None:
        """Тест разделения GG источника."""
        # Создаем тестовые задачи для GG
        tasks: list = [
            [('df1', 'file1', {})],
            [('df2', 'file2', {})],
            [('df3', 'file3', {})],
            [('df4', 'file4', {})],
            [('df5', 'file5', {})],
        ]

        source_name2tasks = {'gg': tasks}

        # Мокаем необходимые компоненты
        with (
            patch('yolo.mkdirs'),
            patch('yolo.train_val_test_split') as mock_split,
            patch('yolo.tasks2statistic') as mock_stat,
            patch('yolo.class_statistic2superclass_statistic') as mock_super,
            patch('yolo.pd.DataFrame.to_csv'),
            patch('yolo.pd.DataFrame.to_excel'),
            patch('yolo.df2img'),
        ):
            # Настраиваем мок для разделения
            mock_split.return_value = (
                tasks[0:3],  # train
                tasks[3:4],  # val
                tasks[4:5],  # test
            )

            # Настраиваем моки статистики
            mock_stat.return_value = pd.DataFrame({'shapes': [0], 'tracks': [0]})
            mock_super.return_value = pd.DataFrame({'shapes': [0], 'tracks': [0]})

            train, val, test = sources2statistic_and_train_val_test_tasks(
                source_name2tasks=source_name2tasks,
                yolo_ds_dir=str(tmp_path / 'test'),
                labels_convertor=Mock(),
                val_size=0.2,
                test_size=0.2,
                random_state=42,
            )

            # Проверяем разделение
            assert len(train) == 3
            assert len(val) == 1
            assert len(test) == 1


class TestYAMLGeneration:
    """Тесты для генерации YAML файлов."""

    def test_gen_yaml(self, tmp_path: Path) -> None:
        """Тест генерации YAML файла."""
        yolo_ds_dir = tmp_path / 'dataset'
        im_trn_dir = yolo_ds_dir / 'images' / 'train'
        im_val_dir = yolo_ds_dir / 'images' / 'val'
        im_tst_dir = yolo_ds_dir / 'images' / 'test'

        superclasses = {
            0: 'animal',
            1: 'vehicle',
            -1: 'unused',  # Должен быть исключен
        }

        yaml_file = tmp_path / 'data.yaml'

        # Создаем директории
        im_trn_dir.mkdir(parents=True)

        gen_yaml(
            file_name=str(yaml_file),
            yolo_ds_dir=str(yolo_ds_dir),
            im_trn_dir=str(im_trn_dir),
            im_val_dir=str(im_val_dir),
            im_tst_dir=str(im_tst_dir),
            superclasses=superclasses,
        )

        assert yaml_file.exists()

        # Проверяем содержимое
        with yaml_file.open() as f:
            content = yaml.safe_load(f)

        assert content['path'] == str(yolo_ds_dir)
        assert content['train'] == str(Path(im_trn_dir).relative_to(yolo_ds_dir))
        assert content['val'] == str(Path(im_val_dir).relative_to(yolo_ds_dir))
        assert content['test'] == str(Path(im_tst_dir).relative_to(yolo_ds_dir))
        assert content['names'] == {0: 'animal', 1: 'vehicle'}
        assert -1 not in content['names']  # Неиспользуемый класс исключен

    def test_gen_yaml_with_dir(self, tmp_path: Path) -> None:
        """Тест генерации YAML файла при передаче директории."""
        yolo_ds_dir = tmp_path / 'dataset'
        yaml_dir = tmp_path / 'config'
        yaml_dir.mkdir()

        gen_yaml(
            file_name=str(yaml_dir),
            yolo_ds_dir=str(yolo_ds_dir),
            im_trn_dir='/train',
            im_val_dir='/val',
            im_tst_dir='/test',
            superclasses={0: 'class'},
        )

        expected_file = yaml_dir / 'data.yaml'
        assert expected_file.exists()


class TestTask2YOLO:
    """Тесты для функции task2yolo."""

    def _test_task2yolo_basic_setup(self) -> pd.DataFrame:
        """Общая настройка для тестов task2yolo."""
        return pd.DataFrame(
            {
                'label': ['cat'],
                'type': ['rectangle'],
                'points': [[100, 200, 300, 400]],
                'rotation': [0],
                'outside': [False],
                'true_frame': [0],
            }
        )

    @patch('yolo.ImReadBuffer')
    @patch('yolo.YOLOLabels')
    @patch('cv2.imwrite')
    @patch('shutil.copyfile')
    @patch('yolo.draw_contrast_text')
    def test_task2yolo_basic(  # noqa: PLR0913
        self,
        mock_draw_text: Mock,
        mock_copyfile: Mock,
        mock_imwrite: Mock,
        mock_yolo_labels: Mock,
        mock_buffer: Mock,
        tmp_path: Path,
    ) -> None:
        """Базовый тест преобразования задачи в YOLO формат."""
        df = self._test_task2yolo_basic_setup()
        task = [(df, str(tmp_path / 'test.jpg'), {0: 0})]

        # Создаем директории
        images_dir = tmp_path / 'images'
        labels_dir = tmp_path / 'labels'
        preview_dir = tmp_path / 'preview'

        images_dir.mkdir()
        labels_dir.mkdir()
        preview_dir.mkdir()

        # Настраиваем моки
        mock_buffer_instance = Mock()
        mock_buffer_instance.file = str(tmp_path / 'test.jpg')
        mock_buffer_instance.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_buffer.return_value = mock_buffer_instance

        mock_yolo_instance = Mock()
        mock_yolo_instance.save.return_value = True
        mock_yolo_labels.return_value = mock_yolo_instance

        mock_imwrite.return_value = True
        mock_copyfile.side_effect = None
        mock_draw_text.return_value = np.zeros((600, 800, 3), dtype=np.uint8)

        # Мокаем labels_convertor
        mock_convertor = Mock()
        mock_convertor.side_effect = lambda _: 0

        task2yolo(
            sample_ind=0,
            mode='box',
            task=task,
            labels_convertor=mock_convertor,
            images_dir=str(images_dir),
            lablels_dir=str(labels_dir),
            preview_dir=str(preview_dir),
        )

        # Проверяем вызовы
        mock_buffer.assert_called()
        mock_yolo_labels.assert_called()

    @patch('yolo.ImReadBuffer')
    @patch('yolo.YOLOLabels')
    @patch('cv2.imwrite')
    @patch('cv2.resize')
    def test_task2yolo_with_scale(  # noqa: PLR0913
        self,
        mock_resize: Mock,
        mock_imwrite: Mock,
        mock_yolo_labels: Mock,
        mock_buffer: Mock,
        tmp_path: Path,
    ) -> None:
        """Тест преобразования с масштабированием."""
        df = self._test_task2yolo_basic_setup()
        task = [(df, str(tmp_path / 'test.jpg'), {0: 0})]

        # Создаем директории
        images_dir = tmp_path / 'images'
        labels_dir = tmp_path / 'labels'

        images_dir.mkdir()
        labels_dir.mkdir()

        # Настраиваем моки
        mock_buffer_instance = Mock()
        mock_buffer_instance.file = str(tmp_path / 'test.jpg')
        mock_buffer_instance.return_value = np.zeros((600, 800, 3), dtype=np.uint8)
        mock_buffer.return_value = mock_buffer_instance

        mock_yolo_instance = Mock()
        mock_yolo_instance.save.return_value = True
        mock_yolo_labels.return_value = mock_yolo_instance

        mock_imwrite.return_value = True
        mock_resize.return_value = np.zeros((300, 400, 3), dtype=np.uint8)

        # Мокаем CVATPoints для масштабирования
        with patch('yolo.CVATPoints') as mock_cvat:
            mock_cvat_instance = Mock()
            mock_cvat_instance.__mul__ = lambda self, _: self
            mock_cvat_instance.flatten.return_value = [50, 100, 150, 200]
            mock_cvat.return_value = mock_cvat_instance

            # Мокаем labels_convertor
            mock_convertor = Mock()
            mock_convertor.side_effect = lambda _: 0

            task2yolo(
                sample_ind=0,
                mode='box',
                task=task,
                labels_convertor=mock_convertor,
                images_dir=str(images_dir),
                lablels_dir=str(labels_dir),
                scale=0.5,
            )

        # Проверяем вызовы
        mock_resize.assert_called()

    @patch('yolo.ImReadBuffer')
    @patch('yolo.split_image_and_labels2tiles')
    @patch('yolo.YOLOLabels')
    @patch('cv2.imwrite')
    def test_task2yolo_with_tiling(  # noqa: PLR0913
        self,
        mock_imwrite: Mock,
        mock_yolo_labels: Mock,
        mock_split: Mock,
        mock_buffer: Mock,
        tmp_path: Path,
    ) -> None:
        """Тест преобразования с разбиением на тайлы."""
        df = self._test_task2yolo_basic_setup()
        task = [(df, str(tmp_path / 'test.jpg'), {0: 0})]

        # Создаем директории
        images_dir = tmp_path / 'images'
        labels_dir = tmp_path / 'labels'

        images_dir.mkdir()
        labels_dir.mkdir()

        # Настраиваем моки
        mock_buffer_instance = Mock()
        mock_buffer_instance.file = str(tmp_path / 'test.jpg')
        test_image = np.zeros((2000, 3000, 3), dtype=np.uint8)
        mock_buffer_instance.return_value = test_image
        mock_buffer.return_value = mock_buffer_instance

        mock_yolo_instance = Mock()
        mock_yolo_instance.save.return_value = True
        mock_yolo_labels.return_value = mock_yolo_instance

        mock_imwrite.return_value = True

        # Настраиваем разбиение на тайлы
        mock_split.return_value = [
            (df, test_image[:1000, :1500]),
            (df, test_image[:1000, 1500:]),
            (df, test_image[1000:, :1500]),
            (df, test_image[1000:, 1500:]),
        ]

        # Мокаем labels_convertor
        mock_convertor = Mock()
        mock_convertor.side_effect = lambda _: 0

        task2yolo(
            sample_ind=0,
            mode='seg',
            task=task,
            labels_convertor=mock_convertor,
            images_dir=str(images_dir),
            lablels_dir=str(labels_dir),
            max_imsize=(1080, 1920),
        )

        # Проверяем вызовы
        mock_split.assert_called()
        assert mock_yolo_labels.call_count == 4  # По одному для каждого тайла


class TestTasks2YOLO:
    """Тесты для функции tasks2yolo."""

    def _test_tasks2yolo_basic_setup(self, tmp_path: Path) -> list:
        """Общая настройка для тестов tasks2yolo."""
        df = pd.DataFrame(
            {
                'label': ['cat'],
                'type': ['rectangle'],
                'points': [[100, 200, 300, 400]],
                'rotation': [0],
                'outside': [False],
                'true_frame': [0],
            }
        )

        return [
            [(df, str(tmp_path / 'test1.jpg'), {0: 0})],
            [(df, str(tmp_path / 'test2.jpg'), {0: 0})],
        ]

    @patch('yolo.flat_tasks')
    @patch('yolo.sort_tasks')
    @patch('yolo.fill_na_in_track_id_in_all_tasks')
    @patch('yolo.init_task_object_file_graphs')
    @patch('yolo.mpmap')
    @patch('yolo.drop_unused_track_ids_in_graphs')
    def test_tasks2yolo_basic(  # noqa: PLR0913
        self,
        mock_drop: Mock,
        mock_mpmap: Mock,
        mock_init: Mock,
        mock_fill: Mock,
        mock_sort: Mock,
        mock_flat: Mock,
        tmp_path: Path,
    ) -> None:
        """Базовый тест преобразования задач в YOLO формат."""
        tasks = self._test_tasks2yolo_basic_setup(tmp_path)

        # Настраиваем моки
        mock_flat.return_value = tasks
        mock_sort.return_value = tasks
        mock_fill.return_value = tasks
        mock_init.return_value = [None, None]
        mock_mpmap.return_value = [None, None]
        mock_drop.return_value = [None, None]

        with patch('yolo.mkdirs'):
            tasks2yolo(
                mode='box',
                tasks=tasks,
                labels_convertor=Mock(),
                images_dir=str(tmp_path / 'images'),
                lablels_dir=str(tmp_path / 'labels'),
            )

        # Проверяем вызовы
        mock_flat.assert_called_once_with(tasks)
        mock_sort.assert_called_once()
        mock_mpmap.assert_called()

    @patch('yolo.flat_tasks')
    @patch('yolo.sort_tasks')
    @patch('yolo.fill_na_in_track_id_in_all_tasks')
    @patch('yolo.init_task_object_file_graphs')
    @patch('yolo.mpmap')
    @patch('yolo.drop_unused_track_ids_in_graphs')
    @patch('yolo.make_copy_bal')
    def test_tasks2yolo_with_copybal(  # noqa: PLR0913
        self,
        mock_bal: Mock,
        mock_drop: Mock,
        mock_mpmap: Mock,
        mock_init: Mock,
        mock_fill: Mock,
        mock_sort: Mock,
        mock_flat: Mock,
        tmp_path: Path,
    ) -> None:
        """Тест преобразования с копирующей балансировкой."""
        tasks = self._test_tasks2yolo_basic_setup(tmp_path)

        # Настраиваем моки
        mock_flat.return_value = tasks
        mock_sort.return_value = tasks
        mock_fill.return_value = tasks
        # Инициализируем графы для каждой задачи (2 задачи)
        mock_init.return_value = [Mock(), Mock()]
        # mpmap возвращает графы для каждой задачи
        mock_mpmap.return_value = [Mock(), Mock()]
        # drop_unused возвращает отфильтрованные графы
        mock_drop.return_value = [Mock(), Mock()]
        mock_bal.return_value = Mock()

        with patch('yolo.mkdirs'):
            tasks2yolo(
                mode='box',
                tasks=tasks,
                labels_convertor=Mock(),
                images_dir=str(tmp_path / 'images'),
                lablels_dir=str(tmp_path / 'labels'),
                use_copybal=True,
            )

        # Проверяем, что функции балансировки вызывались
        mock_fill.assert_called()
        mock_init.assert_called()
        mock_bal.assert_called()


class TestExtensionSets:
    """Тесты для наборов расширений."""

    def test_yolo_vid_exts(self) -> None:
        """Тест набора видео расширений."""
        assert '.mp4' in yolo_vid_exts
        assert '.avi' in yolo_vid_exts
        assert '.mov' in yolo_vid_exts
        assert '.mkv' in yolo_vid_exts
        assert '.jpg' not in yolo_vid_exts

    def test_yolo_img_exts(self) -> None:
        """Тест набора изображений расширений."""
        assert '.jpg' in yolo_img_exts
        assert '.png' in yolo_img_exts
        assert '.bmp' in yolo_img_exts
        assert '.jpeg' in yolo_img_exts
        assert '.mp4' not in yolo_img_exts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
