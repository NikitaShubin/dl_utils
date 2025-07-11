'''
********************************************
*  Набор самописных утилит для TensorFlow. *
*                                          *
*                                          *
* Основные функции:                        *
*   backbone2encoder - функция,            *
*       использующая базовую модель для    *
*       построения новой, извлекающей      *
*       признаки на разных масштабах       *
*       (используется в качестве           *
*       кодирующей части в U-Net);         *
*                                          *
*   NamingLayer - слой, который просто     *
*       задаёт имя тензору;                *
*                                          *
*   MaxFscore - метрика, подсчитывающая    *
*       максимальное достигаемое значение  *
*       F-меры, и/или соответствующий ему  *
*       порог;                             *
*                                          *
*   get_model_elements - функция,          *
*       возвращающая словарь, значениями   *
*       которого являются подходящие под   *
*       описание элементы;                 *
*                                          *
*   MultiGPU - Класс-утилита для работы с  *
*       несколькими GPU;                   *
*                                          *
*   show_tensorboard_info - функция,       *
*       выводящая информацию о том, как    *
*       запустить и по какой ссылке        *
*       открыть TensorBoard.               *
********************************************
'''

import os

import tensorflow  as tf
layers       = tf.keras.layers
callbacks    = tf.keras.callbacks
applications = tf.keras.applications
backend      = tf.keras.backend
models       = tf.keras.models

from utils import TimeIt, mpmap, obj2yaml, yaml2obj, cls, a2hw
from alb_utils import AlbTransforms


class TFInit:
    '''
    Управление инициализацией TF.
    Может быть использован как контекст.
    '''

    def __init__(self,
                 less_gpu_mem=False,
                 log_level='ERROR'):

        # Фиксируем входные параметры:
        self.less_gpu_mem = less_gpu_mem
        self.log_level = log_level

        # Получаем доступ к логеру TF%:
        self.logger = tf.get_logger()

        # Выполняем все нужные изменения:
        self.__enter__()

    def __enter__(self):
        # Понижаем детальность логирования TensorFlow ...
        # ... до уровня ошибок, если надо:
        if self.log_level:

            # Фиксируем текущий уровень логирования:
            self.level = self.logger.getEffectiveLevel()

            # Отключаем предупреждения (оставляем только ошибки):
            self.logger.setLevel(self.log_level)

        # Минимизируем занимаемую память, если надо:
        if self.less_gpu_mem:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)

    def __exit__(self, type, value, traceback):
        # Возвращаем исходный уровень логирования, если он был изменён:
        if self.log_level:
            self.logger.setLevel(self.level)


# Использовать минимум памяти GPU:
def tf_less_mem():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


class MultiGPU():
    '''
    Класс-утилита для работы с несколькими GPU.
    '''

    def __init__(self):

        # Инициируем остновные поля:

        # Число доступных GPU:
        self.num = len(tf.config.list_physical_devices('GPU'))
        self.multi = self.num > 1  # Больше одной GPU?
        self.any = self.num > 0    # GPU вообще есть?

        # Определяем наиболее подходящую стратегию распределённых вычислений:
        self.strategy = tf.distribute.MirroredStrategy() if self.multi \
            else tf.distribute.get_strategy()

    def batch_size_multiplier(self, batch_size, default=1.):
        '''
        Увеличивает batch_size пропорционально числу GPU.

        Если GPU нет, то в ход идёт параметр "default":
            Если default - целое число, то его и берут в качестве нового GPU
                (полезно в случае, если на CPU гораздо больше памяти, и 
                 batch_size можно сильно увеличить);
            Если default - вещественное число, то оно берётся как коэффициент,
                на который исхоодный batch_size умножается.
        '''

        # Если GPU есть, то умножаем исходный batch_size на число GPu:
        if self.any:
            batch_size = batch_size * self.num

        # Если GPU нет но default задан целым числом, то берём это число как
        # batch_size:
        elif isinstance(default, int):
            batch_size = default

        # Если GPU нет но default задан вещественным числом, то оспринимаем
        # default как коэффициент масштабирования:
        elif isinstance(default, float):
            batch_size = int(batch_size * default)

        # Иначе выводим ошибку:
        else:
            raise ValueError(
                'Параметр "default" дожен быть типа float или int!')

        return batch_size


def load_IO_file_names2dataset(path):
    '''
    Формирует tf-датасет, содержащий списки имён входных/выходных файлов.
    '''
    # Определяем имена подпапок:
    source_path = os.path.join(path, 'inp', '*')  # Путь ко входным файлам
    target_path = os.path.join(path, 'out', '*')  # Путь к выходным файлам

    # Создаём два датасета:
    source_dataset = tf.data.Dataset.list_files(source_path, shuffle=False)
    # Датасет имён входных файлов.
    target_dataset = tf.data.Dataset.list_files(target_path, shuffle=False)
    # Датасет имён выходных файлов.

    # Объединяем два датасета в один, генерирующий пару имён файлов вход-выход:
    return tf.data.Dataset.zip((source_dataset, target_dataset))


def make_load_and_transform_func(num_classes,
                                 transforms=None,
                                 target_size=None):
    '''
    Создаёт tf-функцию, которая читает изображения и
    выполняет над ними Albumentation-преобразования:
    '''
    if isinstance(transforms, int):
        transforms = (transforms, transforms)

    if transforms is None:

        if target_size is None:
            @tf.function()
            def func(image_file, mask_file):
                # Чтение изображений
                image = tf.io.decode_image(tf.io.read_file(image_file),
                                           expand_animations=False)
                mask = tf.io.decode_image(tf.io.read_file(mask_file),
                                          expand_animations=False)

                # Постобработка обоих тензоров:
                image = tf.cast(image, tf.float32) / 255.
                mask = tf.one_hot(mask[..., 0], num_classes)

                return image, mask

        else:
            @tf.function()
            def func(image_file, mask_file):
                # Чтение изображений
                image = tf.io.decode_image(tf.io.read_file(image_file),
                                           expand_animations=False)
                mask = tf.io.decode_image(tf.io.read_file(mask_file),
                                          expand_animations=False)

                # Меняем размеры итоговых изображений:
                image = tf.image.resize(image, target_size, 'bilinear')
                mask = tf.image.resize(mask, target_size, 'nearest')

                # Постобработка обоих тензоров:
                image = tf.cast(image, tf.float32) / 255.
                mask = tf.one_hot(mask[..., 0], num_classes)

                return image, mask

    else:

        @tf.numpy_function(Tout=(tf.uint8, tf.uint8))
        def tf_transforms(image, mask):
            image, mask = transforms(image=image, mask=mask).values()
            return image, mask

        if target_size is None:
            @tf.function()
            def func(image_file, mask_file):
                # Чтение изображений
                image = tf.io.decode_image(tf.io.read_file(image_file),
                                           expand_animations=False)
                mask = tf.io.decode_image(tf.io.read_file(mask_file),
                                          expand_animations=False)

                # Применяем Albumentations-преобразования к паре изображений
                # вход-выход:
                image, mask = tf_transforms(image, mask)

                # Постобработка обоих тензоров:
                image = tf.cast(image, tf.float32) / 255.
                mask = tf.one_hot(mask[..., 0], num_classes)

                return image, mask

        else:
            @tf.function()
            def func(image_file, mask_file):
                # Чтение изображений
                image = tf.io.decode_image(tf.io.read_file(image_file),
                                           expand_animations=False)
                mask = tf.io.decode_image(tf.io.read_file(mask_file),
                                          expand_animations=False)

                # Применяем Albumentations-преобразования к паре изображений
                # вход-выход:
                image, mask = tf_transforms(image, mask)
                image.set_shape((None, None, None))
                mask.set_shape((None, None, None))
                # Для корректной работы последующего изменения размеров
                # преобразованных изображений необходимо вновь явно задать
                # размеры этих тензоров (хотя бы указать, что они трёхмерные).

                # Меняем размеры итоговых изображений:
                image = tf.image.resize(image, target_size, 'bilinear')
                mask = tf.image.resize(mask, target_size, 'nearest')

                # Постобработка обоих тензоров:
                image = tf.cast(image, tf.float32) / 255.
                mask = tf.one_hot(mask[..., 0], num_classes)

                return image, mask

    return func


def get_dataset(path,
                num_classes,
                transforms=None,
                target_size=None,
                batch_size=32,
                shuffle=False):
    '''
    Формирует датасет tf.data.Data для обучения, валидации или тестирования.
    '''
    # Загружаем список имеющихся файлов:
    ds = load_IO_file_names2dataset(path)

    # Кешируем, т.к. этот список меняться не будет:
    ds = ds.cache()

    # Перемешиваем очерёдность, если надо:
    if shuffle:
        ds = ds.shuffle(len(ds))

    # Применяем преобразование, если задано:
    ds = ds.map(make_load_and_transform_func(num_classes, transforms,
                                             target_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    # Если transforms == None, то просто выполняется чтение изображений и
    # необходимые их преобразования.

    # Группируем семплы в батчи, если задан batch_size:
    if batch_size:
        ds = ds.batch(batch_size, drop_remainder=shuffle)
        # Если включена рандомизация очереди семплов, то ...
        # ... отбрасываем последний неполный батч, т.к. ...
        # ... он может иметь размер 1, что вызывает NaN-ы ...
        # ... в слоях пакетной нормализации при обучении.

    # Включаем опережающую подготовку данных:
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # ds = ds.prefetch(batch_size)

    return ds


def get_datasets(path,
                 num_classes,
                 train_transforms=None,
                 val_test_transforms=None,
                 im_size=(608, 800),
                 batch_size=32):
    '''
    Возвращает все 3 датасета: обучающий, проверочный, тестовый.
    '''
    # Собираем бъединённые преобразования из списков преобразований, если надо:
    train_transforms = \
        AlbTransforms.compose_transforms(train_transforms, True)
    val_test_transforms = \
        AlbTransforms.compose_transforms(val_test_transforms, True)

    train_ds = get_dataset(os.path.join(path, 'train'),
                           num_classes,
                           train_transforms,
                           im_size,
                           batch_size,
                           shuffle=True)
    val_ds = get_dataset(os.path.join(path, 'val'),
                         num_classes,
                         val_test_transforms,
                         im_size,
                         batch_size,
                         shuffle=False)
    test_ds = get_dataset(os.path.join(path, 'test'),
                          num_classes,
                          val_test_transforms,
                          im_size,
                          batch_size,
                          shuffle=False)

    return train_ds, val_ds, test_ds