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
import cv2

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow  as tf
import tensorboard as tb
from tensorflow.keras import layers, callbacks, applications, backend, models


from utils import TimeIt, mpmap, obj2yaml, yaml2obj, cls, a2hw


class TFInit:
    '''
    Управление инициализацией TF.
    Может быть использован как контекст.
    '''
    def __init__(self              ,
                 less_gpu_mem=False,
                 log_level='ERROR' ):
        
        # Фиксируем входные параметры:
        self.less_gpu_mem = less_gpu_mem
        self.log_level    = log_level
        
        # Получаем доступ к логеру TF%:
        self.logger       = tf.get_logger()
        
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
    
    def __exit__(self, type, value, tb):
        
        # Возвращаем исходный уровень логирования, если он был изменён:
        if self.log_level:
            logger.setLevel(self.level)


# Использовать минимум памяти GPU:
def tf_less_mem():
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


class SegDataLoader(tf.keras.utils.Sequence):
    '''
    Даталоадер для данных с сегментацией.
    '''
    def __init__(self,
                 path                 : 'Путь до папок inp и out, в которых лжат изображения входа и выхода'              ,
                 num_classes          : 'Число классов сегментации'                                                = None ,
                 transforms           : 'Преобразователь входа и выхода (типа albumentations или аналогичный)'     = None ,
                 batch_size           : 'Размер минивыборки'                                                       = 32   ,
                 shuffle              : 'Флаг перемешивания выборки перед началом каждой следующей эпохи'          = False,
                 epoch_multiplier     : 'Во сколько раз увеличить размер эпохи путём повторения семплов?'          = 1    ,
                 use_multitherading   : 'Использовать ли многопроцессорность? Может как ускорить, так и замедлить' = False,
                 drop_incomplete_batch: 'Флаг отбрасывания последнего батча в эпохе, если он короче batch_size'    = False):
        
        # Пути до вложенных папок со входными и выходными данными:
        inp_path = os.path.join(path, 'inp')
        out_path = os.path.join(path, 'out')
        
        # Формируем списки входных и выходных файлов:
        inps = sorted(os.listdir(inp_path))
        outs = sorted(os.listdir(out_path))
        
        # Всем ли именам файлов входных и выходных данных можно установить взаимнооднозначное соответствие:
        for inp, out in zip(inps, outs):
            assert os.path.splitext(inp)[0].lower() == os.path.splitext(out)[0].lower()
        
        # Делаем полные пути до файлов:
        self.inps = [os.path.join(inp_path, _) for _ in inps]
        self.outs = [os.path.join(out_path, _) for _ in outs]
        
        # Сохраняем остальные параметры:
        self.num_classes           = num_classes
        self.transforms            = transforms
        self.batch_size            = batch_size
        self.shuffle               = shuffle
        self.epoch_multiplier      = epoch_multiplier
        self.use_multitherading    = use_multitherading
        self.drop_incomplete_batch = drop_incomplete_batch
        
        # Формируем вектор индексов семплов в одной эпохе:
        self.indexes = np.arange(len(self.inps) * self.epoch_multiplier)
        
        # Перемешиваем индексы, если надо:
        self.on_epoch_end()
    
    # В конце каждой эпохи перемешиваем последовательность, если нужно:
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    # Число батчей в выборке:
    def __len__(self):
        
        # Сначала получаем вещественное число:
        num = len(self.inps) * self.epoch_multiplier / self.batch_size
        
        # В зависимости от флага drop_incomplete_batch ...
        # ... округляем в большую или меньшую сторону:
        num = np.fix(num) if self.drop_incomplete_batch else np.ceil(num) 
        
        return int(num)
    
    # Получаем список индексов семплов по индексу батча:
    def batch_id2sample_ids(self, batch_id):
        
        # Определяем началный и конечный индексы текущего батча:
        sample_start_id =        batch_id * self.batch_size
        sample_end_id   = sample_start_id + self.batch_size
        
        # Если это последний батч эпохи и он укороченный, ...
        # .. то просто забираем все индексы оставшиеся семплов:
        if sample_end_id >= len(self.indexes):
            sample_end_id = None
        
        return self.indexes[sample_start_id: sample_end_id]
    
    # Загружает и аугментирует семпл:
    def get_sample(self, sample_id):
        
        # Замыкаем итератор, чтобы при выходе за ...
        # ... реальные индексы шло повторение с начала:
        sample_id = sample_id % len(self.inps)
        
        # Загружаем изображения:
        inp = cv2.imread(self.inps[sample_iCustomImaged], cv2.IMREAD_COLOR    )
        out = cv2.imread(self.outs[sample_id]           , cv2.IMREAD_GRAYSCALE)
        
        inp = inp[..., ::-1].astype('float32') / 255.
        
        # Применяем аугментацию, если она задана:
        if self.transforms:
            inp, out = self.transforms(image=inp, mask=out).values()
        
        if self.num_classes:
            out = tf.keras.utils.to_categorical(out, num_classes=self.num_classes, dtype='float32')
        else:
            out = out.astype('float32') / 255.
        
        return inp, out
        # Важно отметить, что перевод изображения во float32 выполняется до аугментации, ...
        # ... а работа с маской (включая создание каналов по классам) уже после! ...
        # ... Цветовые и геометрические изменения точнее выполняются в числах с ...
        # ... плавающей точкой, а все операции с масками являются целочисленными, ...
        # ... поэтому их легче проводить в uint8.
    
    # Формируем очередной батч:
    def __getitem__(self, idx):
        
        # Формируем список семплов:
        X, Y = [], []
        
        # Получаем списки семплов в параллельном или обычном режимах:
        if self.use_multitherading:
            for x, y in mpmap(self.get_sample, self.batch_id2sample_ids(idx)):
                X.append(x[np.newaxis, ...])
                Y.append(y[np.newaxis, ...])
        else:
            for sample_id in self.batch_id2sample_ids(idx):
                x, y = self.get_sample(sample_id)
                X.append(x[np.newaxis, ...])
                Y.append(y[np.newaxis, ...])
        
        # Объединяем списки семплов в батчи:
        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0)
        
        return X, Y
    
    # Показываем пример одного батча:
    def show(self):
        
        # Берём индекс случайного батча:
        idx = np.random.randint(len(self))
        
        # Получаем сам батч:
        X, Y = self.__getitem__(idx)
        
        # Выводим все изображения батча:
        for x, y in zip(X, Y):
            fig = plt.figure(figsize=(24, 24), layout='tight')
            plt.subplot(121); plt.imshow(x); plt.axis(False)
            plt.subplot(122); plt.imshow(y); plt.axis(False)
            plt.show()


def get_keras_application_model_constructor_list(obj=applications):
    '''
    Формирует список конструкторов всех доступных Keras-моделей.
    Полезен, например, для бенчмарка базовых моделей (backbones).
    '''
    # Если в атрибутах объекта есть __call__ и __code__, ...
    # ... а одним из входных параметров является input_shape, ...
    # ... то считаем объект конструктором модели:
    if {'__call__', '__code__'} < set(dir(obj)) and 'input_shape' in obj.__code__.co_varnames:
        return [obj]

    # Получаем список атрибутов объекта:
    attrs = dir(obj)

    # Если объект является функцией/функтором, но не является ...
    # ... конструктором модели, то лезть вглубь нет смысла: 
    if '__call__' in attrs:
        return []

    # Рекурсивно перебираем все атрибуты объекта:
    models = []
    for attr in attrs:

        # Исключаем скрытые атрибуты:
        if attr[0] == '_':
            continue

        # Пополняем список моделей с помощью рекурсивного вызова:
        models += get_keras_application_model_constructor_list(getattr(obj, attr))
    
    return models


def backbone2encoder(backbone):
    '''
    Сборка модели извлечения многомасштабных признаков на основе базовой модели
    (backbone). Полезно при формировании U-Net-подобных архитектур, кодирующая часть
    которых является свёрточной частью какой-то предобученной модели (backbone).
    '''
    # Инициируем список признаков обязательным последним выходом
    outputs = [backbone.layers[-1].output]
    
    # А дальше будем этот список заполнять.
    # Для этого пробегаем по всем выходным тензорам,
    # начиная с предпоследнего, и добавляем все выходы,
    # размеры которых в 2 раза больше предыдущих
    # (т.е. после них размер уменьшается в 2 раза):
    if None in backbone.output_shape[1:]:
        # Если размер входного тензора базовой модели недоопределён (есть None-ы),
        # то приходится определять размеры всех признаков опытным путём:
        
        # Доопределим размер входного тензора (все неизвестные размеры заменяются на 512):
        batch_size = backbone.input_shape[0]
        inp_shape = [  1 if batch_size is None else batch_size] + \
                    [512 if dim        is None else dim for dim in backbone.input_shape[1:]]
        
        # Создаём сам этот тензор, заполненный нулями:
        input_tensor = tf.zeros(inp_shape)
        
        # Собираем новую модель с телом базовой модели, имеющей выход на каждом слое:
        model = tf.keras.models.Model(backbone.inputs, [layer.output for layer in backbone.layers])
        
        # Прогоняем нулевой тензор через модель и снимаем тензоры со всех слоёв:
        output_tensors = model(input_tensor)
        
        # Размер следующего искомого тензора (в 2 раза больше предыдущего):
        cur_size = output_tensors[-1].shape[-2] * 2 
        
        # Перебор всех тензоров, начиная с предпоследнего и к началу:
        for layer, output_tensor in zip(reversed(   backbone.layers[:-1]),
                                        reversed(output_tensors[:-1])):
            # Если размер текущего тензора действительно соответствует искомому, то ...
            if len(output_tensor.shape) > 2 and output_tensor.shape[-2] == cur_size:
                # ... добавляем выход соответствующего слоя в список...
                outputs.append(layer.output)
                
                # ... и в два раза увеличиваем размер следующего искомого тензора:
                cur_size *= 2
    else:
        # Если размер входного тензора базовой модели полностью определён
        # (None-ов нет), то и размеры всех промежуточных слоёв тоже определены:
        
        # Размер следующего искомого тензора (в 2 раза больше предыдущего):
        cur_size = backbone.layers[-1].output_shape[-2] * 2
        
        # Перебор всех слоёв, начиная с предпоследнего и к началу:
        for layer in reversed(backbone.layers[:-1]):
            # Если размер текущего тензора действительно соответствует искомому, то ...
            if len(layer.output_shape) > 2 and layer.output_shape[-2] == cur_size:
                # ... добавляем выход соответствующего слоя в список...
                outputs.append(layer.output)
                
                # ... и в два раза увеличиваем размер следующего искомого тензора:
                cur_size *= 2
    
    # Восстанавливаем последовательность выходов (отменяем реверсию):
    outputs = outputs[::-1]
    
    # Оборачиваем в модель и возвращаем:
    return tf.keras.models.Model(backbone.inputs, outputs, name='encoder')


# Слой, который просто задаёт имя тензору:
def NamingLayer(name):
    return layers.Lambda(lambda x: x, name=name)
# Бывает полезно для последующего поиска нужного ...
# ... узла в вычислительном графе модели по его имени.


def global_pool_2D_with_bottleneck(inp            : 'Входной тензор'                                ,
                                   bottleneck_size: 'Число признаков "бутылочного горлышка"' = 0.25 ,
                                   out_size       : 'Число признаков выходного тензора'      = 1.   ,
                                   mode           : 'Режим Пулинга (avg, max или both)'      ='both'):
    '''
    Выполняет GlobalPool2D ко входному тензору, после чего
    применяется свёрточный bottleneck. Этот блок можно
    применять для формирования Res-блока:
    x + global_pool_2D_with_bottleneck(x)
    '''
    # Принудительно переводим тип пулинга в нижний регистр:
    mode = mode.lower()
    
    # Если bottleneck_size задан вещественным числом, то ...
    # ... берём его как коэффициент от размера входного тензора:
    if not isinstance(bottleneck_size, int):
        bottleneck_size = int(inp.shape[-1] * bottleneck_size)
    
    # Если out_size задан вещественным числом, то ...
    # ... берём его как коэффициент от размера входного тензора:
    if not isinstance(out_size, int):
        out_size = int(inp.shape[-1] * out_size)
    
    # Если нужны оба типа пулинга:
    if mode == 'both':
        gap = layers.GlobalAvgPool2D(keepdims=True)(inp)
        gmp = layers.GlobalMaxPool2D(keepdims=True)(inp)
        gp = layers.Concatenate()([gap, gmp])
    
    # Если нужен MaxPool:
    elif mode == 'max':
        gp = layers.GlobalMaxPool2D(keepdims=True)(inp)
    
    # Если нужен AvgPool:
    elif mode == 'avg':
        gp = layers.GlobalAvgPool2D(keepdims=True)(inp)
    
    else:
        raise ValueError('Параметр "mode" должен быть одним из {"both", "max", "avg"}!')
    
    # Строим бутылочное горлышко:
    bn = layers.Conv2D(bottleneck_size, 1, activation=layers.LeakyReLU())(gp)
    bn = layers.BatchNormalization()(bn)
    bn = layers.Conv2D(out_size, 1)(bn)
    
    return bn


# Res-блок:
def res_block(inp, net, name='ResBlock'):
    with tf.name_scope(name):
        return inp + net(inp)


def fractal_unet_node(skip_connection = None,
                      upsampled       = None,
                      donwsampled     = None,
                      out_filters        = 4    ,
                      latent_filters     = 4    ,
                      latent_kernel_size = 3    ,
                      use_out_skip       = True ,
                      is_out_layer       = False,
                      dropout_rate       = 0.1  ,
                      name               = None ):
    '''
    Блок-функция, реализующая узел в Fractal-U-Net.
    '''
    connections = []
    if skip_connection is not None:
        connections.append(skip_connection)
    if upsampled is not None:
        upsampled = layers.MaxPooling2D()(upsampled)
        connections.append(upsampled)
    if donwsampled is not None:
        donwsampled = layers.UpSampling2D()(donwsampled)
        connections.append(donwsampled)
    
    if len(connections) > 1:
        x0 = layers.Concatenate()(connections)
    elif len(connections) == 1:
        x0 = connections[0]
    else:
        raise ValueError('Должен быть задан хотябы один вход из трёх!')

    # Применяем Dropout ко всем входам, если надо:
    if dropout_rate:
        #x0 = layers.       Dropout  (dropout_rate)(x0)
        x0 = layers.SpatialDropout2D(dropout_rate)(x0)
    
    x = layers.Conv2D(latent_filters, 1, use_bias=False)(x0)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.DepthwiseConv2D(latent_kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(out_filters, 1, use_bias=is_out_layer)(x)
    if not is_out_layer:
        x = layers.BatchNormalization()(x)
    if is_out_layer:
        x = layers.Activation('sigmoid' if out_filters == 1 else 'sigmoid', name=name)(x)
    else:
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(name=name)(x)
    
    if use_out_skip:
        skip = layers.Concatenate()([x, x0])
        return x, skip
    else:
        return x


def Resizing2DLike(like, interpolation='bilinear'):
    '''
    Масштабирование 2D-карты признаков к размеру образца.
    Не меняет число признаков.
    Не работает для карт признаков неопределённого размера!
    '''
    # Определяем целевых размеры тензора:
    h, w = backend.int_shape(like)[1:-1]

    # Исключаем неопределённость целевых размеров:
    if None in {h, w}:
        raise ValueError('Конечные размеры должны быть определены! Получены h, w == %s, %s' % (h, w)) 

    # Возвращаем слой, выполняющий нужные преобразования
    return layers.Resizing(h, w, interpolation=interpolation)


def global_pool_conv2D(x, mode='both', filters=512):
    '''
    Создаёт глобализованную карту признаков следующим образом:
        1) применяется MaxPool и/или AvgPool;
        2) вектор признаков преобразуется с помощью связки Conv2D + BN + ReLU;
        3) разтяжение вектора обратно в карту исходного размера.
    
    В каждом пикселе итоговой катры вектор признаков одинаков.
    '''
    # Принудительно переводим строку режима в нижний регистр:
    mode = mode.lower()
    
    # Список тензоров глобальных признаков:
    global_poolings = []
    
    # Глобальный макспулинг, если нужно:
    if mode in {'both', 'max'} and filters:
        global_poolings.append(layers.GlobalMaxPooling2D(keepdims=True)(x))
    
    # Глобальный эвереджпулинг, если нужно:
    if mode in {'both', 'avg'} and filters:
        global_poolings.append(layers.GlobalAveragePooling2D(keepdims=True)(x))
    
    # Если имеется хоть один из пулингов:
    if global_poolings:
        # Получение объединённого вектора признаков:
        global_poolings = layers.Concatenate()(global_poolings)
        
        # Conv2D + BN + ReLU:
        global_poolings = layers.Conv2D(filters, 1, use_bias=False)(global_poolings)
        global_poolings = layers.BatchNormalization()(global_poolings)
        global_poolings = layers.Activation('relu')(global_poolings)
        
        # Тиражирование вектора признаков в карту признаков исходного размера:
        global_poolings = Resizing2DLike(x, interpolation='nearest')(global_poolings)
        
        return global_poolings
    
    # Если ни один из пулингов не использовался, возвращаем None:
    return


def InputModel(input_tensor, input_shape, input_batch_size=None):
    '''
    Получаем и тензор, и вход из тензора или входа.
    Используется для включения базовых моделей в tf.keras.models.Model.
    '''
    # Если тензор не передан, то используем заданный размер входа:
    if input_tensor is None:
        tensor = layers.Input(shape=input_shape,
                              batch_size=input_batch_size)
        
        # Определяем вход итоговой сети:
        inputs = tensor
    
    # Если тензор передан, то используем его:
    else:
        # Определяем вход итоговой сети:
        tensor = input_tensor
        
        # Определяем вход итоговой сети:
        inputs = tf.keras.utils.get_source_inputs(input_tensor)
        # Т.о. все предобработки вовлекаеются в модель.
    
    return tensor, inputs


def backbone_with_preprop(model_constructor: 'Конструктор модели из tf.keras.applications' = applications.MobileNetV3Large,
                          name             : 'Имя модели'                                  = None                         ,
                          training         : 'Режим обучения (или же тестирования)'        = False                        ,
                          trainable        : 'Разморозка весов'                            = False                        ):
    '''
    Создаёт предобученную базовую модель без головы из заданного конструктора
    из tf.keras.applications, добавляя ей предобработку для диапазона [0, 1].
    '''
    # Формируем базовую модель с подходящей предобработкой.
    backbone = model_constructor(include_top=False,
                                 weights='imagenet',
                                 include_preprocessing=False)
    
    # Берём исходное имя модели, если не задано другое:
    name = name or backbone.name
    
    # Собираем модель с предобработкой и выключаем режим обучения:
    inp = layers.Input((None, None, 3))
    out = layers.Rescaling(2, -1)(inp)
    out = backbone(out, training=training) # Указываем режим обучения/инференса
    backbone = models.Model(inp, out, name=name)
    # Режим обучения нужно выключить, чтобы не обновлять веса в слоях пакетной нормализации!
    # Это нужно во избежание хаотизации.

    # Задаём обучаемость:
    backbone.trainable = trainable
    
    return backbone


class BackboneShapes:
    '''
    Выполняет рассчёт оптимальных размеров входных и выходных тензоров
    для базовой сети (backbone) с учётом числа слоёв пулинга в ней.
    
    Пример:
    >>> bbs = BackboneShapes(true_input_shape=(1200, 1600), desirable_scale=1 / 1.6)
    >>> bbs.input_shape, bbs.output_shape
    ([736, 992], [23, 31])
    '''
    def __init__(self,
                 true_input_shape : 'Размер реальных данных'               = (1080, 1920),
                 desirable_scale  : 'Желаемый коэффициент масштабирования' = 1.          ,
                 num_poolings     : 'Число пулинг слоёв 2х2'               = 5           ,
                 output_filters   : 'Число признаков выходного слоя'       = None        ):
        
        # Если во входном размере указано даже число каналов, то сохраняем их отдельно:
        if hasattr(true_input_shape, '__len__') and len(true_input_shape) > 2:
            self.true_input_shape =      true_input_shape[:2]
            self.input_channels   = list(true_input_shape[2:])
        else:
            self.true_input_shape = a2hw(true_input_shape)
            self.input_channels   = []
        
        # Работаем с числом выходных фильтров, если указано:
        if output_filters is None:
            self.output_filters = []
        elif hasattr(output_filters, '__len__'):
            self.output_filters = list(output_filters)
        else:
            self.output_filters = [output_filters]
        
        # Рассчёт желаемого размера входного изображения с учётом масштабирования:
        self.desirable_input_shape = np.array(self.true_input_shape) * np.array(desirable_scale)
        
        # Коэффициент уменьшения размера карт признаков в результате применения пулинг-слоёв:
        self.pad_scale = 2 ** num_poolings
        
        # Желаемый размер выходной карты признаков:
        self.desirable_output_shape = self.desirable_input_shape / self.pad_scale
        
        # Рассчёт рекомендуемых размеров входа и выхода:
        self.recommended_output_shape = self.desirable_output_shape.astype(int)        # Размер карты признаков
        self.recommended_input_shape  = self.recommended_output_shape * self.pad_scale # Размер входного изображения
        
        # Перевод рекомендуемых размеров в списки:
        self. input_shape = list(self. recommended_input_shape) + self.input_channels
        self.output_shape = list(self.recommended_output_shape) + self.output_filters
        # Это интерфейсная часть классов.
        # Размер переводится в список для удобного добавления числа каналов в случае необходимости.


def Deeplabv3Plus(backbone         : 'Базовая модель для извлечения признаков'                          ,
                  input_tensor     : 'Входной тензор'                                   = None          ,
                  input_shape      : 'Размер входа (если input_tensor не задан)'        = (256, 256, 3) ,
                  input_batch_size : 'Размер батча (если input_tensor не задан)'        = None          ,
                  activation       : 'Тип ф-ии активации на выходе'                     = 'auto'        ,
                  pool_mode        : 'Тип глобалпулинга: {"avg", "max", "both"}'        = 'both'        ,
                  dropout_rate     : 'Доля отбрасываемых признаков для Dropout'         = 0.1           ,
                  out_filters      : 'Число нейронов на выходе (число классов)'         = 1             ,
                  feat_filters     : 'Число нейронов в параллельной с пулингом свёртке' = 256           ,
                  pool_filters     : 'Число нейронов после глобалпулинга'               = 256           ,
                  head_filters     : 'Число нейронов в предпоследней свёртке'           = 256           ,
                  resize_output    : 'Растягивать выход до размера входа'               = True          ,
                  name             : 'Имя модели'                                       ='deeplabv3plus'):
    '''
    Собирает модель сегментации Deeplabv3+.
    '''
    # Доопределяем тип выходной активации, если надо:
    if activation and activation.lower() == 'auto':
        activation = 'softmax' if out_filters > 2 else 'sigmoid'
    
    # Получаем входной тензор и вход для собираемой модели:
    img_input, inputs = InputModel(input_tensor, input_shape, input_batch_size=None)
    # Используется для универсализации способа задавать вход.
    
    # Применяем базовую модель для извлечения признаков:
    x = backbone(img_input)
    
    # Получаем глобализованную карту признаков:
    global_poolings = global_pool_conv2D(x, pool_mode, pool_filters)
    # Если pool_filters == 0, то global_poolings == None!
    
    # Сокращаем число признаков в параллельной с пулингом ветке, если надо:
    if feat_filters:
        x = layers.Conv2D(feat_filters, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    # Это предпредпоследняя свёртка в сети. 
    
    # Формируем из Res-блок на базе global_pool_conv2D, если надо:
    if global_poolings is not None:
        res = layers.Concatenate()([x, global_poolings])
    else:
        res = x
    
    # Предпоследняя свёртка:
    if head_filters:
        x = layers.Conv2D(head_filters, 1, use_bias=False)(res)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    
    # Dropout-слой, если нужно:
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    
    # Выходные слои:
    outputs = layers.Conv2D(out_filters, 1)(x)                                 # Последняя свёртка
    if resize_output:
        outputs = Resizing2DLike(img_input, interpolation='bilinear')(outputs) # Приведение к исходному размеру, если надо

    # Добавляем ф-ию активации или argmax, если надо:
    if activation in {'softmax', 'sigmoid'}:
        outputs = tf.keras.layers.Activation(activation)(outputs) # Ф-ия активации, если нужна
    elif activation and activation.lower() == 'argmax':
        if outputs.shape[-1] > 1:
            outputs = tf.argmax(outputs, axis=-1)                 # Argmax              для прода, если каналов несколько
        else:
            outputs = tf.cast(outputs > 0, tf.int64)              # Пороговая обработка для прода, если канал всего один
    
    # Сборка слоёв в итоговую модель:
    return tf.keras.models.Model(inputs, outputs, name=name)


def make_fractal_unet(backbone, im_size=(256, 256),
                      latent_filters=16,
                      out_filters=4,
                      inner_dropout_rate=0.1,
                      out_dropout_rate=0.5):
    '''
    Создаёт фрактальную U-Net.
    '''
    # Формируем энкодер на основе базовой модели:
    ec = backbone2encoder(backbone)
    
    # Определяем входной слой:
    inp = layers.Input(list(im_size) + [3])
    
    # Формируем список признаков, извлечённых ...
    # ... кодирующей частью из входного слоя:
    features = ec(inp * 2 - 1)
    
    # Вешаем Res-блок с GlobalPooling и Bottleneck на последнюю карту признаков:
    features[-1] = res_block(features[-1], global_pool_2D_with_bottleneck)
    
    ########################################
    # Строим саму сетку фрактальной U-Net: #
    ########################################
    
    node_kwargs = {   'out_filters': out_filters,
                   'latent_filters': latent_filters,
                     'dropout_rate': inner_dropout_rate}
    
    # Сборка всех связей:
    out1, skip1 = fractal_unet_node(skip_connection=features[0], upsampled=None, donwsampled=features[1], use_out_skip=True , **node_kwargs)
    out2, skip2 = fractal_unet_node(skip_connection=features[1], upsampled=out1, donwsampled=features[2], use_out_skip=True , **node_kwargs)
    out3, skip3 = fractal_unet_node(skip_connection=features[2], upsampled=out2, donwsampled=features[3], use_out_skip=True , **node_kwargs)
    out4, skip4 = fractal_unet_node(skip_connection=features[3], upsampled=out3, donwsampled=features[4], use_out_skip=True , **node_kwargs)
    out5, skip5 = fractal_unet_node(skip_connection=features[4], upsampled=out4, donwsampled=features[5], use_out_skip=True , **node_kwargs)
    out6        = fractal_unet_node(skip_connection=features[5], upsampled=out5, donwsampled=None       , use_out_skip=False, **node_kwargs)
    
    out1, skip1 = fractal_unet_node(skip_connection=skip1, upsampled=None, donwsampled=out2, use_out_skip=True , **node_kwargs)
    out2, skip2 = fractal_unet_node(skip_connection=skip2, upsampled=out1, donwsampled=out3, use_out_skip=True , **node_kwargs)
    out3, skip3 = fractal_unet_node(skip_connection=skip3, upsampled=out2, donwsampled=out4, use_out_skip=True , **node_kwargs)
    out4, skip4 = fractal_unet_node(skip_connection=skip4, upsampled=out3, donwsampled=out5, use_out_skip=True , **node_kwargs)
    out5        = fractal_unet_node(skip_connection=skip5, upsampled=out4, donwsampled=out6, use_out_skip=False, **node_kwargs)
    
    out1, skip1 = fractal_unet_node(skip_connection=skip1, upsampled=None, donwsampled=out2, use_out_skip=True , **node_kwargs)
    out2, skip2 = fractal_unet_node(skip_connection=skip2, upsampled=out1, donwsampled=out3, use_out_skip=True , **node_kwargs)
    out3, skip3 = fractal_unet_node(skip_connection=skip3, upsampled=out2, donwsampled=out4, use_out_skip=True , **node_kwargs)
    out4        = fractal_unet_node(skip_connection=skip4, upsampled=out3, donwsampled=out5, use_out_skip=False, **node_kwargs)
    
    out1, skip1 = fractal_unet_node(skip_connection=skip1, upsampled=None, donwsampled=out2, use_out_skip=True , **node_kwargs)
    out2, skip2 = fractal_unet_node(skip_connection=skip2, upsampled=out1, donwsampled=out3, use_out_skip=True , **node_kwargs)
    out3        = fractal_unet_node(skip_connection=skip3, upsampled=out2, donwsampled=out4, use_out_skip=False, **node_kwargs)
    
    out1, skip1 = fractal_unet_node(skip_connection=skip1, upsampled=None, donwsampled=out2, use_out_skip=True , **node_kwargs)
    out2        = fractal_unet_node(skip_connection=skip2, upsampled=out1, donwsampled=out3, use_out_skip=False, **node_kwargs)
    
    _   , skip1 = fractal_unet_node(skip_connection=skip1, upsampled=None, donwsampled=out2, use_out_skip=True , **node_kwargs)
    # К сожалению, тело фрактальной U-Net пока строится вручную.
    
    out = skip1
    out = layers.BatchNormalization()(out)
    
    out_filters = 16
    
    size_before = backend.int_shape(out)
    gap_out = layers.GlobalAveragePooling2D(keepdims=True)(out)
    gmp_out = layers.    GlobalMaxPooling2D(keepdims=True)(out)
    gp_out = layers.Concatenate()([gap_out, gmp_out])
    gp_out = layers.Conv2D(out_filters, 1, use_bias=False)(gp_out)
    gp_out = layers.BatchNormalization()(gp_out)
    gp_out = layers.ReLU()(gp_out)
    gp_out = layers.experimental.preprocessing.Resizing(*size_before[1:3], interpolation='bilinear')(gp_out)
    
    out = layers.Conv2D(out_filters, 1, use_bias=False)(out)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)
    
    out = layers.Concatenate()([out, gp_out])
    out = layers.Dropout(out_dropout_rate)(out)
    
    # Создаём выходной слой:
    out = layers.Conv2D(3, 1, activation='softmax', use_bias=False)(out)
    
    # Формируем итоговую модель:
    model = models.Model(inp, out, name='model')
    
    return model


class MaxFscore(tf.keras.metrics.AUC):
    '''
    # Метрика: максимальное достигаемое значение
    F-меры (maxF), и/или соответствующий ему порог (optTH).
    '''
    # Конструктор класса:
    def __init__(self,
                 mode: 'Определяет, нужна ли только maxF (="F"), только optTH ' + 
                       '(="TH") или оба параметра в виде комплексного числа(="F+TH")',
                 beta: 'Параметр Бетта, приоретизирующий точность и полноту'          = 1    ,
                 **kwargs):
        
        # Парсим параметр "mode":
        self.mode = mode.upper()
        
        use_f  = 'F'  in self.mode # Нужно ли выводить максимальный F-score?
        use_th = 'TH' in self.mode # Нужно ли выводить оптимальный  порог  ?
        
        # Хоть один из двух параметров должен быть запрошен:
        if not (use_f or use_th):
            raise ValueError('Параметр "mode" должен иметь одно из значений: "F", "TH", "F+TH"')
        
        # Если имя явно не указано ...
        if 'name' not in kwargs:
            # ... то определяем его исходя из параметра "mode":
            f_name  = 'F%s' % beta if use_f  else '' # Нужен ли maxF ?
            th_name = 'TH'         if use_th else '' # Нужен ли otpTH?
            kwargs['name'] = f_name + th_name
        
        # Инициируем внутренние переменные класса-родителя ...
        super(MaxFscore, self).__init__(**kwargs)
        
        # ... и дописываем свои:
        self.bs     = beta ** 2 # Квадрат бетты
        self.use_f  = use_f
        self.use_th = use_th
    
    # Возвращает значение метрики:
    def result(self):
        
        # Обновление всех внутренних переменных:
        super(MaxFscore, self).result()
        
        # Извлекаем нужные переменные:
        tp = self. true_positives[:-1] # Верно обнаруженные
        fn = self.false_negatives[:-1] # Пропуски
        fp = self.false_positives[:-1] # Ложные обнаружения
        
        # Пороги, если нужны:
        if self.use_th:
            th = self._thresholds[:-1] 
            
            # Пороги хранятся не в tf.float, а в np.ndarray и их надо сконвертировать:
            th = tf.convert_to_tensor(th, tp.dtype)
        
        p = tp / (tp + fp) # Точность при всех порогах
        r = tp / (tp + fn) # Полнота  при всех порогах
        
        F = (1 + self.bs) * p * r / (self.bs * p + r) # F-мера при всех порогах
        
        # Формируем маску, ислкючающую Nan-ы в F:
        none_nan_mask = ~tf.math.is_nan(F)
        
        # Строим маскированный вектор F-метрик:
        maskedF = F[none_nan_mask]
        
        # Генерируем возвращяемую величину:
        
        # Если требуется оптимальный порог:
        if self.use_th:
            # Строим маскированный вектор порогов:
            maskedTH = th[none_nan_mask]
            
            # Находим индекс, соответствующий максимальной F-мере в маскированном векторе:
            ind = tf.math.argmax(maskedF)
            
            # Получаем оптимальный порог:
            optTH = maskedTH[ind]
            
            # Если нужны и максимальное значение F-меры, и соответствующий порог бинаризации:
            if self.use_f:
                
                # Получаем максимальное значение F-меры:
                maxF = maskedF[ind]
                
                # Возвращаем две величины в виде комплексного числа, т.к. иначе они усреднятся Keras-ом:
                return tf.dtypes.complex(maxF, optTH)
            
            # Если нужен только оптимальный порог:
            else:
                return optTH
        
        # Если нужно только максимальное значение F-меры: 
        else:
            return tf.math.reduce_max(maskedF)


def get_model_elements(model,
                       element_name_part: 'Строка или список строк, одна из которых должна присутствовать в имени элемента' = None,
                       element_class    : 'Класс искомых элементов'                                                         = None,
                       recursive        : 'Рекурсивный режим (искать и в подмоделях)'                                       = True) -> dict:
    '''
    Возвращает словарь, значениями которого являются подходящие под описание элементы модели.
    '''
    model_elements = {}
    
    # Если element_name_part - строка или None, то оборачиваем её в кортеж:
    if isinstance(element_name_part, str) or element_name_part is None:
        element_name_part = (element_name_part,)
    
    # Перебор по всем слоям модели:
    for layer_ind, layer in enumerate(model.layers):
        
        # Если имя тип слоя соответствует искомому ...
        if element_class is None or isinstance(layer, element_class):
            
            # ... и при переборе всех искомых имён ...
            for name_part in element_name_part:
                
                # ... найдётся совпадение с именем текущего слоя ...
                if name_part is None or name_part in layer.name:
                    
                    # ... то добавляем его в всписок:
                    model_elements[layer_ind] = layer
                    break
        
        # Если включён рекурсивный режим, а слой имеет свои слои, то погружаемся внутрь:
        if recursive and hasattr(layer, 'layers'):
            # Получаем словарь подслоёв:
            sub_layers = get_model_elements(layer, element_name_part, element_class, recursive)
            
            # Интегрируем в общий список с добавлением индекса текущего слоя:
            for sub_ind, sub_layer in sub_layers.items():
                model_elements[tuple([layer_ind] + list(sub_ind))] = sub_layer
    
    return model_elements


class MultiGPU():
    '''
    Класс-утилита для работы с несколькими GPU.
    '''
    def __init__(self):
        
        # Инициируем остновные поля:
        self.num = len(tf.config.list_physical_devices('GPU')) # Число доступных GPU
        self.multi = self.num > 1                              # Больше одной GPU?
        self.any   = self.num > 0                              # GPU вообще есть?
        
        # Определяем наиболее подходящую стратегию распределённых вычислений:
        self.strategy = tf.distribute.MirroredStrategy() if self.multi else tf.distribute.get_strategy()
    
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
        
        # Если GPU нет но default задан целым числом, то берём это число как batch_size:
        elif isinstance(default, int  ):
            batch_size = default
        
        # Если GPU нет но default задан вещественным числом, то оспринимаем default как коэффициент масштабирования:
        elif isinstance(default, float):
            batch_size = int(batch_size * default)
        
        # Иначе выводим ошибку:
        else:
            raise ValueError('Параметр "default" дожен быть типа float или int!')
        
        return batch_size


def recompile_model(model, make_train_function=False):
    '''
    Перекомпиляция модели с теми же параметрами.
    '''
    with model.distribute_strategy.scope():
        
        # Повторная компиляция модели с прежними параметрами:
        model.compile_from_config(model.get_compile_config())
        
        # Повторное создание ф-ии обучения, если надо:
        if make_train_function:
            model.make_train_function()


class Tensorboard(callbacks.TensorBoard):
    '''
    Tensorboard-колбек, который сразу запускает
    и проецсс TensorBoard-а в отдельном потоке.
    Останавливается только после завершения процесса,
    в котором экземпляр этого класса был создан.
    '''
    def __init__(self, *args, **kwargs):
        
        # Инициализируем экземпляр родительского класса:
        super().__init__(*args, **kwargs)
        
        # Запускаем сервис
        self.run_service(self.log_dir)
    
    # Запуск TensorBoard-демона фоном:
    @staticmethod
    def run_service(log_dir):
        tbd = tb.program.TensorBoard()
        tbd.configure(bind_all=True, load_fast=False, logdir=log_dir)
        url = tbd.launch()
        return url
    
    # Вывод информации:
    @staticmethod
    def show_tensor_board_info(log_dir):
        string_info  = 'Запускать TensorBoard следует командой:'
        string_info += '\n\t'
        string_info += '\033[1m' # Начало жирного шрифта
        string_info += 'killall -9 tensorboard; nohup tensorboard --logdir='
        string_info += f'"{os.path.abspath(log_dir)}" '
        string_info += '--bind_all --load_fast=false &'
        string_info += '\033[0m' # Конец жирного шрифта
        string_info += '\n'
        string_info += '\n'
        string_info += 'Доступ к TensorBoard осуществляется по адресу:'
        string_info += '\n\t'
        string_info += 'http://%s:6006/' % (os.uname()[1])
        
        print(string_info)
    # Полезно вызывать перед завершением процесса, в котором ...
    # ... экземпляр класса был создан, т.к. с его завершением ...
    # ... прекратится и работа внутреннего TensorBoard-демона. ...
    # ... T.e. потом TensorBoard надо будет запускать уже ...
    # ... внешним процессом, если он вообще будет нужен.
    
    # Аналогичен show_tensor_board_info, но для текущего экземпляра класса:
    def show_info(self):
        self.show_tensor_board_info(self.log_dir)
    
    # Для работы в контексте:
    def __enter__(self                 ): return self
    def __exit__ (self, type, value, tb): self.show_tensor_board_info(self.log_dir)


class KerasHistory:
    '''
    Объект, содержащий историю обучения Keras-модели.
    Позволяет сохранять, загружать и отрисовывать графики.
    '''
    def __init__(self, hist):
        
        # Если подан весь объект History (возвращается от model.fit), ...
        # ... то берём из него только саму историю изменения метрик:
        if isinstance(hist, tf.keras.callbacks.History):
            hist = hist.history
        
        self.hist = hist

    # Загрузка истории из файла, сохранённого через keras.callbacks.CSVLogger:
    @classmethod
    def from_csv(cls, file):
        
        # Загрузка датафрейма:
        df = pd.read_csv(file).set_index('epoch')
        
        # Получаем из дадафрейма обычный словарь истории изменения всех метрик:
        hist = {column:list(df[column]) for column in df.columns if column != 'epoch'}

        # Возвращаем экземпляр класса:
        return cls(hist)
    
    # Группируем метрики по именам:
    def group(self):
        # Инициируем словарь имя_метрики -> словарь_векторов_значений:
        metrics = {}
        # Словарь векторов значений имеет 3 возможных ключа:
        # train, val и test.
        
        # Заполняем словарь метрик:
        for name in self.hist.keys():
            
            # Если название метрики начинается с 'val_':
            if len(name) > 4 and 'val_' == name[:4]:
                
                # Уточняем название метрики:
                true_name = name[4:]
                
                # Записываем значение метрики на проверочной выборке в конец списка:
                metrics[true_name] = metrics.get(true_name, {}) | {'val': self.hist[name]}
            
            # Если название метрики начинается с 'test_':
            elif len(name) > 5 and 'test_' == name[:5]:
                
                # Уточняем название метрики:
                true_name = name[5:]
                
                # Записываем значение метрики на проверочной выборке в конец списка:
                metrics[true_name] = metrics.get(true_name, {}) | {'test': self.hist[name]}
            
            # Иначе это и есть название метрики:
            else:
                
                # Записываем значения метрики на обуч. выборке в начало списка:
                metrics[name] = metrics.get(name, {}) | {'train': self.hist[name]}
        
        return metrics
    
    # Отрисовка графиков истории обучения:
    def show(self):
        
        # Группируем метрики по именам:
        metrics = self.group()

        # Подготовка фона графика, обозначающего изменения скорости обучения:
        bg = None
        if 'lr' in metrics:

            # Берём историю изменения скоростей обучения:
            lr = metrics['lr']['train']

            # Если скорость обучения вообще менялась:
            if len(np.unique(lr)) > 1:
                
                # Нормалилазция диапазона скоростей обучения в логарифмическом масштабе:
                bg = np.log(lr)
                bg -= bg.min()
                bg /= bg.max()
        # Предпологается, что область одинакового lr окрашивает фон отдельным цветом.
        
        # Отрисовка графиков:
        plt.figure(figsize=(24, 6))
        for subplot_ind, (name, plots) in enumerate(metrics.items(), 1):
            plt.subplot(1, len(metrics), subplot_ind)
            plt.title(name)
            plt.grid()
            
            # Если найден всего один график с таким именем, то отображаем его без наворотов:
            if len(plots) == 1:
                plt.plot(plots['train'])
            
            # Если графиков два и более, то добавляем легенду:
            else:
                for key, value in plots.items():
                    plt.plot(value, label=key)
                plt.legend()
            
            # Для функции потерь и скорости обучения применяем логарифмический масштаб:
            if name in ['lr', 'loss']:
                plt.gca().set_yscale('log')
            
            # Красим фон в зависимости от значений lr:
            if bg is not None:
                plt.imshow([bg],
                           extent=(-0.5, len(bg) - 0.5) + plt.ylim(),
                           alpha=0.2,
                           aspect='auto')
    
    # Сохраняет историю в yaml-файл:
    def save(self, file='hist.yaml'):
        
        # Переводим все числовые значения в строки:
        hist = {name: list(map(float, values)) for name, values in self.hist.items()}
        
        # Сохраняем:
        obj2yaml(hist, file)
    
    # Загружает историю из yaml-файла:
    @classmethod
    def load(cls, file='hist.yaml'):
        
        # Загружаем:
        hist = yaml2obj(file)
        
        # Переводим все строковые значения обратно в числа:
        hist = {name: list(map(float, values)) for name, values in hist.items()}
        
        # Оборачиваем в экземпляр класса и возвращаем:
        return cls(hist)


class Warmup(tf.keras.callbacks.Callback):
    '''
    Выполняет прогрев через постепенное увеличение скорости обучения.
    '''
    def __init__(self, steps=10000):
        self.steps = steps # Номер шага, на котором должен быть закончен прогрев
        self.restart() # Инициализация внутренних параметров

    # Инициализация внутренних параметров:
    def restart(self):
        self.target_lr = None  # Инициализация конечной скорости обучения (считывается на первом шаге)
        self.step      = 0     # Инициализация счётчика шагов
    # Вынесено в отдельную ф-ию для возможности начать повторный прогрев в середине обучения.
    
    def on_train_batch_begin(self, batch, logs=None):
        # Если это первый запуск, то запоминаем текущий LR как целевой:
        if self.target_lr is None:
            self.target_lr = backend.get_value(self.model.optimizer.lr)

        # Прирост счётчика шагов:
        self.step += 1
        
        if self.step <= self.steps:
            backend.set_value(self.model.optimizer.lr, self.target_lr * self.step / self.steps)
    
    def on_epoch_end(self, epoch, logs=None):
        # Ведём лог пока прогрев не завершится:
        if self.step <= self.steps:
            logs = logs or {}
            logs["lr"] = backend.get_value(self.model.optimizer.lr)

    on_train_batch_end = on_epoch_end


class TestCallback(callbacks.Callback):
    '''
    Рассчитывает метрики для тестовой выборки в конце каждой эпохи.
    '''
    def __init__(self, *args, **kwargs):
        # Если параметры не переданы, то ставим флаг, указывающий ...
        # ... что брать датасет надо из самой модели:
        self.data_from_model = len(args) == len(kwargs) == 0
        # При использовании Keras-Tuner требуется, чтобы колбек мог быть ...
        # ... полностью скопирован с помощью deepcopy. Однако, сам deepcopy ...
        # ... не поддерживает объекты типа tf.Data и т.п., поэтому тестовый ...
        # ... датасет в таких случаях надо привязывать к самой модели, а не ...
        # ... колбеку. Да, это костыль. Но лучше чичего придумать я не смог.
        
        self.args = args
        self.kwargs = kwargs | {'verbose': 0}
    
    def on_epoch_end(self, epoch, logs={}):
        
        # Рассчёт значения метрики:
        if self.data_from_model:
            metrics_values = self.model.evaluate(self.model.test_data, **self.kwargs)
        else:
            metrics_values = self.model.evaluate(*self.args, **self.kwargs)
        
        # Внесение значения метрики в лог:
        for key, val in zip(self.model.metrics_names, metrics_values):
            logs['test_' + key] = val
    
    # Вшивает тестовые данные в саму модель:
    @staticmethod
    def prepare_model(model, test_data):

        # Тестовые данные в модели изначально храниться не должны:
        assert not hasattr(model, 'test_data')
        
        model.test_data = test_data
    


class PlotHistory(callbacks.Callback):
    '''
    Рассчитывает метрики для тестовой выборки в конце каждой эпохи.
    '''
    def __init__(self, file2save=None, show=True, *args, **kwargs):
        self.file2save = file2save
        self.show = show
    
    def on_epoch_end(self, epoch, logs={}):
        
        # Получаем историю:
        kh = KerasHistory(self.model.history)
        
        # Сохраняем в файл, если надо:
        if self.file2save:
            kh.save(self.file2save)
        
        # Отрисовываем графики, если надо:
        if self.show:
            cls()
            kh.show()
            plt.show()
            print()


class AutoFinetune(callbacks.EarlyStopping):
    '''
    Аналогичен keras.callbacks.EarlyStopping, но работает в 2 этапа:
        1) Обучение модели как есть. Предпологается, что у неё заморожены какие-то слои.
            После соблюдения условий остановки в первый раз начинается второй этап.
        2) Все слои размораживаются, меняется скорость обучения, если нужно. Обучение возобновляется.
            Второй раз остановка выполняется уже окончательно, как заложено в родительской EarlyStopping.
    '''
    def __init__(self,
                 lr_on_finetune       : 'Скорость обучения на тонкой настройке: {"same", float, None}'            = 'same',
                 stop_on_lr_less_then : 'Так же останавливается, если lr < заданного значения на тонкой настройке'= 1e-10 ,
                 do_on_restart        : 'Ф-ия, выполняющаяся в начале второго этапа обучения'                     = None  ,
                 *args, **kwargs):
        # Резервируем параметр восстановления весов для этапа тонкой настройки:
        if len(args) >= 7:
            self.restore_best_weights_after_finetune = args[7]
            args[7] = False
        if 'restore_best_weights' in kwargs:
            self.restore_best_weights_after_finetune = kwargs['restore_best_weights']
            kwargs['restore_best_weights'] = False
        # Иначе "лучшие веса" восстановятся ещё перед тонкой настройкой.
        
        # Инициируем родительский класс:
        super().__init__(*args, **kwargs)
        
        # Добавляем оставшиеся параметры:
        self.lr_on_finetune       = lr_on_finetune
        self.stop_on_lr_less_then = stop_on_lr_less_then
        self.do_on_restart        = do_on_restart
        
        # Флаг тонкой настройки:
        self.is_pretrain = True
    
    # Получить текущую скорость обучения:
    def _get_lr(self):
        return float(backend.get_value(self.model.optimizer.learning_rate))
    
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        
        # Если ещё идёт этап предобучения:
        if self.is_pretrain:
            
            # Если на старне тонкой настройки нужна та же скорость, что и на предобучении, то:
            if isinstance(self.lr_on_finetune, str) and self.lr_on_finetune.lower() == 'same':
                
                # Запоминаем нужную скорость обучения:
                self.lr_on_finetune = self._get_lr()
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        # Если указано нижнее значение скорости обучения:
        if self.stop_on_lr_less_then:
            
            # Останавливаем обучение, если порог уже преодалён:
            if self.stop_on_lr_less_then and self.stop_on_lr_less_then > self._get_lr():
                self.model.stop_training = True
        
        # Если пора переходить на этап точной настройки:
        if self.model.stop_training and self.is_pretrain:
            
            # Меняем флаг этапа:
            self.is_pretrain = False
            
            # Подготовка состояний модели:
            self.model.trainable = True # Размораживаем всю модель
            
            # Повторная компиляция модели с повторным созданием ф-ии обучения:
            recompile_model(self.model, make_train_function=True)
            # Нужно для фиксации разморозки весов.
            
            # Подготовка состояний обучения:
            self.model.stop_training = False # Возвращяем флаг остановки обучения в исходное состояние
            self.on_train_begin()            # Повторно инициируем часть внутренних переменных
            self.start_from_epoch += epoch   # Сдвигаем начало работы вправо на число пройденных эпох
            
            # Запуск внешней ф-ии, если она задана:
            if self.do_on_restart:
                self.do_on_restart()
            
            # Меняем скорость обучения, если надо:
            if self.lr_on_finetune:
                backend.set_value(self.model.optimizer.lr, self.lr_on_finetune)

