import os

# Используем разные реализации модуля keras в зависимости от переменной
# окружения "KERAS_MODULE":

# Извлекаем тип keras-модуля из переменных окружения:
keras_module = os.getenv('KERAS_MODULE', 'keras').lower()

if   keras_module ==    'keras': import keras
elif keras_module == 'tf_keras': import tf_keras as keras
elif keras_module == 'tf.keras': from tensorflow import keras
elif keras_module == 'tfmot'   : from tensorflow_model_optimization.python.core.keras.compat import keras
else: raise ValueError(f'Неожиданное значение переменной окружения "KERAS_MODULE": "{keras_module}"!')
#print(keras.__version__)

import signal
import contextlib
import numpy as np
import pandas as pd
import tensorboard as tb
from matplotlib import pyplot as plt

from utils import a2hw, obj2yaml

layers       = keras.layers
losses       = keras.losses
models       = keras.models
applications = keras.applications
backend      = keras.backend
optimizers   = keras.optimizers
callbacks    = keras.callbacks
K            = keras.backend


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

        # Рассчёт желаемого размера входного изображения с учётом
        # масштабирования:
        self.desirable_input_shape = np.array(self.true_input_shape) * \
            np.array(desirable_scale)

        # Коэффициент уменьшения размера карт признаков в результате
        # применения пулинг-слоёв:
        self.pad_scale = 2 ** num_poolings

        # Желаемый размер выходной карты признаков:
        self.desirable_output_shape = self.desirable_input_shape / \
            self.pad_scale

        # Рассчёт рекомендуемых размеров входа и выхода:
        self.recommended_output_shape = self.desirable_output_shape.astype(int)         # Размер карты признаков
        self.recommended_input_shape  = self.recommended_output_shape * self.pad_scale  # Размер входного изображения

        # Перевод рекомендуемых размеров в списки:
        self. input_shape = list(self. recommended_input_shape) + \
            self.input_channels
        self.output_shape = list(self.recommended_output_shape) + \
            self.output_filters
        # Это интерфейсная часть классов.
        # Размер переводится в список для удобного добавления числа каналов в
        # случае необходимости.


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

        # Если включён рекурсивный режим, а слой имеет свои слои, то
        # погружаемся внутрь:
        if recursive and hasattr(layer, 'layers'):
            # Получаем словарь подслоёв:
            sub_layers = get_model_elements(layer, element_name_part,
                                            element_class, recursive)

            # Интегрируем в общий список с добавлением индекса текущего слоя:
            for sub_ind, sub_layer in sub_layers.items():
                model_elements[tuple([layer_ind] + list(sub_ind))] = sub_layer

    return model_elements


class SegDataLoader(keras.utils.Sequence):
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

        # Всем ли именам файлов входных и выходных данных можно установить
        # взаимнооднозначное соответствие:
        for inp, out in zip(inps, outs):
            assert ...
            os.path.splitext(inp)[0].lower() == ...
            os.path.splitext(out)[0].lower()

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

        # Замыкаем итератор, чтобы при выходе за
        # реальные индексы шло повторение с начала:
        sample_id = sample_id % len(self.inps)

        # Загружаем изображения:
        inp = cv2.imread(self.inps[sample_iCustomImaged],
                         cv2.IMREAD_COLOR)
        out = cv2.imread(self.outs[sample_id],
                         cv2.IMREAD_GRAYSCALE)

        inp = inp[..., ::-1].astype('float32') / 255.

        # Применяем аугментацию, если она задана:
        if self.transforms:
            inp, out = self.transforms(image=inp, mask=out).values()

        if self.num_classes:
            out = keras.utils.to_categorical(out, num_classes=self.num_classes,
                                             dtype='float32')
        else:
            out = out.astype('float32') / 255.

        return inp, out
        # Важно отметить, что перевод изображения во float32 выполняется до
        # аугментации, а работа с маской (включая создание каналов по  классам)
        # уже после! Цветовые и геометрические изменения точнее выполняются в
        # числах с плавающей точкой, а все  операции с масками являются
        # целочисленными, поэтому их легче проводить в uint8.

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
            plt.subplot(121)
            plt.imshow(x)
            plt.axis(False)
            plt.subplot(122)
            plt.imshow(y)
            plt.axis(False)
            plt.show()


def get_keras_application_model_constructor_list(obj=keras.applications):
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


def get_all_layers(model, except_types=(layers.InputLayer,), recurcive=True):
    '''
    Получает список всех слоёв в модели, включая подмодели.
    '''
    # Делаем except_types кортежем, если он вообще задан:
    if except_types:
        if not isinstance(except_types, (tuple, list, set)):
            except_types = (except_types,)
        else:
            except_types = tuple(except_types)

    all_layers = []
    for layer in model.layers:

        # Если попалась подмодель и включён рекурсивный режим, то лезем внутрь:
        if recurcive and isinstance(layer, models.Model):
            all_layers += get_all_layers(layer, except_types)

        # Если найден слой - вносим в список:
        elif isinstance(layer, layers.Layer):
            if not (except_types and isinstance(layer, except_types)):
                all_layers.append(layer)

        else:
            raise ValueError(
                f'Объект {layer} не является ни слоем, ни подмоделью!'
            )

    return all_layers


def get_hashed_trainable_weights(layer):
    '''
    Строит кортеж тренируемых параметров слоя.
    Используется для проверки изменилось ли хоть что-то в слое после его разморозки.
    '''
    return tuple([tuple(w.numpy().flatten()) for w in layer.trainable_weights])


def try_to_unfreeze_layer(layer):
    '''
    Размораживает слой и возвращает True, если это изменило список обучаемых
    параметров. Используется при последовательной разморозке всех слоёв.
    '''
    trainable_weights = get_hashed_trainable_weights(layer)
    layer.trainable = True
    return trainable_weights != get_hashed_trainable_weights(layer)


def backbone2encoder(backbone, return_outputs=False):
    '''
    Сборка модели извлечения многомасштабных признаков на основе базовой
    модели (backbone). Полезно при формировании U-Net-подобных архитектур,
    кодирующая часть которых является свёрточной частью какой-то предобученной
    модели (backbone).
    '''
    # Создаём модель, если передан конструктор:
    backbone = model_constructor2model(backbone)

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
        # '''

        #input_tensor = K.zeros(inp_shape) if hasattr(K, 'zeros') else keras.ops.zeros(inp_shape)
        input_tensor = np.zeros(inp_shape)
        # '''
        #input_tensor = np.zeros(inp_shape)

        # Собираем новую модель с телом базовой модели, имеющей выход на каждом
        # слое:

        backbone_layers = get_all_layers(backbone, recurcive=False)  # Список всех слоёв модели, кроме входных
        all_outputs=[layer.output for layer in backbone_layers]
        model = keras.models.Model(inputs=backbone.inputs, outputs=all_outputs)

        # Прогоняем нулевой тензор через модель и снимаем тензоры со всех
        # слоёв:
        output_tensors = model(input_tensor)

        # Размер следующего искомого тензора (в 2 раза больше предыдущего):
        cur_size = output_tensors[-1].shape[-2] * 2

        # Перебор всех тензоров, начиная с предпоследнего и к началу:
        for layer, output_tensor in zip(reversed(backbone.layers[:-1]),
                                        reversed( output_tensors[:-1])):
            # Если размер текущего тензора действительно соответствует
            # искомому, то ...
            if len(output_tensor.shape) > 2 and \
                    output_tensor.shape[-2] == cur_size:
                # ... добавляем выход соответствующего слоя в список...
                outputs.append(layer.output)

                # ... и в два раза увеличиваем размер следующего искомого
                # тензора:
                cur_size *= 2

        inp_shape = tuple(inp_shape[1:])

    else:
        # Если размер входного тензора базовой модели полностью определён
        # (None-ов нет), то и размеры всех промежуточных слоёв тоже определены:

        # Размер следующего искомого тензора (в 2 раза больше предыдущего):
        cur_size = backbone.layers[-1].output.shape[-2] * 2

        # Перебор всех слоёв, начиная с предпоследнего и к началу:
        for layer in reversed(backbone.layers[:-1]):
            # Если размер текущего тензора действительно соответствует
            # искомому, то ...
            if len(layer.output.shape) > 2 and layer.output.shape[-2] == cur_size:
                # ... добавляем выход соответствующего слоя в список...
                outputs.append(layer.output)

                # ... и в два раза увеличиваем размер следующего искомого
                # тензора:
                cur_size *= 2

        inp_shape = backbone.input_shape

    # Если самый первый слой меняет размер тензора, то его вход тоже добавляем
    # в список признаков:
    if inp_shape[:-1] != layer.output.shape[:-1]:
        outputs.append(backbone.input)

    # Восстанавливаем последовательность выходов (отменяем реверсию):
    outputs = outputs[::-1]

    # Возвращаем выходы, если нужны только они:
    if return_outputs:
        return outputs

    # Иначе борачиваем в модель и возвращаем её:
    else:
        return keras.models.Model(backbone.inputs, outputs, name='encoder')


def replace_head(model, filters, activation=None, freaze_nohead_layers=True):
    '''
    Заменяет последний свёрточный слой на новый с иной активацией.
    '''

    # Полный список слоёв:
    all_layers = get_all_layers(model, recurcive=False)

    # Ищем первый с конца свёрточный слой:
    for layer_ind in reversed(range(len(all_layers))):
        layer = all_layers[layer_ind]

        # Если слой не является свёрточным, идём дальше:
        if not isinstance(layer, layers.Conv2D):
            continue

        # Извлекаем параметры слоя:
        layer_config = layer.get_config()

        # Если в слое уже нужное число нейронов, то заменять не будем:
        if layer_config['filters'] == filters:

            # Ставим слою новую активацию:
            layer.activation = layers.Activation(activation)

            # Если надо заморозить все слои, кроме текущего:
            if freaze_nohead_layers:
                for layer2freeze in all_layers:
                    if layer2freeze == layer:
                        break
                    layer2freeze.trainable = False
                else:
                    raise Exception('Ошибка заморозки слоёв!')

            # Если этот слой и есть финальный, то пересобирать модель не надо:
            if layer_ind == len(all_layers) - 1:

                # На всякий случай перекомпилим модель и возвращаем:
                return recompile_model(model)

            # Если слой не был последним, возвращаем пересобранную модель:
            else:
                return models.Model(model.inputs,
                                    layer.output,
                                    name=model.name)

        # Если число фильтров не равно желаемому, то придётся действительно
        # заменить слой:
        else:

            # Сначала соберём модель без последнего слоя:
            nohead_model = models.Model(model.inputs,
                                        all_layers[layer_ind - 1].output)

            # Замораживаем её слои, если надо:
            if freaze_nohead_layers:
                nohead_model.trainable = False

            # Берём конфигурацию исходного слоя и заменяем в нём число
            # параметров и активацию:
            layer_config['activation'] = activation
            layer_config['filters'] = filters

            # Собираем новую модель с нужной свёрткой:
            inp = nohead_model.inputs
            out = nohead_model.call(inp)
            out = layers.Conv2D(**layer_config)(out)
            new_model = models.Model(inp, out, name=model.name)

            return new_model

        raise Exception('Здесь код выполняться не должен!')

    else:
        raise Exception('В модели не найден ни один свёрточный слой!')


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
        gp  = layers.Concatenate()([gap, gmp])

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
    with keras.backend.name_scope(name):
        return layers.Add()([inp, net(inp)])


def spatial_pyramid_pooling(x,
                            dilation_rates=[0, 6, 12, 18],
                            num_channels=256,
                            activation='relu',
                            dropout=0.0):
    '''
    Строит пиромидальный пулинг, использующийся, например в DeepLabV3+
    '''
    if dilation_rates:

        # Параметры свёрток:
        kwargs = {
            'filters': num_channels,
            'kernel_size': 3,
            'padding': 'same',
            'use_bias': False,
        }

        xfs = []
        for dilation_rate in dilation_rates:

            # Свёртка с размером 1х1, если нужна:
            if dilation_rate == 0:
                xf = layers.Conv2D(num_channels, 1, use_bias=False)(x)

            # Свёртки 3x3 с различным прореживанием ядра:
            else:
                xf = layers.Conv2D(**kwargs, dilation_rate=dilation_rate)(x)

            xfs.append(xf)

        # Объединение выходов предыдущих слоёв в один тензор:
        xf = layers.Concatenate()(xfs)

    # Если список прореживаний не задан вообще, то берём признаки из входа:
    else:
        xf = x

    # Добавляем к полученным признакам BN и активацию:
    xf = layers.BatchNormalization()(xf)
    xf = layers.Activation(activation)(xf)

    # Вектор признаков из GlobalAveragePooling:
    gp = layers.GlobalAveragePooling2D(keepdims=True)(x)
    gp = layers.Conv2D(num_channels, 1, use_bias=False)(gp)
    gp = layers.BatchNormalization()(gp)
    gp = layers.Activation(activation)(gp)

    # Объединяем признаки пирамидальной свёртки и глобалпулинга уже
    # после свёртки, чтобы не растягивать вектор пулинга до размера других
    # карт признаков:
    xf = layers.Conv2D(num_channels, 1, use_bias=False)(xf)
    gp = layers.Conv2D(num_channels, 1, use_bias=False)(gp)
    out = xf + gp
    # Можно было бы растянуть вектор пулинга до размера других карт признаков
    # и конкатенировать с ними, но тогда последующая свёртка выполняла бы
    # много одинаковых операций. Это выглядело бы понятнее, но было бы менее
    # вычислительно эффективно.

    out = layers.BatchNormalization()(out)
    out = layers.Activation(activation)(out)

    if dropout:
        out = layers.Dropout(dropout)(out)

    return out


def fractal_unet_node(skip_connection = None,
                      upsampled       = None,
                      donwsampled     = None,
                      out_filters        = 4    ,
                      latent_filters     = 4    ,
                      latent_kernel_size = 3    ,
                      use_out_skip       = True ,
                      is_out_layer       = False,
                      dropout_rate       = 0.1  ,
                      spatial_dropout    = False,
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
        if spatial_dropout:
            x0 = layers.SpatialDropout2D(dropout_rate)(x0)
        else:
            x0 = layers.Dropout(dropout_rate)(x0)

    x = layers.Conv2D(latent_filters, 1, use_bias=False)(x0)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(latent_kernel_size, padding='same',
                               use_bias=False)(x)
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
    h, w = like.shape[1:-1]

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
        global_poolings = Resizing2DLike(x)(global_poolings)

        return global_poolings

    # Если ни один из пулингов не использовался, возвращаем None:
    return


def InputModel(input_tensor, input_shape, input_batch_size=None):
    '''
    Получаем и тензор, и вход из тензора или входа.
    Используется для включения базовых моделей в keras.models.Model.
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
        inputs = keras.utils.get_source_inputs(input_tensor)
        # Т.о. все предобработки вовлекаеются в модель.

    return tensor, inputs


def model_constructor2model(model_constructor, **model_kwargs):
    '''
    Создаёт модель по её конструктору
    '''
    # Если передана сама модель, то ничего не делаем:
    if isinstance(model_constructor, models.Model):
        return model_constructor

    # Если указано лишь имя конструктора, ищем его в kera.applications:
    if isinstance(model_constructor, str):
        model_constructor = getattr(applications, model_constructor)

    # Формируем базовую модель с подходящей предобработкой.
    try:
        backbone = model_constructor(include_top=False,
                                     weights='imagenet',
                                     include_preprocessing=False,
                                     **model_kwargs)
    except:
        backbone = model_constructor(include_top=False,
                                     weights='imagenet')
    return backbone


def backbone_with_preprop(model_constructor: 'Конструктор модели из keras.applications' = applications.MobileNetV3Large,
                          name             : 'Имя модели'                               = None                         ,
                          training         : 'Режим обучения (или же тестирования)'     = False                        ,
                          trainable        : 'Разморозка весов'                         = False                        ,
                          as_submodel      : 'Нужно ли вернуть отдельную модель'        = True                         ,
                          as_encoder       : 'Выводить ли не только последний слой, но и промежуточные' = False        ,
                          **model_kwargs):
    '''
    Создаёт предобученную базовую модель без головы из заданного конструктора
    из keras.applications, добавляя ей предобработку для диапазона [0, 1].

    Для предобученных моделей режим обучения лучше выключить,
    чтобы избежать хаотизации в слоях пакетной нормализации!
    '''
    backbone = model_constructor2model(model_constructor, **model_kwargs)

    # Берём исходное имя модели, если не задано другое:
    name = name or backbone.name

    # Собираем модель с предобработкой и выключаем режим обучения:
    inp = layers.Input(model_kwargs.get('input_shape', (None, None, 3)))
    out = layers.Rescaling(2, -1)(inp)

    # Делаем из базовой модели кодировщих с несколькими выходами, если надо:
    if as_encoder:
        backbone = backbone2encoder(backbone)

    # Если нужно создать отдельную модель со всей инкапсуляцией слоёв:
    if as_submodel:

        # Указываем режим обучения/инференса:
        out = backbone(out, training=training)

    # Если икапсуляция в модель будет мешать и нужно раскрыть все слои:
    else:
        out = backbone.call(out, training=training)
    # Полезно для обучения с учётом квантизации (tfmot).

    backbone = models.Model(inputs=inp, outputs=out, name=name)

    # Задаём обучаемость:
    backbone.trainable = trainable

    return backbone


def UNet(backbone        : 'Базовая модель для извлечения признаков'                              ,
         input_tensor    : 'Входной тензор'                                        = None         ,
         input_shape     : 'Размер входа (если input_tensor не задан)'             = (256, 256, 3),
         use_submodels   : 'Инкапсулировать базовую модель и голову как подмодели' = False        ,
         input_batch_size: 'Размер батча (если input_tensor не задан)'             = None         ,
         activation      : 'Тип ф-ии активации на выходе'                          = 'auto'       ,
         pool_mode       : 'Тип глобалпулинга: {"avg", "max", "both"}'             = 'both'       ,
         dropout_rate    : 'Доля отбрасываемых признаков для Dropout'              = 0.1          ,
         spatial_dropout : 'Использовать канальный Dropout вместо обычного'        = False        ,
         out_filters     : 'Число нейронов на выходе (число классов)'              = 1            ,
         feat_filters    : 'Число нейронов перед каждым даунсемплингом'            = 8            ,
         pool_filters    : 'Число нейронов после глобалпулинга'                    = 256          ,
         name            : 'Имя модели'                                            = 'UNet'       ):
    '''
    Собирает модель сегментации Deeplabv3+.
    '''
    # Доопределяем тип выходной активации, если надо:
    if activation and activation.lower() == 'auto':
        activation = 'softmax' if out_filters > 2 else 'sigmoid'

    # Получаем входной тензор и вход для собираемой модели:
    img_input, inputs = InputModel(input_tensor, input_shape,
                                   input_batch_size=input_batch_size)
    # Используется для универсализации способа задавать вход.

    # Применяем базовую модель для извлечения признаков:
    encoder = backbone_with_preprop(backbone,
                                    as_encoder=True,
                                    as_submodel=False)
    features_list = encoder(img_input) if use_submodels \
        else encoder.call(img_input)

    # Добавляем входной тензор к списку признаков:
    features_list.insert(0, img_input)

    # Берём последнюю карту признаков:
    x = features_list.pop()

    # Добавляем глобалпулинг в последнюю карту признаков:
    if pool_filters:

        # Получаем глобализованную карту признаков:
        global_poolings = global_pool_conv2D(x, pool_mode, pool_filters)
        # Если pool_filters == 0, то global_poolings == None!

        # Объединяем последнюю карту признаков с глобализированной картой
        # признаков:
        x = layers.Concatenate()([x, global_poolings])

    # Перебираем все карты признаков, кроме последней, в обратном порядке:
    for features in reversed(features_list):

        # Повышаем размер предыдущей карты признаков в 2 раза:
        x = layers.UpSampling2D(2)(x)

        # Объединяем текущую карту признаков с увеличенной предыдущей:
        x = layers.Concatenate()([features, x])

        # Conv2D + BN + ReLU:
        x = layers.Conv2D(feat_filters, 3,
                          padding='same',
                          use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    # Dropout-слой, если нужно:
    if dropout_rate:
        if spatial_dropout:
            x = layers.SpatialDropout2D(dropout_rate)(x)
        else:
            x = layers.Dropout(dropout_rate)(x)

    # Выходные слои:
    outputs = layers.Conv2D(out_filters, 1)(x)  # Последняя свёртка

    # Добавляем ф-ию активации или argmax, если надо:
    if activation in {'softmax', 'sigmoid'}:  # Ф-ия активации, если нужна:
        outputs = layers.Activation(activation)(outputs)

    elif activation and activation.lower() == 'argmax':

        # Argmax для прода, если каналов несколько:
        if outputs.shape[-1] > 1:
            outputs = K.argmax(outputs, axis=-1)

        # Пороговая обработка для прода, если канал всего один:
        else:
            outputs = K.cast(outputs > 0, 'int64')

    # Сборка слоёв в итоговую модель:
    return keras.models.Model(inputs, outputs, name=name)


def Deeplabv3Plus(backbone         : 'Базовая модель для извлечения признаков'                                          ,
                  bb_preprop_kwargs: 'Параметры ф-ии backbone_with_preprop'                            = {}             ,
                  input_tensor     : 'Входной тензор'                                                  = None           ,
                  input_shape      : 'Размер входа (если input_tensor не задан)'                       = (256, 256, 3)  ,
                  use_submodels    : 'Инкапсулировать базовую модель и голову как подмодели'           = False          ,
                  input_batch_size : 'Размер батча (если input_tensor не задан)'                       = None           ,
                  activation       : 'Тип ф-ии активации на выходе'                                    = 'auto'         ,
                  out_filters      : 'Число нейронов на выходе (число классов)'                        = 1              ,
                  out_dropout_rate : 'Доля отбрасываемых признаков для выходного Dropout'              = 0.1            ,
                  spatial_dropout  : 'Использовать на выходе канальный Dropout вместо обычного'        = False          ,
                  pre_out_filters  : 'Число нейронов на gпредпоследней свёртке'                        = 128            ,
                  en_lvl_4_dec     : 'Уровень карты признаков, использующейся в обход SPP (от 0 до 4)' = 2              ,
                  dec_filters      : 'Число нейронов в свёртке для низкоуровневой карты признаков'     = 48             ,
                  spp_filters      : 'Число нейронов в пирамидальных свёртках'                         = 256            ,
                  spp_dilations    : 'Размеры прореживаний в пирамидальных свёртках'                   = [0, 6, 12, 18] ,
                  spp_dropout      : 'Дропаут в пирамидальных свёртках'                                = 0.             ,
                  name             : 'Имя модели'                                                      = 'deeplabv3plus'):
    '''
    Собирает модель сегментации Deeplabv3+.
    '''

    # Получаем входной тензор и вход для собираемой модели:
    img_input, inputs = InputModel(input_tensor, input_shape,
                                   input_batch_size=input_batch_size)
    # Используется для универсализации способа задавать вход.

    # Применяем базовую модель для извлечения признаков, есл инадо:
    if isinstance(backbone, models.Model) and len(backbone.outputs) < 5:
        raise ValueError('Из переданной модели нельзя извлечь низкоуровневые' +
                         ' признаки! Пожалуйста, передайте имя, конструктор ' +
                         'или сразу модель с несколькими выходами.')

    # Собираем базовую модель, извлекающую промежуточные признаки, если это
    # ещё не сделано:
    if isinstance(backbone, str):
        bb_preprop_kwargs['as_encoder'] = True
        encoder = backbone_with_preprop(backbone, **bb_preprop_kwargs)
    else:
        assert isinstance(backbone, models.Model)

        encoder = backbone

    features_list = encoder(img_input) if use_submodels \
        else encoder.call(img_input)
    assert isinstance(features_list, list)

    # Применяем пирамедельюые свёртки к карте самых верхнеуровневых признаков:
    spp = spatial_pyramid_pooling(features_list[-1],
                                  dilation_rates=spp_dilations,
                                  num_channels=spp_filters,
                                  activation='relu',
                                  dropout=spp_dropout)

    # Уточняем уровень обходной карты признаков:
    if en_lvl_4_dec is not None and en_lvl_4_dec < 0:
        en_lvl_4_dec = en_lvl_4_dec + len(features_list)

    # Если низкоуровневый признак не указан (или указан высокоуровневый), то
    # не используем обходной путь вообще:
    if en_lvl_4_dec in {None, len(features_list) - 1}:
        up_rate = 2 ** (len(features_list) - 1)
        out = layers.UpSampling2D(up_rate, interpolation='bilinear')(spp)

    else:
        # Повышаем разрешение до размера обходной карты признаков:
        up_rate = 2 ** ((len(features_list) - 1) - en_lvl_4_dec)
        if up_rate > 1:
            spp = layers.UpSampling2D(up_rate, interpolation='bilinear')(spp)

        # Берём карту признаков нижнего уровня:
        dec_inp = features_list[en_lvl_4_dec]

        # Conv + BN + Activation в обход SPP:
        dec = layers.Conv2D(dec_filters, 1, use_bias=False)(dec_inp)
        dec = layers.BatchNormalization()(dec)
        dec = layers.Activation('relu')(dec)

        # Объединяем SPP с картой более низкоуровневых признаков:
        out = layers.Concatenate()([spp, dec])

        # Предфинальное трио Conv + BN + Activation:
        out = layers.Conv2D(pre_out_filters, 1, use_bias=False)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)

        # Повышаем разрешение до размера входного изображения:
        up_rate = 2 ** (en_lvl_4_dec)
        if up_rate > 1:
            out = layers.UpSampling2D(up_rate, interpolation='bilinear')(out)

    # Последние Dropout и свёртка:
    if out_dropout_rate:
        dol = layers.SpatialDropout2D if spatial_dropout else layers.Dropout
        out = dol(out_dropout_rate)(out)
    out = layers.Conv2D(out_filters, 1)(out)

    # Доопределяем тип выходной активации, если надо:
    if activation and activation.lower() == 'auto':
        activation = 'softmax' if out_filters > 2 else 'sigmoid'

    # Добавляем ф-ию активации или argmax, если надо:
    if activation in {'softmax', 'sigmoid'}:  # Ф-ия активации, если нужна:
        outputs = layers.Activation(activation)(out)

    elif activation and activation.lower() == 'argmax':

        # Argmax для прода, если каналов несколько:
        if outputs.shape[-1] > 1:
            outputs = K.argmax(outputs, axis=-1)

        # Пороговая обработка для прода, если канал всего один:
        else:
            outputs = K.cast(outputs > 0, 'int64')

    else:
        raise ValueError(f'Неожиданная ф-ия активации: {activation}!')

    # Сборка слоёв в итоговую модель:
    return keras.models.Model(inputs, outputs, name=name)


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


class MaxFscore(keras.metrics.AUC):
    '''
    # Метрика: максимальное достигаемое значение
    F-меры (maxF), и/или соответствующий ему порог (optTH).
    '''
    # Конструктор класса:
    def __init__(self,
                 mode: 'Определяет, нужна ли только maxF (="F"), только optTH ' + 
                       '(="TH") или оба параметра в виде комплексного числа(="F+TH")',
                 beta: 'Параметр Бетта, приоретизирующий точность и полноту' = 1,
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

            # Пороги хранятся не в keras.float, а в np.ndarray и их надо
            # сконвертировать:
            th = keras.ops.convert_to_tensor(th, tp.dtype)

        p = tp / (tp + fp)  # Точность при всех порогах
        r = tp / (tp + fn)  # Полнота  при всех порогах

        F = (1 + self.bs) * p * r / (self.bs * p + r)  # F-мера при всех порогах

        # Формируем маску, ислкючающую Nan-ы в F:
        none_nan_mask = ~keras.ops.isnan(F)

        # Строим маскированный вектор F-метрик:
        maskedF = F[none_nan_mask]

        # Генерируем возвращяемую величину:

        # Если требуется оптимальный порог:
        if self.use_th:
            # Строим маскированный вектор порогов:
            maskedTH = th[none_nan_mask]

            # Находим индекс, соответствующий максимальной F-мере в
            # маскированном векторе:
            ind = K.argmax(maskedF)

            # Получаем оптимальный порог:
            optTH = maskedTH[ind]

            # Если нужны и максимальное значение F-меры, и соответствующий
            # порог бинаризации:
            if self.use_f:

                # Получаем максимальное значение F-меры:
                maxF = maskedF[ind]

                # Возвращаем две величины в виде комплексного числа, т.к. иначе
                # они усреднятся Keras-ом:
                return keras.ops.convert_to_tensor(maxF.numpy() + optTH.numpy() * 1j)
                # Приходится костылить через перевод в numpy, т.к пока конструктор ...
                # ... комплексных чисел в keras не выведен в интерфейсную часть.

            # Если нужен только оптимальный порог:
            else:
                return optTH

        # Если нужно только максимальное значение F-меры: 
        else:
            return keras.ops.max(maskedF)


class FBetaScore(keras.metrics.FBetaScore):
    '''
    Аналогичен keras.metrics.FBetaScore, но работает тензорах любых
    размерностей.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten = keras.layers.Flatten()

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)
        return super().update_state(y_true, y_pred, *args, **kwargs)


class TrainingMode:
    '''
    Контекст, меняющий параметр training для всех слоёв внутри, 
    при применении которых параметр training не был задан явно.
    Используется backend.learning_phase и backend.set_learning_phase.

    Не работает с Keras3.
    '''
    def __init__(self, training_mode=False):

        # training_mode должен быть бинарным:
        assert isinstance(training_mode, bool)

        # learning_phase == True, если training_mode == 
        self.learning_phase = 0 if training_mode else 1

    # При старте контекста временно подменяем learning_phase:
    def __enter__(self):
        # Фиксируем исходное состояние:
        self.external_learning_phase = backend.learning_phase()

        # Перезаписываете нужным состоянием:
        backend.set_learning_phase(self.learning_phase)

    # При завершении восстанавливаем исходное состояние:
    def __exit__ (self, type, value, traceback):
        backend.set_learning_phase(self.external_learning_phase)


class KerasHistory:
    '''
    Объект, содержащий историю обучения Keras-модели.
    Позволяет сохранять, загружать и отрисовывать графики.
    '''

    def __init__(self, hist):

        # Если подан весь объект History (возвращается от model.fit), ...
        # ... то берём из него только саму историю изменения метрик:
        if isinstance(hist, callbacks.History):
            hist = hist.history

        self.hist = hist

    # Загрузка истории из файла, сохранённого через keras.callbacks.CSVLogger:
    @classmethod
    def from_csv(cls, file):

        # Загрузка датафрейма:
        df = pd.read_csv(file).set_index('epoch')

        # Получаем из дадафрейма обычный словарь истории изменения всех метрик:
        hist = {column: list(df[column]) for column in df.columns if column != 'epoch'}

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

        # Сшиваем learning_rate и lr в одну метрику, если в записях
        # присутствуют обе:
        if 'learning_rate' in metrics:
            if 'lr' in metrics:
                lr1 = metrics['learning_rate']
                lr2 = metrics['lr']

                for key in set(lr1.keys()) | set(lr2.keys()):
                    if key not in lr1:
                        lr = lr2[key]
                    elif key not in lr2:
                        lr = lr1[key]
                    else:
                        lr = [lr1_ or lr2_ for lr1_, lr2_ in zip(lr1[key], lr2[key])]

                    metrics['lr'][key] = lr

                del metrics['learning_rate']

            else:
                metrics['lr'] = metrics['learning_rate']
                del metrics['learning_rate']
        # Всё это надо будет удалить когда исправится проблема с дублированием
        # метрик!

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

                # Нормалилазция диапазона скоростей обучения в логарифмическом
                # масштабе:
                bg = np.log(lr)
                bg -= bg.min()
                bg /= bg.max()
        # Предпологается, что область одинакового lr окрашивает фон отдельным
        # цветом.

        # Отрисовка графиков:
        plt.figure(figsize=(24, 6))
        for subplot_ind, (name, plots) in enumerate(metrics.items(), 1):

            # Сами графики:
            plt.subplot(1, len(metrics), subplot_ind)
            plt.title(name)
            plt.grid()

            for key, value in plots.items():
                plt.plot(value, label=key)
            plt.legend()

            # Для функции потерь и скорости обучения применяем логарифмический
            # масштаб:
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


class Warmup(callbacks.Callback):
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
            #self.target_lr = backend.get_value(self.model.optimizer.lr)
            self.target_lr = self.model.optimizer.learning_rate.numpy()

        # Прирост счётчика шагов:
        self.step += 1

        if self.step <= self.steps:
            #backend.set_value(self.model.optimizer.lr, self.target_lr * self.step / self.steps)
            self.model.optimizer.learning_rate = self.target_lr * self.step / self.steps

    def on_epoch_end(self, epoch, logs=None):
        # Ведём лог пока прогрев не завершится:
        if self.step <= self.steps:
            logs = logs or {}
            #logs["lr"] = backend.get_value(self.model.optimizer.lr)
            logs["lr"] = self.model.optimizer.learning_rate.numpy()

    on_train_batch_end = on_epoch_end


class TestCallback(callbacks.Callback):
    '''
    Рассчитывает метрики для тестовой выборки в конце каждой эпохи.
    '''
    def __init__(self, *args, name='test', **kwargs):
        # Если параметры не переданы, то ставим флаг, указывающий ...
        # ... что брать датасет надо из самой модели:
        self.data_from_model = len(args) == len(kwargs) == 0
        # При использовании Keras-Tuner требуется, чтобы колбек мог быть ...
        # ... полностью скопирован с помощью deepcopy. Однако, сам deepcopy ...
        # ... не поддерживает объекты типа tf.Data и т.п., поэтому тестовый ...
        # ... датасет в таких случаях надо привязывать к самой модели, а не ...
        # ... колбеку. Да, это костыль. Но лучше чичего придумать я не смог.

        self.name = name
        self.data_name = name + '_data'
        self.args = args
        self.kwargs = kwargs | {'verbose': 0}

    def on_epoch_end(self, epoch, logs={}):

        # Рассчёт значения метрики:
        if self.data_from_model:
            metrics_values = self.model.evaluate(
                getattr(self.model, self.data_name), **self.kwargs)
        else:
            metrics_values = self.model.evaluate(*self.args, **self.kwargs)

        # Внесение значения метрики в лог:
        for key, val in zip(self.model.metrics_names, metrics_values):
            logs[self.name + '_' + key] = val

    # Вшивает тестовые данные в саму модель:
    @staticmethod
    def prepare_model(model, test_data, data_name='test_data'):

        # Тестовые данные в модели изначально храниться не должны:
        assert not hasattr(model, data_name)

        # Устанавливаем новый атрибут:
        setattr(model, data_name, test_data)


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
                 stop_on_lr_less_than : 'Так же останавливается, если lr < заданного значения на тонкой настройке'= 1e-10 ,
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
        self.stop_on_lr_less_than = stop_on_lr_less_than
        self.do_on_restart        = do_on_restart

        # Флаг тонкой настройки:
        self.is_pretrain = True

    # Получить текущую скорость обучения:
    def _get_lr(self):
        return self.model.optimizer.learning_rate.numpy()                    # для    keras
        #return float(backend.get_value(self.model.optimizer.learning_rate)) # для tf.keras

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
        if self.stop_on_lr_less_than:

            # Останавливаем обучение, если порог уже преодалён:
            if self.stop_on_lr_less_than and self.stop_on_lr_less_than > self._get_lr():
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

            # Если указано нижнее значение скорости обучения:
            if self.stop_on_lr_less_than:

                # Останавливаем обучение, если порог уже преодалён:
                solt = self.stop_on_lr_less_than
                if solt and solt > self._get_lr():
                    self.model.stop_training = True

            # Если пора переходить на этап точной настройки:
            if self.model.stop_training and self.is_pretrain:

                # Меняем флаг этапа:
                self.is_pretrain = False

                # Подготовка состояний модели:
                self.model.trainable = True  # Размораживаем всю модель
                print('!' * 100)
                print('Перекомпиляция!!!!!')
                print('!' * 100)
                self.model.compile_from_config(
                    self.model.get_compile_config())  # Повторная компиляция модели с прежними параметрами
                self.model.make_train_function()      # Повторное создание ф-ии обучения

                # Подготовка состояний обучения:
                self.model.stop_training = False  # Возвращяем флаг остановки обучения в исходное состояние
                self.on_train_begin()             # Повторно инициируем часть внутренних переменных
                self.start_from_epoch += epoch    # Сдвигаем начало работы вправо на число пройденных эпох

                # Запуск внешней ф-ии, если она задана:
                if self.do_on_restart:
                    self.do_on_restart()

                # Меняем скорость обучения, если надо:
                if self.lr_on_finetune:
                    backend.set_value(self.model.optimizer.lr, self.lr_on_finetune)


class EndToEndEarlyStopping(callbacks.EarlyStopping):
    '''
    Обычный EarlyStopping, но хранящий лучшие значения весов и метрик даже
    после перезапуска обучения.
    '''

    def on_train_begin(self, logs=None):

        # Фиксируем лучшие значения метрики и весов:
        best = self.best if hasattr(self, 'best') else None
        best_weights = self.best_weights if hasattr(self, 'best_weights') else None

        # Выполняем оригинальный метод:
        super().on_train_begin(logs)

        # Восстанавливаем зафиксированные лучшие значения метрики и весов:
        self.best = best
        self.best_weights = best_weights


# Мягкий выход из программы (без вывода ошибок):
def soft_exit(*args, **kwargs):
    print('Получен сигнал завершения работы программы.')
    print('Обучение завершается .')
    sys.exit(0)
# Используется в DoOnKill.


class DoOnKill(callbacks.Callback):
    '''
    Обратный вызов, перехватывающий попытку
    завершить работу программы через kill * $PID.

    Вдохновлено этим:
    https://stackoverflow.com/questions/1112343/how-do-i-capture-sigint-in-python
    '''

    # Флаг останова и его изменение в случае поступления сигнала:
    stop = False
    # Является классовой переменной для использования одного состояния во всех
    # экземплярах данного класса. Это нужно для работы сквозь эксперименты в
    # KerasTuner.

    def __init__(self,
                 stop_on='on_train_end',
                 exit_func=soft_exit,
                 stop_signals=[signal.SIGTERM]):
        '''
        signal.SIGTERM = `kill    $PID`
        signal.SIGINT  = `kill -2 $PID` или "Ctrl + C"
        '''
        # Если в качестве обратных вызовов передана строка, ...
        # ... то оборачиваем её в список из одного элемента:
        if isinstance(stop_on, str):
            stop_on = [stop_on]

        # Делаем stop_signals списком, если надо:
        if isinstance(stop_signals, signal.Signals):
            stop_signals = [stop_signals]
        elif not hasattr(stop_signals, '__iter__'):
            raise ValueError('"stop_signals" должен быть экземпляром signals.Signals ' + \
                             f'или списком таких экземпляров. Получен: {type(stop_signals)}.')

        # Увязываем обратный вызов с каждым из сигналов:
        for stop_signal in stop_signals:
            signal.signal(stop_signal, self.set_stop)

        if exit_func:
            self.exit_func = exit_func

            # Привязываем функцию выхода ко всем необходимым событиям:
            for method_name in stop_on:

                # Методы должны соответствовать шаблону on_*_{begin/end}:
                method_name_elements = method_name.split('_')
                if method_name_elements[0 ] != 'on' or             \
                   method_name_elements[-1] not in {'begin', 'end'}:
                    raise ValueError(f'Некорректный метод "{method_name}"!\n' + \
                                      'Имя метода должно соответствовать '    + \
                                      'шаблону on_*_{begin/end}.'               )

                # Устанавливаем заданную функцию каждым из указанных методов:
                if hasattr(super(), method_name):
                    setattr(self, method_name, self.do_on_event)
                else:
                    raise ValueError(f'Неизвестный метод "{method_name}"!')

    # Установка флага останова:
    def set_stop(self, signum, frame):
        print('\n')
        print('┍━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑')
        print('│                                             │')
        print('│ Получен сигнал завершения работы программы! │')
        print('│  Сигнал будет обработан в ближайшее время.  │')
        print('│                                             │')
        print('┕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙')

        self.__class__.stop = True

    # Вызов функции останова, если влаг установлен:
    def do_on_event(self, *args, **kwargs):
        if self.stop:
            self.exit_func(self, *args, **kwargs)
            self.__class__.stop = False
    # По итогу флаг будет возвращён в исходное состояние.
    # Сделано на случай, если exit_func не подразумевает прерывание работы программы.


class CarefulStopping:
    '''
    Создаёт:
    1) коллбек, перехватывающий внешние прерывания обучения (например через kill),
    и вызывающий ошибку только после указанного события;
    2) контекст, перехвающий эту ошибку и корректно завершающий работу программы.

    Полезно чтобы аккуратно завершить обучение без его перезапуска,
    если используется rerun.sh.
    '''

    def __init__(self,
                 error_class=RuntimeError,
                 desc='\n┍━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑' +
                      '\n│                                │' +
                      '\n│ Обучение завершается в связи с │' +
                      '\n│  получением внешнего сигнала!  │' +
                      '\n│                                │' +
                      '\n┕━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙' +
                      '\n'):
        # Тип KerasTuner-ошибки, которая должна вызываться и перехватываться:
        self.error_class = error_class
        self.desc = desc

    # Вызывает заданную ошибку, которая фиксируется на сервере, снимая задачу
    # с воркера, но перехватывается контекстом данного класса, чтобы программа
    # завершилась без ошибок:
    def do_on_kill(self, *args, **kwargs):
        raise self.error_class(self.desc)

    def get_callback(self, event_name='on_train_begin'):
        return DoOnKill(event_name, exit_func=self.do_on_kill)

    # Функция-контекст, перехватывающаяя заданный тип ошибок:
    @contextlib.contextmanager
    def context(self, *args, **kwargs):
        try:
            yield
        except self.error_class:
            print(self.desc)
    # https://vaclavkosar.com/software/Python-Context-Manager-With-Statement-Exception-Handling


class TensorBoardContext:
    '''
    Контекст, запускающий для Tensorboard-колбека соответствующий
    сервис при открытии, и выводящий информацию о том как его
    запустить самостоятельно при закрытии.

    Важно:
    Не закрывает сервис при закрытии контекста, т.к. это не поддерживается
    функциональностью TensorBoard. Закрытие сервися выполняется при
    завершении процесса, вызвавшего выполнение кода. Т.о. контекст
    лучше всего задавать от запуска обучения и до самого завершения
    программы. Пример:

        ```
        ⋮

        from tf_utils import TensorBoardContext ...
        ⋮

        with TensorBoardContext('log/dir') as tb:
            ⋮

            hist = model.fit(..., callbacks=[..., tb, ...], ...)
            ⋮

        ```

    '''
    default_port = 6006
    default_ports = {None, default_port, str(default_port)}

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], callbacks.TensorBoard):
            self.tb_cbk = args[0]

        elif 'tensor_board_callback' in kwargs:
            self.tb_cbk = kwargs['tensor_board_callback']

        else:
            self.tb_cbk = callbacks.TensorBoard(*args, **kwargs)

        # Запоминаем номер порта:
        self.port = kwargs.get('port', self.default_port)

    # Запуск TensorBoard-демона фоном:
    @staticmethod
    def run_service(log_dir):
        tbd = tb.program.TensorBoard()
        tbd.configure(bind_all=True, load_fast=False, logdir=log_dir)
        url = tbd.launch()
        return url

    # Вывод информации:
    @classmethod
    def show_tensor_board_info(cls, log_dir, port=6006):
        string_info = 'Запускать TensorBoard следует командой:'
        string_info += '\n\t'
        string_info += '\033[1m'  # Начало жирного шрифта
        string_info += 'killall -9 tensorboard; nohup tensorboard --logdir='
        string_info += f'"{os.path.abspath(log_dir)}" '
        string_info += '' if port in cls.default_ports else f'--port={port} '
        string_info += '--reload_multifile=true '
        string_info += '--bind_all --load_fast=false &'
        string_info += '\033[0m'  # Конец жирного шрифта
        string_info += '\n'
        string_info += '\n'
        string_info += 'Доступ к TensorBoard осуществляется по адресу:'
        string_info += '\n\t'
        string_info += 'http://%s:%s/' % (os.uname()[1], port)

        print(string_info)
    # Полезно вызывать перед завершением процесса, в котором ...
    # ... экземпляр класса был создан, т.к. с его завершением ...
    # ... прекратится и работа внутреннего TensorBoard-демона. ...
    # ... T.e. потом TensorBoard надо будет запускать уже ...
    # ... внешним процессом, если он вообще будет нужен.

    # Аналогичен run_service, но для текущего экземпляра класса:
    def run(self):
        if self.port in self.default_ports:
            url = self.run_service(self.tb_cbk.log_dir)
        else:
            url = self.run_service(self.tb_cbk.log_dir, self.port)

        # Извлекаем номер порта из полученного URL:
        self.port = str(url[:-1].split(':')[-1])

    # Аналогичен show_tensor_board_info, но для текущего экземпляра класса:
    def show_info(self):
        self.show_tensor_board_info(self.tb_cbk.log_dir, self.port)

    # При старте контекста запускаем TensorBoard-демон:
    def __enter__(self):
        self.run()
        return self.tb_cbk  # Возвращаем сам колбек

    # При завершении выводим информацию о том, как повторно ...
    # ... запустить демон, после завершения текущей программы:
    def __exit__(self, type, value, traceback):
        self.show_info()


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

    return model


class Distiller(keras.Model):
    '''
    Дистилляция.
    https://keras.io/examples/vision/knowledge_distillation/
    '''

    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_tracker = keras.metrics.Mean(name='loss')

        self.gc = 0

    """
    @property
    def metrics(self):
        '''
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics
        '''
        return self.student.metrics
    """

    def compile(self,
                optimizer,
                metrics,
                student_loss_fn,
                distillation_loss_fn='kld',
                alpha=0.1,
                temperature=3):
        super().compile(optimizer=optimizer, metrics=metrics)
        if isinstance(student_loss_fn, str):
            student_loss_fn = keras.losses.get(student_loss_fn)
            assert student_loss_fn is not None
        if isinstance(distillation_loss_fn, str):
            distillation_loss_fn = keras.losses.get(distillation_loss_fn)
            assert distillation_loss_fn is not None
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self,
                     x=None,
                     y=None,
                     y_pred=None,
                     sample_weight=None,
                     allow_empty=False):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            keras.backend.softmax(teacher_pred / self.temperature, axis=1),
            keras.backend.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature**2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Адский костыль:
        self.loss_tracker.update_state(loss)

        return loss

    def call(self, x):
        return self.student(x)