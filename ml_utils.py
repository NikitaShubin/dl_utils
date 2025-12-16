import numpy as np
from sklearn.model_selection import train_test_split

from utils import isint, isfloat, flatten_list


def ohe(tensor, num_classes):
    '''
    One-hot encoding для Numpy-тензора.
    '''
    return np.eye(num_classes, dtype=np.float32)[tensor]


def nchw2nhwc(tensor): return tensor.transpose((0, 2, 3, 1))
def nhwc2nchw(tensor): return tensor.transpose((0, 3, 1, 2))
def chw2hwc(tensor): return tensor.transpose((1, 2, 0))
def hwc2chw(tensor): return tensor.transpose((2, 0, 1))


def is_channel_first(shape):
    '''
    По размерам тензора изображения определяет, стоит ли измерение каналов
    перед измерениями размеров самого изображения.
    '''

    if not hasattr(shape, '__len__') or len(shape) not in {3, 4}:
        raise ValueError('Параметр `shape` должен быть вектором ' +
                         f'из 3 или 4 элементов, получен: {shape}!')

    # Множество возможного числа каналов изображения:
    num_channels = {1, 2, 3, 4}

    # Если тензор четырёхмерный, то отбрасываем первое измерение (батч):
    if len(shape) == 4:
        shape = shape[1:]

    if shape[0] < min(shape[1:]) and shape[0] in num_channels:
        return True  # (B)CHW
    elif shape[-1] < min(shape[:-1]) and shape[0] in num_channels:
        return False  # (B)HWC
    else:
        raise ValueError('Невозможно определить измерение канала ' +
                         f'в тензоре размерностью {shape}')


def soft_train_test_split(*args, test_size, random_state=0):
    '''
    Разделение данных на две подвыборки. Аналогичен train_test_split,
    но работает и при экстремальных случаях вроде нулевой длины
    выборки или нулевого размера одного из итоговых подвыборок.
    '''

    try:
        return train_test_split(*args, test_size=test_size, shuffle=True,
                                random_state=random_state)

    # Если случай экстремальный:
    except ValueError:

        # Рассчитываем относительный размер тестовой выборки:
        if isfloat(test_size):
            test_size_ = test_size
        elif len(args[0]) > 0:
            test_size_ = test_size / len(args[0])
        else:               # Если выборка нулевой длины, то
            test_size_ = 1  # избегаем деления на ноль.

        # Создаём список пустых объектов того же типа, что были в args:
        emptys = [type(arg)() for arg in args]

        # Если тестовая выборка должна быть больше проверочной - отдаём
        # всё ей, иначе всё отходит обучающей:
        pairs = zip(emptys, args) if test_size_ > 0.5 else zip(args, emptys)
        return flatten_list([list(pair) for pair in pairs], depth=1)


def train_val_test_split(*args, val_size=0.2, test_size=0.1, random_state=0):
    '''
    Режет выборку на обучающую, проверочную и тестовую.
    '''
    # Если на входе пустой список, возвращаем 3 пустых списка:
    if len(args[0]) == 0:
        return [], [], []

    # Величины val_size и test_size должны адекватно соотноситься с размером
    # выборки:
    if isint(val_size) and isint(test_size):
        assert val_size + test_size <= len(args[0])
    elif isfloat(val_size) and isfloat(test_size):
        assert val_size + test_size <= 1.
    else:
        raise ValueError('Параетры `val_size` и `test_size` должны быть ' +
                         f'одного типа. Получены: {val_size} и {test_size}!')

    # Получаем тестовую выборку:
    trainval_test = soft_train_test_split(*args, test_size=test_size,
                                          random_state=random_state)
    train_val = trainval_test[::2]
    test = trainval_test[1::2]

    # Если val_size задан целым числом, то используем его как есть:
    if isint(val_size):
        val_size_ = val_size

    # Если val_size - вещественное число, то долю надо перерасчитать:
    elif isfloat(val_size):

        # Если при этом test_size целочисленного типа, то переводим его в
        # дроби:
        if isint(test_size):
            test_size = test_size / len(args[0])

        # Перерасчитываем val_size с учётом уменьшения ...
        # ... выборки после отделения тестовой стоставляющей:
        val_size_ = val_size / (1. - test_size) if test_size < 1. else 0

    else:
        raise ValueError('Неподходящий тип "' +
                         str(type(val_size)) + '" переменной val_size!')

    # Разделяем оставшуюся часть выборки на обучающую и проверочную:

    train_val = soft_train_test_split(*train_val, test_size=val_size_,
                                      random_state=random_state)
    train = train_val[::2]
    val = train_val[1::2]

    triples = zip(train, val, test)
    
    return flatten_list([list(triple) for triple in triples], depth=1)