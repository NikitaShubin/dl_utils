import numpy as np

def ohe(tensor, num_classes):
    return np.eye(num_classes, dtype=np.float32)[tensor]

def nchw2nhwc(tensor): return tensor.transpose((0, 2, 3, 1))
def nhwc2nchw(tensor): return tensor.transpose((0, 3, 1, 2))
def  chw2hwc (tensor): return tensor.transpose(  (1, 2, 0) )
def  hwc2chw (tensor): return tensor.transpose(  (2, 0, 1) )

def is_channel_first(shape):

    if not hasattr(shape, '__len__') or len(shape) not in {3, 4}:
        raise ValueError(f'Параметр `shape` должен быть вектором из 3 или 4 элементов, получен: {shape}!')

    if len(shape) == 4:
        shape = shape[1:]

    if   shape[ 0] < min(shape[1:  ]): return True
    elif shape[-1] < min(shape[ :-1]): return False
    else: raise ValueError(f'Невозможно определить измерение канала в тензоре размерностью {shape}')

