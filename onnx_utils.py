import onnx
import tf2onnx
import onnxmltools
import onnxruntime

import numpy as np

from onnxruntime import quantization
from tqdm import tqdm

from utils import rmpath
from ml_utils import is_channel_first, chw2hwc, hwc2chw


def get_weights(path):
    '''
    Возвращает список весов.
    https://stackoverflow.com/a/52424141/14474616
    '''
    model = onnx.load(path)
    INTIALIZERS = model.graph.initializer
    weights = []
    for initializer in INTIALIZERS:
        W = onnx.numpy_helper.to_array(initializer)
        weights.append(W)

    return weights


class DataReader(quantization.calibrate.CalibrationDataReader):
    '''
    Преобразует генератор данных (например tf.data.Dataset)
    в объект, позволяющий генерировать входы для onnx-модели.

    Используется при калибровке модели для статической оптимизации.
    '''

    def __init__(self, ds, model):
        # Определяем имя входа:
        model = onnxruntime.InferenceSession(
            model,
            providers=['CUDAExecutionProvider',
                       'CPUExecutionProvider']
        )
        self.input_name = model.get_inputs()[0].name

        # Определяем число элементов:
        self.datasize = len(ds)

        # Формируем итератор:
        self.iter = iter(ds)

    # Получает очередной батч из генератора и возвращает
    # его в уже подготовленном для onnx виде:
    def get_next(self):
        # Получаем очередную минивыборку из итератора:
        batch = next(self.iter, None)

        # Признаком достижения конца итератора является возвращение None:
        if batch is None:
            return

        # Возвращаем подготовленный входной тензор:
        return {self.input_name: np.array(batch[0])}


def keras2onnx(model,
               f32='f32.onnx',
               f16='f16.onnx',
               dyn='dyn.onnx',
               stc='stc.onnx',
               ds=None,
               tmp_file='tmp.onnx',
               *args, **kwargs):
    '''
    Сохраняет keras-модель в следующие onnx-модели:
        полноценную float32,
        упрощённую  float16,
        динамическую uint8 ,
        статическую   int8 .

    args и kwargs - параметры, передающиеся напрямую в
    tf2onnx.convert.from_keras.

    Для построения статической модели необходимо задать
    генератор данных, используемый, например, при обучении
    модели. Он нужен для калибровки сети перед дискретизацией.
    '''
    # Keras -> ONNX Float32:
    onnx32, _ = tf2onnx.convert.from_keras(model, *args, **kwargs)

    # Сохраняем ONNX Float32, если надо:
    if f32:
        onnx.save_model(onnx32, f32)

    # Конвертируем и сохраняем ONNX Float16, если надо:
    if f16:
        onnx16 = \
        onnxmltools.utils.float16_converter.convert_float_to_float16(onnx32)

        onnxmltools.utils.save_model(onnx16, f16)
    else:
        onnx16 = None

    # Если нужна оптимизация до int8:
    if (stc and ds) or dyn:

        # Формируем подготовленную для квантизации модель:
        if f32:
            quantization.quant_pre_process(f32, tmp_file,
                                           skip_symbolic_shape=True)
        else:
            onnx.save_model(onnx32, tmp_file)
            quantization.quant_pre_process(tmp_file, tmp_file,
                                           skip_symbolic_shape=True)
        # Сохраняем её в tmp_file.

        # Конвертрируем и сохраняем динамический ONNX Uint8, если надо:
        if dyn:
            quantization.quantize_dynamic(
                tmp_file,
                dyn,
                weight_type=quantization.QuantType.QUInt8
            )
            onnxdn = onnx.load_model(dyn)
        else:
            onnxdn = None

        # Конвертрируем и сохраняем статический ONNX int8, если надо:
        if stc and ds:

            # Генератор калибровочных данных:
            dr = DataReader(tqdm(ds, desc='Калибровка'), tmp_file)

            # Калибровка, конвертация и сохранение модели:
            quantization.quantize_static(tmp_file, stc, dr)
            onnxst = onnx.load_model(stc)
        else:
            onnxst = None

        # Удаляем временный файл:
        rmpath(tmp_file)

    else:
        onnxdn = onnxst = None

    # Возвращаем модели:
    return onnx32, onnx16, onnxdn, onnxst


class ONNXModel:
    '''
    # Обёртка ONNX-модели в функтор для инференса.

    Используется, например, при построении 
    конвейера фильтров в video_utils.py.

    Вынесен отдельно от video_utils.py чтобы не
    нагружать последний зависимостями от onnx-библиотек.
    '''

    def __init__(self, model, name='ONNXModel'):

        # Сохраняем параметры:
        self.model = model
        self.name = name

        # Инициализируем среду выполнения:
        self.sess = onnxruntime.InferenceSession(
            self.model,
            providers=["CUDAExecutionProvider",
                       "CPUExecutionProvider"]
        )

        self.inp = self.sess.get_inputs ()[0]  # Вход  модели
        self.out = self.sess.get_outputs()[0]  # Выход модели

        # Получаем строковое описание типа входа:
        inp_type = self.inp.type
        self.is_channel_first = is_channel_first(self.inp.shape)

        # Строковое описание должно быть вида "tensor(тип)":
        assert inp_type[:7] == 'tensor(' and inp_type[-1] == ')'

        # Берём из строкового описания только сам тип тензора:
        inp_type = inp_type[7:-1]

        # Определяемся c требуемым типом входного тензора:
        if inp_type in {'float16'}:
            self.inp_type = inp_type
        elif inp_type == 'float':
            self.inp_type = np.float32
        else:
            raise ValueError('Неизвестный тип входа: "%s"!' % inp_type)

    # Применение модели к входным данным:
    def __call__(self, image):

        # Подготавливаем данные:
        if self.is_channel_first:
            image = hwc2chw(image)
        data = np.expand_dims(image, 0).astype(self.inp_type)

        # Применяем НС:
        out = self.sess.run([self.out.name], {self.inp.name: data})

        # Придаём выходному тензору нужный вид:
        out = np.array(out)[0, ...]
        if self.is_channel_first:
            out = chw2hwc(out)

        return out