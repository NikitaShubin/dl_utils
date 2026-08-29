"""onnx_utils.py.

********************************************
*         Работа с ONNX-моделями.          *
*                                          *
*   Конвертация keras-моделей в ONNX       *
*   (float32), упрощение до float16,       *
*   квантизация до uint8/int8, извлечение  *
*   весов и инференс ONNX в CV-конвейерах. *
*                                          *
* Зависимости:                             *
* • onnx, onnxmltools, onnxruntime;        *
* • tf2onnx - опционально: нужен только    *
*   для keras2onnx(); без него модуль      *
*   импортируется, но вызов keras2onnx()   *
*   бросает ImportError.                   *
*                                          *
* Основные функции:                        *
* • get_weights() - список весов из        *
*   onnx-модели;                           *
* • keras2onnx() - keras -> ONNX в четырёх *
*   вариантах: f32/f16/dyn/stc.            *
*                                          *
* Основные классы:                         *
* • DataReader - подготовка батчей для     *
*   калибровки при статической             *
*   квантизации;                           *
* • ONNXModel - обёртка модели в функтор   *
*   для инференса.                         *
*                                          *
********************************************
.
"""

import onnx
import onnxmltools  # type: ignore[import-untyped]
import onnxruntime  # type: ignore[import-untyped]

# Выясняем, установлен ли tf2onnx; нужен только для keras2onnx():
try:
    import tf2onnx  # type: ignore[import-untyped]

    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False

from pathlib import Path

import numpy as np
from onnxconverter_common.float16 import (  # type: ignore[import-untyped]
    convert_float_to_float16,
)
from onnxruntime import quantization
from tqdm.auto import tqdm

from ml_utils import chw2hwc, hwc2chw, is_channel_first
from utils import rmpath


def get_weights(path: str | Path) -> list:
    """Возвращает список весов.

    https://stackoverflow.com/a/52424141/14474616.
    """
    model = onnx.load(str(path))
    return [onnx.numpy_helper.to_array(i) for i in model.graph.initializer]


class DataReader(quantization.calibrate.CalibrationDataReader):
    """Преобразует генератор данных в объект для калибровки onnx-модели.

    Например, входом служит генератор tf.data.Dataset,
    из которого извлекаются батчи для калибровки.

    Используется при калибровке модели для статической оптимизации.
    """

    def __init__(self, ds, model: str | Path) -> None:
        """Инициализирует калибровочный ридер.

        Определяет имя входа модели и формирует итератор по данным.
        """
        # Определяем имя входа:
        sess = onnxruntime.InferenceSession(
            model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.input_name = sess.get_inputs()[0].name

        # Определяем число элементов:
        self.datasize = len(ds)

        # Формируем итератор:
        self.iter = iter(ds)

    def get_next(self) -> dict[str, np.ndarray]:
        """Возвращает очередной подготовленный для onnx батч.

        По исчерпании итератора возвращается None.
        """
        # Получаем очередную минивыборку из итератора:
        batch = next(self.iter, None)

        # Признаком достижения конца итератора является возвращение None:
        if batch is None:
            return None

        # Возвращаем подготовленный входной тензор:
        return {self.input_name: np.array(batch[0])}


def keras2onnx(
    model,
    f32='f32.onnx',
    f16='f16.onnx',
    dyn='dyn.onnx',
    stc='stc.onnx',
    ds=None,
    tmp_file='tmp.onnx',
    *args: list,
    **kwargs: object,
):
    """Конвертирует keras-модель в onnx-модели нескольких видов.

    - полноценную float32;
    - упрощённую float16;
    - динамическую uint8;
    - статическую int8.

    args и kwargs - параметры, передающиеся напрямую в
    tf2onnx.convert.from_keras.

    Для построения статической модели необходимо задать
    генератор данных, используемый, например, при обучении
    модели. Он нужен для калибровки сети перед дискретизацией.
    """
    # Без tf2onnx конвертация невозможна:
    if not TF2ONNX_AVAILABLE:
        msg = (
            'Для keras2onnx() требуется пакет tf2onnx. '
            'Установите его: pip install tf2onnx'
        )
        raise ImportError(msg)

    # Keras -> ONNX Float32:
    onnx32, _ = tf2onnx.convert.from_keras(model, *args, **kwargs)

    # Сохраняем ONNX Float32, если надо:
    if f32:
        onnx.save_model(onnx32, f32)

    # Конвертируем и сохраняем ONNX Float16, если надо:
    if f16:
        onnx16 = convert_float_to_float16(onnx32)

        onnxmltools.utils.save_model(onnx16, f16)
    else:
        onnx16 = None

    # Если нужна оптимизация до int8:
    if (stc and ds) or dyn:
        # Формируем подготовленную для квантизации модель:
        if f32:
            quantization.quant_pre_process(f32, tmp_file, skip_symbolic_shape=True)
        else:
            onnx.save_model(onnx32, tmp_file)
            quantization.quant_pre_process(tmp_file, tmp_file, skip_symbolic_shape=True)
        # Сохраняем её в tmp_file.

        # Конвертрируем и сохраняем динамический ONNX Uint8, если надо:
        if dyn:
            quantization.quantize_dynamic(
                tmp_file,
                dyn,
                weight_type=quantization.QuantType.QUInt8,
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
    """Обёртка ONNX-модели в функтор для инференса.

    Используется, например, при построении
    конвейера фильтров в video_utils.py.

    Вынесен отдельно от video_utils.py чтобы не
    нагружать последний зависимостями от onnx-библиотек.
    """

    def __init__(self, model, name: str = 'ONNXModel') -> None:
        """Инициализация обёртки."""
        # Сохраняем параметры:
        self.model = model
        self.name = name

        # Инициализируем среду выполнения:
        self.sess = onnxruntime.InferenceSession(
            self.model,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )

        self.inp = self.sess.get_inputs()[0]  # Вход  модели
        self.out = self.sess.get_outputs()[0]  # Выход модели

        # Получаем строковое описание типа входа:
        inp_type = self.inp.type
        self.is_channel_first = is_channel_first(self.inp.shape)

        # Строковое описание должно быть вида "tensor(тип)":
        if not (inp_type.startswith('tensor(') and inp_type.endswith(')')):
            msg = f'Некорректный формат типа входа: "{inp_type}"!'
            raise ValueError(msg)

        # Берём из строкового описания только сам тип тензора:
        inp_type = inp_type[7:-1]

        # Определяемся c требуемым типом входного тензора:
        if inp_type == 'float16':
            self.inp_type = inp_type
        elif inp_type == 'float':
            self.inp_type = np.float32
        else:
            msg = f'Неизвестный тип входа: "{inp_type}"!'
            raise ValueError(msg)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Применение модели к входным данным."""
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
