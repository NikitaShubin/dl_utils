"""Тесты для модуля onnx_utils (работа с ONNX-моделями)."""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest

# Моки для зависимостей перед импортом модуля:
sys.modules['tf2onnx'] = MagicMock()

# Импортируем после установки моков:
import onnx_utils  # noqa: E402
from onnx_utils import DataReader, ONNXModel, get_weights, keras2onnx  # noqa: E402


class TestGetWeights:
    """Тесты для функции get_weights."""

    def test_returns_all_initializers(self) -> None:
        """Возвращает numpy-массивы всех инициализаторов модели."""
        initializer_1 = Mock()
        initializer_2 = Mock()
        mock_model = Mock()
        mock_model.graph.initializer = [initializer_1, initializer_2]
        expected = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        with (
            patch('onnx.load', return_value=mock_model),
            patch('onnx.numpy_helper.to_array', side_effect=expected) as mock_to_array,
        ):
            result = get_weights('model.onnx')

        np.testing.assert_array_equal(result, expected)
        assert mock_to_array.call_args_list == [
            call(initializer_1),
            call(initializer_2),
        ]

    def test_empty_initializers(self) -> None:
        """Модель без инициализаторов -> пустой список."""
        mock_model = Mock()
        mock_model.graph.initializer = []

        with (
            patch('onnx.load', return_value=mock_model),
            patch('onnx.numpy_helper.to_array'),
        ):
            result = get_weights('model.onnx')

        assert result == []


class TestDataReader:
    """Тесты для класса DataReader."""

    def test_initialization(self) -> None:
        """Инициализация: имя входа берётся из первого входа сессии."""
        mock_input = Mock()
        mock_input.name = 'input'
        ds = [(np.ones((1, 3)), 0), (np.ones((1, 3)), 1)]

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            reader = DataReader(ds, 'model.onnx')

        assert isinstance(reader, DataReader)
        assert reader.input_name == 'input'
        assert reader.datasize == 2
        mock_session.assert_called_once_with(
            'model.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )

    def test_initialization_with_path(self) -> None:
        """Путь как pathlib.Path передаётся в InferenceSession как есть."""
        mock_input = Mock()
        mock_input.name = 'input'
        ds = [(np.ones((1, 3)), 0)]
        model_path = Path('model.onnx')

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            reader = DataReader(ds, model_path)

        assert reader.input_name == 'input'
        mock_session.assert_called_once_with(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )

    def test_get_next_returns_prepared_batches(self) -> None:
        """get_next возвращает входы numpy-словарями и None в конце."""
        mock_input = Mock()
        mock_input.name = 'input'
        batches = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
        ds = [(batch, label) for label, batch in enumerate(batches)]

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            reader = DataReader(ds, 'model.onnx')
            result_1 = reader.get_next()
            result_2 = reader.get_next()
            result_end = reader.get_next()

        assert result_1 is not None
        assert result_2 is not None
        np.testing.assert_array_equal(result_1['input'], batches[0])
        np.testing.assert_array_equal(result_2['input'], batches[1])
        assert result_end is None


class TestKeras2Onnx:
    """Тесты для функции keras2onnx."""

    def test_raises_without_tf2onnx(self) -> None:
        """Без tf2onnx вызов бросает ImportError."""
        with (
            patch.object(onnx_utils, 'TF2ONNX_AVAILABLE', new=False),
            pytest.raises(ImportError, match='tf2onnx'),
        ):
            keras2onnx(Mock())

    def test_exports_f32_and_f16(self) -> None:
        """Сохраняет f32 и f16 и возвращает кортеж из четырёх моделей."""
        model = Mock()
        onnx32 = Mock()
        onnx16 = Mock()

        with (
            patch(
                'onnx_utils.tf2onnx.convert.from_keras',
                return_value=(onnx32, None),
            ) as mock_from_keras,
            patch('onnx.save_model') as mock_save_model,
            patch(
                'onnx_utils.convert_float_to_float16',
                return_value=onnx16,
            ) as mock_convert,
            patch('onnxmltools.utils.save_model') as mock_save_16,
        ):
            models = keras2onnx(model, dyn=None, stc=None)

        assert models == (onnx32, onnx16, None, None)
        mock_from_keras.assert_called_once_with(model)
        mock_save_model.assert_called_once_with(onnx32, 'f32.onnx')
        mock_convert.assert_called_once_with(onnx32)
        mock_save_16.assert_called_once_with(onnx16, 'f16.onnx')

    def test_f16_disabled(self) -> None:
        """f16=None -> второй элемент кортежа None, конвертация не выполняется."""
        onnx32 = Mock()

        with (
            patch('onnx_utils.tf2onnx.convert.from_keras', return_value=(onnx32, None)),
            patch('onnx.save_model'),
            patch('onnx_utils.convert_float_to_float16') as mock_convert,
            patch('onnxmltools.utils.save_model') as mock_save_16,
        ):
            models = keras2onnx(Mock(), f16=None, dyn=None, stc=None)

        assert models == (onnx32, None, None, None)
        mock_convert.assert_not_called()
        mock_save_16.assert_not_called()

    def test_dynamic_quantization(self) -> None:
        """Параметр dyn задан -> динамическая квантизация до uint8."""
        onnx32 = Mock()
        onnx_dyn = Mock()

        with (
            patch('onnx_utils.tf2onnx.convert.from_keras', return_value=(onnx32, None)),
            patch('onnx.save_model') as mock_save_model,
            patch('onnx_utils.quantization.quant_pre_process') as mock_pre_process,
            patch('onnx_utils.quantization.quantize_dynamic') as mock_quantize_dynamic,
            patch('onnx.load_model', return_value=onnx_dyn) as mock_load_model,
            patch('onnx_utils.rmpath') as mock_rmpath,
        ):
            models = keras2onnx(Mock(), f32=None, f16=None, dyn='dyn.onnx')

        assert models == (onnx32, None, onnx_dyn, None)
        mock_save_model.assert_called_once_with(onnx32, 'tmp.onnx')
        mock_pre_process.assert_called_once_with(
            'tmp.onnx',
            'tmp.onnx',
            skip_symbolic_shape=True,
        )
        mock_quantize_dynamic.assert_called_once_with(
            'tmp.onnx',
            'dyn.onnx',
            weight_type=onnx_utils.quantization.QuantType.QUInt8,
        )
        mock_load_model.assert_called_once_with('dyn.onnx')
        mock_rmpath.assert_called_once_with('tmp.onnx')

    def test_static_quantization(self) -> None:
        """Параметры stc и ds заданы -> статическая квантизация до int8."""
        onnx32 = Mock()
        onnx_stc = Mock()
        ds = [(np.ones((1, 3)), 0)]
        mock_input = Mock()
        mock_input.name = 'input'

        with (
            patch('onnx_utils.tf2onnx.convert.from_keras', return_value=(onnx32, None)),
            patch('onnx.save_model'),
            patch('onnx_utils.quantization.quant_pre_process'),
            patch('onnx_utils.quantization.quantize_static') as mock_quantize_static,
            patch('onnx.load_model', return_value=onnx_stc) as mock_load_model,
            patch('onnxruntime.InferenceSession') as mock_session,
            patch('onnx_utils.rmpath') as mock_rmpath,
        ):
            mock_session.return_value.get_inputs.return_value = [mock_input]
            models = keras2onnx(
                Mock(),
                f32=None,
                f16=None,
                dyn=None,
                stc='stc.onnx',
                ds=ds,
                tmp_file='prep.onnx',
            )

        assert models == (onnx32, None, None, onnx_stc)
        assert isinstance(mock_quantize_static.call_args.args[2], DataReader)
        mock_load_model.assert_called_once_with('stc.onnx')
        mock_rmpath.assert_called_once_with('prep.onnx')

    def test_passes_kwargs_to_from_keras(self) -> None:
        """Дополнительные kwargs передаются в tf2onnx.convert.from_keras."""
        model = Mock()
        onnx32 = Mock()

        with (
            patch(
                'onnx_utils.tf2onnx.convert.from_keras',
                return_value=(onnx32, None),
            ) as mock_from_keras,
            patch('onnx.save_model'),
        ):
            keras2onnx(model, f16=None, dyn=None, stc=None, target='cuda', opset=13)

        mock_from_keras.assert_called_once_with(model, target='cuda', opset=13)

    def test_accepts_pathlib_paths(self) -> None:
        """Пути как pathlib.Path уходят в onnx-библиотеки без изменений."""
        onnx32 = Mock()
        onnx16 = Mock()
        onnx_dyn = Mock()
        f32_path = Path('f32.onnx')
        f16_path = Path('f16.onnx')
        dyn_path = Path('dyn.onnx')
        tmp_path = Path('tmp.onnx')

        with (
            patch(
                'onnx_utils.tf2onnx.convert.from_keras',
                return_value=(onnx32, None),
            ),
            patch('onnx.save_model') as mock_save_model,
            patch(
                'onnx_utils.convert_float_to_float16',
                return_value=onnx16,
            ) as mock_convert,
            patch('onnxmltools.utils.save_model') as mock_save_16,
            patch('onnx_utils.quantization.quant_pre_process') as mock_pre_process,
            patch('onnx_utils.quantization.quantize_dynamic') as mock_quantize_dynamic,
            patch('onnx.load_model', return_value=onnx_dyn) as mock_load_model,
            patch('onnx_utils.rmpath') as mock_rmpath,
        ):
            models = keras2onnx(
                Mock(),
                f32=f32_path,
                f16=f16_path,
                dyn=dyn_path,
                tmp_file=tmp_path,
            )

        assert models == (onnx32, onnx16, onnx_dyn, None)
        mock_save_model.assert_called_once_with(onnx32, f32_path)
        mock_convert.assert_called_once_with(onnx32)
        mock_save_16.assert_called_once_with(onnx16, f16_path)
        mock_pre_process.assert_called_once_with(
            f32_path,
            tmp_path,
            skip_symbolic_shape=True,
        )
        mock_quantize_dynamic.assert_called_once_with(
            tmp_path,
            dyn_path,
            weight_type=onnx_utils.quantization.QuantType.QUInt8,
        )
        mock_load_model.assert_called_once_with(dyn_path)
        mock_rmpath.assert_called_once_with(tmp_path)


class TestONNXModel:
    """Тесты для класса ONNXModel."""

    def test_init_channel_last(self) -> None:
        """NHWC float-вход -> inp_type np.float32, is_channel_first=False."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'tensor(float)'
        mock_input.shape = [1, 4, 224, 3]
        mock_output = Mock()
        mock_output.name = 'output'
        mock_output.type = 'tensor(float)'
        mock_output.shape = [1, 3, 224, 224]

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            model = ONNXModel('model.onnx')

        assert model.name == 'ONNXModel'
        assert model.inp_type is np.float32
        assert model.is_channel_first is False

    def test_init_channel_first_float16(self) -> None:
        """CHW float16-вход -> inp_type 'float16', is_channel_first=True."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'tensor(float16)'
        mock_input.shape = [1, 3, 224, 224]
        mock_output = Mock()
        mock_output.name = 'output'
        mock_output.type = 'tensor(float16)'

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            model = ONNXModel('model.onnx', name='net')

        assert model.name == 'net'
        assert model.inp_type == 'float16'
        assert model.is_channel_first is True

    def test_init_unknown_input_type_raises(self) -> None:
        """Неизвестный тип входа -> ValueError."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'tensor(int64)'
        mock_input.shape = [1, 4, 224, 3]

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [Mock()]
            with pytest.raises(ValueError, match='int64'):
                ONNXModel('model.onnx')

    def test_init_malformed_input_type_raises(self) -> None:
        """Тип входа без обёртки tensor(...) -> ValueError."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'float32'
        mock_input.shape = [1, 224, 3]

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [Mock()]
            with pytest.raises(
                ValueError,
                match='Некорректный формат типа входа',
            ):
                ONNXModel('model.onnx')

    def test_call_channel_last(self) -> None:
        """Применение к HWC-изображению без смены порядка каналов."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'tensor(float)'
        mock_input.shape = [1, 4, 224, 3]
        mock_output = Mock()
        mock_output.name = 'output'
        mock_output.type = 'tensor(float)'
        mock_output.shape = [1, 4, 224, 3]
        batch_out = np.zeros((1, 4, 224, 3))

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            mock_session.return_value.run.return_value = [batch_out]
            model = ONNXModel('model.onnx')
            image: np.ndarray = np.ones((4, 224, 3), dtype=np.float32)
            result = model(image)

        np.testing.assert_array_equal(result, batch_out)
        run_call = mock_session.return_value.run.call_args
        assert run_call.args[0] == ['output']
        assert list(run_call.args[1].keys()) == ['input']
        np.testing.assert_array_equal(
            run_call.args[1]['input'],
            np.expand_dims(image, 0),
        )

    def test_call_channel_first(self) -> None:
        """Применение к HWC-изображению со сменой порядка каналов."""
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.type = 'tensor(float)'
        mock_input.shape = [1, 3, 224, 224]
        mock_output = Mock()
        mock_output.name = 'output'
        mock_output.type = 'tensor(float)'
        mock_output.shape = [1, 224, 224, 3]
        batch_out = np.zeros((3, 224, 224))

        with patch('onnxruntime.InferenceSession') as mock_session:
            mock_session.return_value.get_inputs.return_value = [mock_input]
            mock_session.return_value.get_outputs.return_value = [mock_output]
            mock_session.return_value.run.return_value = [batch_out]
            model = ONNXModel('model.onnx')
            image: np.ndarray = np.ones((224, 224, 3), dtype=np.float32)
            result = model(image)

        np.testing.assert_array_equal(result, np.transpose(batch_out, (1, 2, 0)))
        run_call = mock_session.return_value.run.call_args
        expected_data = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        np.testing.assert_array_equal(run_call.args[1]['input'], expected_data)


if __name__ == '__main__':
    pytest.main([__file__])


def test_reimport_without_tf2onnx_falls_back() -> None:
    """Без tf2onnx модуль импортируется, а флаг выставляется в False."""
    blocked = {name: None for name in sys.modules if name == 'tf2onnx'}
    with patch.dict(sys.modules, blocked):
        importlib.reload(onnx_utils)
    assert onnx_utils.TF2ONNX_AVAILABLE is False
    with pytest.raises(ImportError, match='tf2onnx'):
        keras2onnx(Mock())
