# Для того, чтобы модуль keras_utils использовал именно tf_keras, а не keras:
import os
os.environ['KERAS_MODULE'] = 'tf_keras'
#os.environ['KERAS_MODULE'] = 'tf_keras'

from functools import partial

import tensorflow_model_optimization as tfmot
from inspect import getcallargs
from keras_utils import keras, get_keras_application_model_constructor_list

layers = keras.layers
#IRNv2CustomScaleLayer = keras.src.applications.inception_resnet_v2.CustomScaleLayer

# Настройка, не выполняющая квантизации:
class NoOpQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers    (self, layer                      ): return []
    def get_activations_and_quantizers(self, layer                      ): return []
    def set_quantize_weights          (self, layer, quantize_weights    ): pass
    def set_quantize_activations      (self, layer, quantize_activations): pass
    def get_output_quantizers         (self, layer                      ): return []
    def get_config                    (self                             ): return {}
# https://github.com/tensorflow/model-optimization/issues/874

# Настройка, выполняющая квантизацию слоёв пакетной нормализации:
class DefaultBNQuantizeConfig(NoOpQuantizeConfig):
    def get_output_quantizers(self, layer):
        return [tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False)]
# https://stackoverflow.com/a/63559933


'''
# Настройка по умолчанию:
class DefaultQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # List all of your weights
    weights = {"kernel": tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)}

    # List of all your activations
    activations = {"activation": tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)}

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.weights.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))
        return output

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.activations.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))
        return output

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        count = 0
        for attribute in self.weights.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_weights[count])
                count += 1

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        count = 0
        for attribute in self.activations.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_activations[count])
                count += 1

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer): return []

    def get_config(self): return {}
# https://stackoverflow.com/a/72363215/14474616
'''


class Quantize:
    # Создаём экземпляры классов настроек квантизации:
    bn_quantize_config    = DefaultBNQuantizeConfig()
    no_op_quantize_config =      NoOpQuantizeConfig()
    #df_quantize_config    =   DefaultQuantizeConfig()

    # Собираем из них словарь для keras.saving.custom_object_scope:
    custom_objects = {'DefaultBNQuantizeConfig': DefaultBNQuantizeConfig,
                           'NoOpQuantizeConfig':      NoOpQuantizeConfig,
                     #   'DefaultQuantizeConfig':   DefaultQuantizeConfig,
                     }
    # Контекст для загрузки сохранённых моделей:
    scope = keras.saving.custom_object_scope(custom_objects)

    @classmethod
    def load_model(cls, *args, **kwargs):
        with tfmot.quantization.keras.quantize_scope(), cls.scope:
            return keras.models.load_model(*args, **kwargs)

    # https://davy.ai/tensorflow-quantization-aware-training/
    @classmethod
    def apply_quantization(cls, layer):
        '''
        Подготавливает слой или целую модель для квантизации.
        '''
        # Для слоёв пакетной нормализации:
        if isinstance(layer, layers.BatchNormalization):
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer,
                quantize_config=cls.bn_quantize_config
            )

        # Для других слоёв, к которым нет конкретной реализации квантизации:
        if isinstance(layer, (layers.Rescaling      ,
                              layers.Normalization  ,
                              layers.Multiply       ,
                              layers.ZeroPadding2D  ,
                              layers.DepthwiseConv2D,
                              layers.Resizing       , # Масштабирование значений, используемое в предобработке
                              #IRNv2CustomScaleLayer ,
                             )):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=cls.no_op_quantize_config)

        # Если для этого слоя не нашлось сценария адекватной квантизации:
        return layer

    def __call__(self, model):
        # Фиксируем исходные параметры компиляции модели:
        compile_config = model.get_compile_config()

        # Подготавливаем модели, помечая нужными метками неподдерживаемые изначально слои:
        prepeared_model = keras.models.clone_model(model, clone_function=self.apply_quantization)

        # Сама квантизация:
        with self.scope:
            quantized_model = tfmot.quantization.keras.quantize_model(prepeared_model)

        # Компилируем новую модель с исходными параметрами, если они были:
        if compile_config is not None:
            quantized_model.compile_from_config(compile_config)

        # del model
        # del prepeared_model

        return quantized_model, prepeared_model


def test_aplications(applications=keras.applications, include_top=False):
    '''
    Тестирует на поддержку конвертации в модель обучения с учётом квантизации
    всех доступных конструкторов моделей из объекта applications.
    '''
    # Извлекаем все доступные конструкторы:
    def get_name(constructor):
        return constructor.__name__
    constructors = sorted(list(set(get_keras_application_model_constructor_list(keras.applications))), key=get_name)

    # Создаём кобъект-квантизатор:
    quantize = Quantize()

    # Список типичных ошибок, которые могут возникнуть:
    errors = ['Shape must be rank 4 but is rank 5'    , # Причину этой ошибки пока не понял
              'keras Model inside another keras Model', # Использована вложенная модель
              'tf.__operators__.add'                  ] # Использована простая сумма тензоров, вместо слоя Add

    # Перебираем все конструкторы из списка:
    for constructor in constructors:

        # Собираем саму модель из конструктора:
        kwargs = {'include_top' :  include_top,
                  'weights'     :  None, # 'imagenet',
                  'input_shape' : (None, None, 3)}
        if 'include_preprocessing' in getcallargs(constructor):
            kwargs['include_preprocessing'] = False
        bb = constructor(**kwargs)

        # Пробуем квантизировать и выводим суть проблемы, если возникла:
        try:
            # Сама квантизация:
            bb = quantize(bb)

            # Выводим имя модели зелёным текстом, если всё прошло удачно:
            print('\t' * 3, '\033[92m', bb.name, '\033[0m', sep='', end='\n')

        # Перехватываем ошибку:
        except Exception as e:

            # Выводим имя модели красным текстом, т.к. что-то пошло не так:
            print('\033[91m', bb.name, '\033[0m', sep='', end=': ')

            # Перебираем типовые проблемы:
            for error in errors:

                # Если текущая проблема - типовая, то выводим лишь её обозначение:
                if error in str(e):
                    print(error)
                    break

            # Если ошибка не относится к типовым, выводим всю информацию:
            else:
                print(e)

    # Удаляем модель, чтобы не занимать память:
    del bb


def prune_low_magnitude(layer, **kwargs):
    '''
    Готовит к прунингу только поддерживаемые слои:
    '''
    # Кортеж неподдерживаемых слоёв:
    not_supported_layers = (keras.layers.Resizing,
                            keras.layers.Rescaling)

    # Если слой из неподдерживаемого списка - возвращаем без изменений:
    if isinstance(layer, not_supported_layers):
        return layer

    # Если передана модель - рекурсивно обрабатываем все входящие в неё слои:
    if isinstance(layer, keras.models.Model):
        return keras.models.clone_model(
            layer,
            clone_function=partial(prune_low_magnitude, **kwargs)
        )

    # Если передан обычный поддерживаемый слой - возвращаем его модификацию:
    return tfmot.sparsity.keras.prune_low_magnitude(layer, ** kwargs)


# Убрать обвес для прунинга с модели:
unprune = tfmot.sparsity.keras.strip_pruning


def get_pruning_callbacks(tensorboard_dir=None):
    '''
    Создаёт список колбеков для прунинга
    '''

    cbs = [tfmot.sparsity.keras.UpdatePruningStep()]

    if tensorboard_dir:
        cbs.append(
            tfmot.sparsity.keras.PruningSummaries(
                log_dir=tensorboard_dir
            )
        )

    return cbs