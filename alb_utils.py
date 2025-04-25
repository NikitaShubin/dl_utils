import albumentations as A
import numpy as np

from utils import overlap_with_alpha


class RandomDust(A.RandomFog):
    '''
    Наложение пылевой взвеси на изображение.
    Действует по аналогии с аlbumentations.RandomFog, но имеет
    доп. параметр (цвет пыли) и не размывает изображение.
    '''
    def __init__(self, color=(255, 255, 255), *args, **keargs):
        super().__init__(*args, **keargs)
        self.color = color

    def apply(self,
              img,
              **params):

        # Дополняем параметры значениями по-умолчанию:
        default_params = {'fog_coef': 0.1,
                          'haze_list': None}
        params = default_params | params

        # Определяем размеры и число каналов исходного изображения:
        height, width, channels = img.shape

        # Инициируем маску пыли с альфа-каналом:
        dust = np.zeros((height, width, channels + 1), dtype=img.dtype)

        # Заполняем альфа-канал маски пыли:
        dust[..., -1] = super().apply(np.zeros_like(img), **params)[..., 0]

        # Заполняем остальные каналы маски пыли:
        for channel, value in enumerate(self.color):
            dust[..., channel] = value

        # Накладываем пыль на исходное изображение и возвращаем результат:
        return overlap_with_alpha(img, dust)


class AlbTransforms:
    '''
    Расширяет функциональность albumentation-преобразований.
    '''
    def __init__(self, transform=None):
        # Если уже передан функтер:
        if hasattr(transform, '__call__'):
            self.transforms = [transform]
            self.transform  =  transform

        # Если итерируемый объект, то объединяем в новый функтор:
        elif hasattr(transform, '__iter__'):
            self.transforms =           transform
            self.transform  = A.Compose(transform)

        # Если преобразование вообще не задано:
        elif transform is None:
            self.transforms = [None]
            self.transform  =  None

        else:
            raise ValueError('Ожидается, что параметр "transform" будет функтером, ' + \
                             f'итерируемым объектом или None. Получен {type(transform)}!')

    # Собирает бъединённое преобразование из списка преобразований:
    @staticmethod
    def compose_transforms(transforms, return_None=False):
        if hasattr(transforms, '__call__'):
            return transforms

        elif hasattr(transforms, '__iter__'):
            return A.Compose(transforms)

        if transforms is None:
            if return_None:
                return None
            else:
                return lambda **kwargs: kwargs

        else:
            raise ValueError('Ожидается, что параметр "transforms" будет функтером, ' + \
                             f'или итерируемым объектом. Получен {type(transforms)}!')

        return image_mask_func

    # Собираем функцию, выполняющую преобразование, входами и выходами которой
    # являются image и mask:
    def image_mask_func(self, image, mask):
        image, mask = self.transform(image=image, mask=mask).values()
        return image, mask
    # Полезно при использовании в mpmap из utils, рассчитанной на позиционные
    # аргументы.

    # Вызов экземпляра класса как функции применяет переданные преобразования:
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)