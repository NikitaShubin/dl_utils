from matplotlib import pyplot as plt
from groundingdino.util.inference import load_model, predict, T, Image
import groundingdino
import cv2
import os
import warnings

from pt_utils import AutoDevice
from cvat import CVATPoints, concat_dfs
from utils import color_float_hsv_to_uint8_rgb


class Dino:
    '''
    Класс предварительной детекции с помощью Grounding DINO.
    '''
    # Преобразует любой формат запросов в словарь:
    @staticmethod
    def _parse_prompt2label(prompt2label={}):

        # Если это уже словарь - оставляем без изменений:
        if isinstance(prompt2label, dict):
            pass

        # Если передана пустота - делаем пустой словарь:
        elif prompt2label in (None, [], (), set()):
            prompt2label = {}

        # Если дан список, кортеж или множество - принудительно создаём
        # из него словарь:
        elif isinstance(prompt2label, (list, tuple, set)):
            prompt2label = dict(zip(prompt2label, prompt2label))

        # Если дана строка - используем её как единственный запрос:
        elif isinstance(prompt2label, str):
            prompt2label = {prompt2label: prompt2label}

        else:
            raise ValueError('Неподдерживаемый тип данных запросов!')

        # Проверка словаря запросов на корректность:
        for prompt in prompt2label.keys():
            if '.' in prompt:
                raise ValueError('Точка используется как ' +
                                 'разделитель запросов и ' +
                                 'не должна быть внутри одного класса!')

        return prompt2label

    # Фиксирует новый словарь запросов -> меток:
    def set_prompt2label(self, prompt2label=None):
        self.prompt2label = self._parse_prompt2label(prompt2label)

    # Переводит словарь запросов в запросы для инференса GroundingDINO:
    @staticmethod
    def _build_caption(prompt2label):
        return '.'.join(prompt2label.keys())

    def __init__(self                                                 ,
                 model_path     = '../groundingdino_swinb_cogcoor.pth',
                 box_threshold  = 0.35                                ,
                 text_threshold = 0.25                                ,
                 device         = 'auto'                              ,
                 prompt2label   = {}                                  ):

        # Определяем имя конфигурационного файла, соответствующую
        # заданной модели:
        model_name = os.path.basename(model_path).lower()
        if 'swinb' in model_name and 'swint' not in model_name:
            config_name = 'GroundingDINO_SwinB_cfg.py'
        elif 'swinb' not in model_name and 'swint' in model_name:
            config_name = 'GroundingDINO_SwinT_OGC.py'
        else:
            raise ValueError('Невозможно определить конфигурационный файл ' +
                             'по имени модели!')

        # Собираем полный путь до конфигурационного файла:
        dino_dir = os.path.dirname(os.path.abspath(groundingdino.__file__))
        config_path = os.path.join(dino_dir, 'config', config_name)

        # Загружаем саму модель:
        self.model = load_model(config_path, model_path)

        # Определяем, доступно ли GPU:
        if device == 'auto':
            device = AutoDevice().device
        self.device = device

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Фиксируем словарь запросов -> меток:
        self.set_prompt2label(prompt2label)

        # Предобработчик изображений:
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ]
        )

    # Читает изображение из файла, если надо:
    @staticmethod
    def imread(img):
        if isinstance(img, str):
            return cv2.imread(img, cv2.IMREAD_COLOR)[..., ::-1]
        else:
            return img

    # Загружает и предобрабатывает изображение:
    def _img_preproc(self, img):

        # Читаем изображение из файла, если надо:
        img = self.imread(img)

        # Выполняем предобработку:
        img, _ = self.transform(Image.fromarray(img), None)

        return img

    # Возвращает само изображение и результаты детекции:
    def _predict(self,
                 img,
                 prompt2label=None,
                 box_threshold=None,
                 text_threshold=None):

        # Читаем изображение из файла, если надо:
        img = self.imread(img)

        # Доопределяем словарь запросов -> меток:
        prompt2label = \
            self._parse_prompt2label(prompt2label) or self.prompt2label

        # Инференс:
        with warnings.catch_warnings(action="ignore"):
            boxes, logits, phrases = predict(
                model=self.model,
                image=self._img_preproc(img),
                caption=self._build_caption(prompt2label),
                box_threshold=box_threshold or self.box_threshold,
                text_threshold=text_threshold or self.text_threshold,
                device=self.device
            )

        # Переводим все запросы в соответствующие метки:
        labels = list(map(prompt2label.get, phrases))

        return img, boxes.numpy(), logits.numpy(), labels

    # Формирование разметки в cvat-формате:
    def img2df(self,
               img,
               prompt2label=None,
               box_threshold=None,
               text_threshold=None,
               df_as_list_of_dfs=False,
               source='GroundingDINO',
               **kwargs):
        # Получаем кадр и результаты детекции:
        img, boxes, logits, labels = self._predict(img,
                                                    prompt2label,
                                                    box_threshold,
                                                    text_threshold)

        # Определяем размер исходного изображения:
        imsize = img.shape[:2]

        dfs = []
        for box, logit, label in zip(boxes, logits, labels):
            points = CVATPoints.from_yolobbox(*box, imsize)
            dfs.append(points.to_dfrow(
                source='GroundingDINO', label=label,  **kwargs
            ))

        if df_as_list_of_dfs:
            return dfs
        else:
            return concat_dfs(dfs)

    # Возвращает изображение с результатами детекции:
    def draw(self,
             img=None,
             prompt2label=None,
             box_threshold=None,
             text_threshold=None,
             color='auto',
             **kwargs):

        # Получаем кадр и результаты детекции:
        img, boxes, logits, labels = self._predict(img,
                                                    prompt2label,
                                                    box_threshold,
                                                    text_threshold)

        # Определяем размер исходного изображения:
        imsize = img.shape[:2]

        # Перечень запросов:
        prompts = self._parse_prompt2label(prompt2label).keys()

        # Формируем словарь класс -> цвет:
        if isinstance(color, dict):
            label2color = color
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            label2color = {label: color for label in prompts}
        elif color == 'auto':
            label2color = {
                label: color_float_hsv_to_uint8_rgb(ind / len(prompts))
                for ind, label in enumerate(prompts)
            }
        else:
            raise ValueError('Неподдерживаемое значение "color": ' +
                             f'{color}!')

        # Наносим обрамляющие прямоугольники с текстом на исходное
        # изображение:
        for box, logit, phrase in zip(boxes, logits, labels):
            points = CVATPoints.from_yolobbox(*box, imsize)
            img = points.draw(img,
                              caption=f'{phrase}: conf={logit:.3}',
                              color=label2color.get(phrase, (0, 0, 0)),
                              **kwargs)

        return img

    # Выводит изображение с результатами детекции на экран:
    def show(self, *args, **kwargs):
        plt.imshow(self.draw(*args, **kwargs))
        plt.axis(False)

    __call__ = img2df