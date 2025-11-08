from matplotlib import pyplot as plt
from groundingdino.util.inference import load_model, predict, T, Image
import groundingdino
import cv2
import os
import warnings
from urllib.request import urlretrieve

from pt_utils import AutoDevice
from cvat import CVATPoints, concat_dfs
from utils import color_float_hsv_to_uint8_rgb, mkdirs, AnnotateIt


class GDino:
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

        # Переводим все классы в нижний регистр:
        prompt2label = {key.lower(): val for key, val in prompt2label.items()}
        # Grounding DINO возвращает None для всех имён классов, в чъих
        # названиях присутствуют символы верхнего регистра.
        # (см. https://github.com/IDEA-Research/GroundingDINO/issues
        #  /70#issuecomment-1642950663)

        for key, val in prompt2label.items():
            if key == 'none' or val == 'None':
                raise ValueError('Использование метки "none" запрещено!')
        # Использовать метку None в любом регистре нельзя, т.к. она
        # используется самим GroundingDINO.

        return prompt2label

    # Фиксирует новый словарь запросов -> меток:
    def set_prompt2label(self, prompt2label=None):
        self.prompt2label = self._parse_prompt2label(prompt2label)

    # Переводит словарь запросов в запросы для инференса GroundingDINO:
    @staticmethod
    def _build_caption(prompt2label):
        return '.'.join(prompt2label.keys())

    def __init__(self                                                ,
                 model_path     = './groundingdino_swinb_cogcoor.pth',
                 box_threshold  = 0.35                               ,
                 text_threshold = 0.25                               ,
                 device         = 'auto'                             ,
                 prompt2label   = {}                                 ):

        # Если в пути не указано имя ни одной модели, то считаем это
        # именем папки, а моделью выберем groundingdino_swinb_cogcoor.pth:
        model_basename = os.path.basename(model_path).lower()
        if model_basename not in {'groundingdino_swint_ogc.pth',
                                  'groundingdino_swinb_cogcoor.pth'}:
            model_basename = 'groundingdino_swinb_cogcoor.pth'
            model_path = os.path.join(model_path, model_basename)

        # Качаем модель, если её не оказалось в указанном месте:
        if not os.path.isfile(model_path):

            # Определяем имя папки для файла-модели:
            model_dir = os.path.abspath(os.path.dirname(model_path))

            # Создаём папку, если её не было:
            mkdirs(model_dir)

            # Путь до модели в вебе:
            if model_basename == 'groundingdino_swinb_cogcoor.pth':
                url = 'https://github.com/IDEA-Research/GroundingDINO/' + \
                      'releases/download/v0.1.0-alpha2/' + \
                      'groundingdino_swinb_cogcoor.pth'
            else:
                url = 'https://github.com/IDEA-Research/GroundingDINO/' + \
                      'releases/download/v0.1.0-alpha/' + \
                      'groundingdino_swint_ogc.pth'

            # Загрузка:
            prefix = f'Загрузка модели {url} в "{model_dir}" '
            with AnnotateIt(prefix + '...', prefix + 'завершена!'):
                urlretrieve(url, model_path)

        # Определяем имя конфигурационного файла, соответствующую
        # заданной модели:
        if model_basename == 'groundingdino_swinb_cogcoor.pth':
            config_name = 'GroundingDINO_SwinB_cfg.py'
        else:
            config_name = 'GroundingDINO_SwinT_OGC.py'

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
        img, boxes, logits, labels = self._predict(
            img, prompt2label, box_threshold, text_threshold
        )

        # Определяем размер исходного изображения:
        imsize = img.shape[:2]

        dfs = []
        for box, logit, label in zip(boxes, logits, labels):
            points = CVATPoints.from_yolobbox(*box, imsize)
            df_row = points.to_dfrow(source='GroundingDINO', **kwargs)
            df_row['label'] = label
            dfs.append(df_row)
            # Метка присвоена вне to_dfrow, чтобы не переводить None в str.
            # Иначе потом task_auto_annottation её не заменит.

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