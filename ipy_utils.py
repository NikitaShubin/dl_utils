import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from IPython import get_ipython
from cv_utils import Mask
from cvat import CVATPoints
from IPython.display import Video, clear_output, display
import ipywidgets
import os
from urllib.parse import quote
from IPython.display import HTML


def path2link(path, link_file_name=None):
    '''
    Генерирует HTML-ссылку для скачивания файла в JupyterLab.

    Параметры:
    path (str): Абсолютный или относительный путь к файлу.
    link_file_name (str, опционально): Текст ссылки. Если не указан,
    используется имя файла.

    Возвращает:
    IPython.display.HTML: Объект, отображающий ссылку в Jupyter.

    Примеры:
    >>> path2link("data/file.txt")
    >>> path2link("data/file.txt", "Скачать файл")
    '''
    # Проверка существования файла:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Файл не найден: {path}')

    # Получение абсолютного пути и нормализация разделителей:
    abs_path = os.path.abspath(path).replace('\\', '/')

    # Кодирование пути для URL:
    encoded_path = quote(abs_path)

    # Формирование URL для JupyterLab:
    href = f'/files/{encoded_path}'

    # Текст ссылки (имя файла, если не задано):
    link_text = link_file_name or os.path.basename(path)

    # Генерация HTML-кода ссылки с атрибутом download:
    html = (f'<a href="{href}" '
            f'download="{os.path.basename(path)}">'
            f'{link_text}</a>')

    return HTML(html)


class IPYInteractiveSegmentation:
    '''
    Интерактивная сегментация в Jupyter-ноутбуке.
    '''
    msk_name = 'mask'
    box_name = 'box'
    pos_name = 'include'
    neg_name = 'exclude'
    clr_name = 'clear'

    # Очищает все подсказки, если нажат класс "clear":
    def on_class_changed(self, new_class):
        if new_class == self.clr_name:
            self.klicker._current_class = self.pos_name
            self.clear()
            self.klicker._update_points()


    def show(self, img):
        self.img = img

        # Включаем интерактивный matplotlib.
        self.ipython.run_line_magic('matplotlib', 'widget')

        # Выводим изображение в ячейку ноутбука:
        self.fig, self.ax = plt.subplots(figsize=(10, 4),
                                         constrained_layout=True)
        self.fig.canvas.header_visible = False    # Убираем название фигуры
        self.fig.canvas.footer_visible = False    # Убираем отступ снизу
        self.fig.canvas.toolbar_position = 'top'  # Панель сверху
        self.ax.imshow(self.img)
        plt.title('Интерактивная сегментация')
        self.ax.axis(False)

        # Зум колёсиком мыши:
        zoom_factory(self.ax)
        panhandler(self.fig, button=2)

        # Инициируем счётчик кликов:
        self.klicker = clicker(self.ax,
                               [self.msk_name,
                                self.box_name,
                                self.pos_name,
                                self.neg_name,
                                self.clr_name],
                               markers=['p', 's', 'o', 'x', 'X'],
                               colors=['k', 'b', 'g', 'r', 'w'])

        # Если было указано начальное состояние точек, то применяем его
        # и очищаем внутреннее состояние чтобы в следующий раз не применять
        # его при повторном вызове self.show:
        if self.init_msk_points + self.init_box_points + \
                self.init_pos_points + self.init_neg_points:
            self.klicker.set_positions({self.msk_name: self.init_msk_points,
                                        self.box_name: self.init_box_points,
                                        self.pos_name: self.init_pos_points,
                                        self.neg_name: self.init_neg_points,
                                        self.clr_name: []})
            self.klicker._update_points()  # Отрисовка точек на экране

            # Очистка:
            self.init_msk_points = []
            self.init_box_points = []
            self.init_pos_points = []
            self.init_neg_points = []

        # Добавляем события, если указана функция интерактивной сегментации:
        self.klicker.on_point_added(self.on_click)
        self.klicker.on_point_removed(self.on_click)
        self.klicker.on_class_changed(self.on_class_changed)
        self.klicker._current_class = self.pos_name
        self.on_click()  # Вызываем отрисовку

    def __init__(self, img=None, points2masks=None, init_msk_points=[],
                 init_box_points=[], init_pos_points=[], init_neg_points=[]):
        self.img = img
        self.points2masks = points2masks

        # Привязываемся к текущему ноутбуку:
        self.ipython = get_ipython()

        # Исходный набор точек:
        self.init_msk_points = init_msk_points
        self.init_box_points = init_box_points
        self.init_pos_points = init_pos_points
        self.init_neg_points = init_neg_points

        # Сразу отрисовываем изображение, если передано:
        if img is not None:
            self.show(img)

    # Переводим список точек в numpy-массив:
    @staticmethod
    def list2np(list_pos):
        if len(list_pos):
            return np.array(list_pos, np.float32)
        else:
            return np.zeros((0, 2), np.float32)

    # Обработка изменения набора точек на изображении (событие):
    def on_click(self, *args, **kwargs):
        # Берём текущее состояние подсказок:
        clicks = self.klicker.get_positions()

        # Переводим подсказки в numpy-массив:
        msk_points = self.list2np(clicks[self.msk_name])
        box_points = self.list2np(clicks[self.box_name])
        pos_points = self.list2np(clicks[self.pos_name])
        neg_points = self.list2np(clicks[self.neg_name])

        # Берём исходное изображение:
        img = self.img.copy()

        # Если функция построения масок задана:
        if self.points2masks:

            # Строим маску по подсказкам, если модель задана:
            masks = self.points2masks(msk_points, box_points,
                                      pos_points, neg_points)

            # Отображаем маску на исходном изображении, если она есть:
            if masks:
                img = self.img.copy()
                for mask in masks:
                    img = Mask(mask > 0).draw(img,
                                              color=(255, 0, 0),
                                              alpha=0.3)

        # Рисуем прямоугольник, если есть точки:
        if box_points.shape[0]:

            # Переводим точки в обрамляющий прямоугольник:
            box = CVATPoints(box_points, 'polygon').astype('rectangle')

            # Есил прямоугольник имеет ненулевую площадь, то отрисовываем:
            if box.area():
                img = box.draw(img, color=(0, 0, 255), alpha=0.5)

        self.ax.imshow(img)
        self.klicker._update_points()

    def clear(self):
        self.klicker.set_positions({self.msk_name: [],
                                    self.box_name: [],
                                    self.pos_name: [],
                                    self.neg_name: [],
                                    self.clr_name: []})
        self.klicker._update_points()
        self.on_click()


class IPYRadioButtons:
    '''
    Выводит радиокнопки выбора вариантов из списка и вызывает колбек setter в
    момент изменения значений.
    '''

    def __init__(self, options, description=' ', setter=None):
        self.setter = setter
        self.radio_buttons = ipywidgets.RadioButtons(options=options,
                                                     description=description)

    def show(self):
        if self.setter:

            def setter(criteria):
                return self.setter(criteria)

            return ipywidgets.interact(setter, criteria=self.radio_buttons)
        else:
            display(self.radio_buttons)


# Очиcтка ячейки вывода:
def cls():
    clear_output(wait=True)


class IPYButton:
    '''
    Обычная кнопка.
    '''

    # Декоратор предварительной очистки ячейки вывода:
    def preprocess(self, func):
        def preprocessed(*args, **kwargs):
            with self.output:
                if self.use_cls:
                    cls()
                return func(*args, **kwargs)
        return preprocessed

    def __init__(self, description=' ', action=None, use_cls=True):
        self.button = ipywidgets.Button(description=description)
        self.output = ipywidgets.Output()
        self.use_cls = use_cls

        # Добавляем к действию предварительную очистку вывода, если надо:
        if action:
            self.button.on_click(self.preprocess(action))

    def show(self):
        display(self.button, self.output)


def show_video(file, clear=True, size=None):
    '''
    Отображает, хранящееся локально на сервере, видео в ячейке Jupyter-ноутбука.
    '''
    # Обрабатываем параметр size:
    if size is None:
        height = width = None
    elif isinstance(size, (list, tuple)) and len(size) == 2 or \
            isinstance(size, np.ndarray) and size.size == 2:
        height, width = size
    elif isinstance(size, int):
        height, width = size, None
    elif isinstance(size, np.ndarray) and size.size == 1:
        height, width = int(size), None
    else:
        raise ValueError(f'Неподдерживаемый формат size = {size}!')

    # Очищаем предыдущий вывод в ячейке, если надо:
    if clear:
        cls()

    return Video(file, embed=False, height=height, width=width)


def concat(*cell_outs, sep='border-top: 1px solid #ccc; margin: 8px 0;'):
    '''
    Объединяет несколько элементов вывода, разделияя их линиями.

    Параметры:
    *cell_outs: элементы для отображения (HTML, текст, др.)
    sep: CSS-стиль разделительной линии (None для отключения)
    '''
    html_content = '<div style="display:flex;flex-direction:column">'

    for i, item in enumerate(cell_outs):

        # Преобразуем элемент в HTML-строку:
        if hasattr(item, '_repr_html_'):
            item_html = item._repr_html_()
        else:
            item_html = f'<div>{str(item)}</div>'

        # Добавляем элемент:
        html_content += f'<div>{item_html}</div>'

        # Добавляем разделитель после всех элементов кроме последнего:
        if i < len(cell_outs) - 1 and sep:
            html_content += f'<div style="{sep}"></div>'

    html_content += '</div>'
    return HTML(html_content)