"""copybal.py
********************************************
*      Копирующая балансировка датасета.   *
*                                          *
* Балансировать YOLO-датасет "влоб"        *
* затруднительно, т.к. обычно изображение  *
* содержит несколько объектов разного      *
* класса, а разрезать изображение на       *
* отдельные объекты не целесообразно ввиду *
* возможной потери полезного контекста     *
* наблюдения этого объекта. Поэтому        *
* балансировать датасет приходится путём   *
* увеличения числа целых изображений       *
* Применение более сложной аугментации,    *
* чем n поворотов на 90 градусов,          *
* отражение по вертикали или горизонтали и *
* масштабирование приводит либо к потере   *
* части объектов, либо к образованию       *
* ненужных полей у изображений. Поэтому    *
* семплы дублируются без изменений. Это    *
* позволяет использовать мягкие ссылки на  *
* файлы, практически не увеличивая размер  *
* занимаемого места на диске даже при      *
* кратном увеличении числа семплов.        *
*                                          *
* Сама балансировка представляет собой     *
* итеративное обновление счётчика копий    *
* файлов с целью минимизации значения      *
* сложной функции потерь. Процесс          *
* опримизации производится с помощью,      *
* PyTorch, а функция потерь нацелена на    *
* учёт следующих противоречивых            *
* требований:                              *
*   1) Число объектов, представляющих      *
*   разные суперклассы должно стремиться к *
*   равенству.                             *
*   2) Число объектов, представляющих      *
*   разные классы в пределах одного        *
*   суперкласса должно стремиться к        *
*   равенству.                             *
*   3) Число появлений одного объекта на   *
*   нескольких семплах должно стремиться к *
*   равенству для всех объектов. Т.е.      *
*   должен учитываться тот факт, что       *
*   некоторые объекты появляются на многих *
*   изображениях, т.к. сами изображения    *
*   взяты из видеопоследовательности.      *
*   4) Балансировка должна минимизировать  *
*   отличия в числе копий для каждого      *
*   файла.                                 *
* Из этих четырёх составляющих и           *
* собирается общая функция потерь.         *
*                                          *
*                                          *
* Основные функции:                        *
*   init_task_object_file_graphs -         *
*       инициирует граф связностей.        *
*                                          *
*   update_object_file_graphs - вносит     *
*       значение target_file_basename во   *
*       все нужные списки графа            *
*       связностей.                        *
*                                          *
*   drop_unused_track_ids_in_graphs -      *
*       исключает объекты, не имеющие ни   *
*       одного целевого файла, из всех     *
*       графов связностей.                 *
*                                          *
*   make_copy_bal - выполняет саму         *
*       балансировку.                      *
*                                          *
********************************************
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pt_utils import AutoDevice, safe_var
from utils import mpmap

# Знак, разделяющий строку идентификации объектов:
sep_char = '\t'


def build_unigue_track_id(file, task_id, subtask_id, track_id, label):
    """Собирает строку идентификации объектов.
    """
    # Если имеется целый ряд файлов, фиксируем папку первого из них:
    if isinstance(file, (list, tuple)):
        file = os.path.join(os.path.dirname(file[0]), '*')

    # Разделяющий знак не должен встречаться в составляющих:
    assert sep_char not in file
    assert sep_char not in label

    return sep_char.join([file, str(task_id), str(subtask_id), str(track_id), label])


def split_unigue_track_id(track_id):
    """Разбивает строку идентификации объектов на составляющие.
    Операция, частично обратная к build_unigue_track_id.
    """
    # Само разбиение:
    file, task_id, subtask_id, track_id, label = track_id.split(sep_char)
    task_id = int(task_id)
    subtask_id = int(subtask_id)
    track_id = int(track_id)

    return file, task_id, subtask_id, track_id, label


def init_object_file_graph_by_task(task, task_id, labels_convertor):
    """Инициирует граф связностей объектов и классов внутри задачи.
    Незаполненными остаются только списки файлов ("file_list")
    для каждого объекта ("track_id").
    """
    # Инициируем список датафреймов:
    rows = [
        pd.DataFrame(
            columns=['track_id', 'file_list', 'class_meaning', 'supeerclass_meaning']
        )
    ]

    # Перебираем все подзадачи
    for subtask_id, (df, file, true_frames) in enumerate(task):
        # Перебираем все уникальные сочетания номера трека и класса в датафрейме:
        for track_id, label in df[['track_id', 'label']].drop_duplicates().values:
            # В рассмотрение берём только используемые классы:
            if labels_convertor(label) >= 0:
                track_id = build_unigue_track_id(
                    file, task_id, subtask_id, track_id, label
                )  # Определяем идентификатор трека
                file_list = []  # Инициируем список файлов
                class_meaning = labels_convertor.any_label2meaning(
                    label
                )  # Определяем расшифровку класса ...
                supeerclass_meaning = labels_convertor.class_meaning2superclass_meaning[
                    class_meaning.lower()
                ]  # ... и суперкласса.

                # Создаём строку с полученными параметрами:
                row = pd.Series(
                    {
                        'track_id': track_id,
                        'file_list': file_list,
                        'class_meaning': class_meaning,
                        'supeerclass_meaning': supeerclass_meaning,
                    }
                )

                # Вносим созданную строку в список:
                rows.append(pd.DataFrame(row).T)

    # Собираем датафрейм-граф из полученных строк:
    task_object_file_graph = pd.concat(rows).set_index('track_id')

    return task_object_file_graph


def init_task_object_file_graphs(
    tasks, labels_convertor, desc='Инициализация графа связностей'
):
    """Инициирует граф связностей объектов и классов для всех задач.
    Незаполненными остаются только списки файлов ("file_list")
    для каждого объекта ("track_id").
    """
    # Выполняем параллельную обработку каждой задачи:
    task_object_file_graphs = mpmap(
        init_object_file_graph_by_task,
        tasks,
        range(len(tasks)),
        [labels_convertor] * len(tasks),
        desc=desc,
    )
    """
    # Собираем все графы в один:
    tasks_object_file_graph = pd.concat(task_object_file_graphs)
    
    # В собранном графе не должно быть повторений идентификаторов треков:
    assert len(set(tasks_object_file_graph.index)) == len(tasks_object_file_graph)
    """
    return task_object_file_graphs


def update_object_file_graphs(
    df,
    object_file_graphs,
    labels_convertor,
    source_file,
    task_id,
    subtask_id,
    target_file_basename,
):
    """Вносит значение target_file_basename во все нужные списки графа связностей object_file_graphs.
    """
    # Делаем копию, чтобы не записывать в исходный датафрейм:
    object_file_graphs = object_file_graphs.copy()

    # Оставляем в датафрейме лишь строки, где контуры не скрыты:
    df = df[df['outside'] == False]

    # Перебираем все уникальные сочетания номера трека и класса в датафрейме:
    for track_id, label in df[['track_id', 'label']].drop_duplicates().values:
        # В рассмотрение берём только используемые классы:
        if labels_convertor(label) >= 0:
            track_id = build_unigue_track_id(
                source_file, task_id, subtask_id, track_id, label
            )  # Определяем идентификатор трека

            # Внесение целевого файла в список для установления связей трек->файл.
            object_file_graphs.loc[track_id, 'file_list'].append(target_file_basename)

    return object_file_graphs


def drop_unused_track_ids_in_graphs(object_file_graphs):
    """Исключает объекты, не имеющие ни одного целевого файла, из всех графов связностей.
    """
    return [df[df['file_list'].apply(len) > 0] for df in object_file_graphs]


def torch_copy_bal(
    files: 'Список имён файлов дублируемых изображений',
    file2index: 'Переход от имени изображения к его индексу',
    classes_collector: 'Словарь расшифровка_классов -> список_списков_индексов_файлов',
    superclass_meaning2class_meaning: 'Словарь расшифровка_классов -> множество_расшифровок_классов',
    steps: 'Максимальное число итераций оптимизации' = 10000,
    max_file_copy_num: 'Максимальное число копий одного файла' = None,
    max_ds_increase_frac: 'Максимальный коэффициент увеличения числа семплов в датасете' = None,
    max_copy_per_step: 'Максимальное число копий, добавляемое за одну итерацию' = 50000,
    lr: 'Скорость "обучения"' = 1e-4,
    device: 'Устройство для вычислений' = 'auto',
    history_file: 'CSV-файл, в который регулярно будет сохраняться история оптимизации' = None,
):
    """Выполняет рассчёт числа копий каждого изображения для увеличения баланса в датасете.
    """
    # Инициируем датафрейм, хранящий историю оптимизации:
    hist = pd.DataFrame(
        columns=[
            'step',
            'loss',
            'files_counter_loss',
            'object_loss',
            'class_loss',
            'superclass_loss',
            'grad_argmin',
            'min_grad',
            'max_copy_num',
            'num_bal_files_frac',
            'ds_frac',
        ]
    ).set_index('step')

    # Автоматически определяем устройство, если надо:
    if device == 'auto':
        device = AutoDevice()()

    # Инициируем единицами счётчик каждого файла в датасете:
    files_counter = torch.autograd.Variable(
        torch.ones(1, len(files), dtype=torch.float64), requires_grad=True
    ).to(device)

    # Следующий цикл будет выполняться под трайем для перехвата прерывания с клавиатуры:
    try:
        # Инициируем предыдущее значение ф-ии потерь для сравнения:
        prev_loss = np.inf

        # Строим вычислительный граф связностей для балансировки через оптимизацию:
        for step in tqdm(range(steps), desc='Балансировка классов'):
            # Инициируем 3 составляющих функции потерь:
            object_loss = []  # Объектная      дисперсия (внутри каждого объекта    )
            class_loss = []  # Классовая      дисперсия (внутри каждого класса     )
            superclass_loss = []  # Суперклассовая дисперсия (внутри каждого суперкласса)

            classes = {}  # Группировка объектов в классы
            superclasses = {}  # Группировка классов  в суперклассы

            # Перебираем классы и соответствующие им списки списков индексов:
            for class_meaning, file_lists in classes_collector.items():
                # Инициируем список объектов:
                objects = []

                # Перебор всех объектов текущего класса, представленных в виде списков индексов файлов:
                for file_list in file_lists:
                    # Список всех счётчиков файлов для текущего объекта:
                    cell_list = [
                        files_counter[:, file_index : file_index + 1]
                        for file_index in file_list
                    ]

                    # Число файлов, на которых появляется текущий объект:
                    object_appearance = torch.concat(cell_list, -1).sum(
                        -1, keepdim=True
                    )

                    # Список счётчиков всех объектов текущего класса:
                    objects.append(object_appearance)

                # Если в списке более одного элемента:
                if len(objects) > 1:
                    # Объединяем список счётчиков в вектор:
                    objects = torch.concat(objects, -1)

                    # Подсчитываем дисперсию появления объектов в пределах одного класса:
                    object_var = safe_var(objects, dim=-1, keepdim=True)

                    # Вносим эту дисперсию в список элементов функции потерь:
                    object_loss.append(object_var)

                    # Объединяем все счётчики, получая общее число появлений объекта во всём датасете:
                    objects = objects.sum(-1, keepdim=True)

                # Если в списке всего один елемент, то берём его без изменений:
                else:
                    objects = objects[0]

                # Внутри каждого класса свой счётчик появления объектов:
                classes[class_meaning] = objects

            # Усредняем внутриклассовые дисперсии по всем классам:
            object_loss = torch.concat(object_loss).mean()

            # Теперь перебираем все суперклассы:
            for (
                superclass_meaning,
                class_meanings,
            ) in superclass_meaning2class_meaning.items():
                # Получаем список счётчиков объектов, принадлежащих текущему суперклассу
                superclass = list(map(classes.get, class_meanings))

                # Если в списке более одного счётчика:
                if len(superclass) > 1:
                    # Объединяем счётчики в вектор:
                    superclass = torch.concat(superclass, -1)

                    # Получаем дисперсию и вносим её в список элементов функции потерь:
                    class_var = safe_var(superclass, dim=-1, keepdim=True)
                    class_loss.append(class_var)

                    # Объединяем все счётчики, получая общее число появлений суперкласса во всём датасете:
                    superclass = superclass.sum(-1, keepdim=True)

                # Если в списке всего один счётчик, то берём его без изменений:
                else:
                    superclass = superclass[0]

                # Внутри каждого суперкласса свой счётчик появления объектов:
                superclasses[superclass_meaning] = superclass

            # Усредняем межклассовые дисперсии внутри каждого суперкласса:
            class_loss = torch.concat(class_loss).mean()

            # Межсуперклассовая дисперсия:
            superclass_loss = safe_var(torch.concat(list(superclasses.values())))

            # Дисперсия счётчиков дублей (для предотвращения слишком большой неровномерности дублирования):
            files_counter_loss = safe_var(files_counter)

            # Общая функция потерь:
            loss = files_counter_loss + object_loss + class_loss + superclass_loss

            # Вне рассчёта градиентов:
            with torch.no_grad():
                # Получаем градиенты
                grads = torch.autograd.grad(loss, files_counter)[0]

                # Убираем все положительные градиенты (мы не можем уменьшить число копий файлов):
                grads[grads > 0] = 0

                # Отсекаем все первышения значений счётчиков по порогу:
                if max_file_copy_num is not None:
                    files_counter[files_counter > max_file_copy_num] = max_file_copy_num

                # Получаем вектор прирощения счётчика:
                files_counter_diff = -lr * grads  # Размер прирощения
                if (
                    files_counter_diff.sum() > max_copy_per_step
                ):  # Занижаем общее число новых копий, если оно превысело порог
                    files_counter_diff = (
                        max_copy_per_step
                        * files_counter_diff
                        / files_counter_diff.sum()
                    )
                files_counter_diff = torch.round(
                    files_counter_diff
                )  # Округляем до целых

                # Фиксируем различные показатели для статистики:
                loss = float(loss.cpu().numpy())
                files_counter_loss = float(files_counter_loss.cpu().numpy())
                object_loss = float(object_loss.cpu().numpy())
                class_loss = float(class_loss.cpu().numpy())
                superclass_loss = float(superclass_loss.cpu().numpy())
                grad_argmin = int(grads[0].argmin().cpu().numpy())
                min_grad = int(-grads.min().cpu().numpy())
                max_copy_num = int(files_counter.max().cpu().numpy())
                lr = float(lr)
                num_files = int((grads < 0).sum().cpu().numpy())
                num_bal_files_frac = float(
                    (files_counter > 1).sum() / files_counter.shape[-1]
                )
                ds_frac = float(files_counter.sum() / files_counter.shape[-1])
                """
                print(              'loss = {:>14.3f}'.format(              loss), end='; ')
                print('files_counter_loss = {:>7.3f}' .format(files_counter_loss), end='; ')
                print(       'object_loss = {:>11.3f}'.format(       object_loss), end='; ')
                print(        'class_loss = {:>14.3f}'.format(        class_loss), end='; ')
                print(   'superclass_loss = {:>14.3f}'.format(   superclass_loss), end='; ')
                print(       'grad_argmin = {:>6d}'   .format(       grad_argmin), end='; ')
                print(          'min_grad = {:>6d}'   .format(          min_grad), end='; ')
                print(      'max_copy_num = {:>4d}'   .format(      max_copy_num), end='; ')
                print('num_bal_files_frac = {:>0.6f}' .format(num_bal_files_frac), end='; ')
                print(           'ds_frac = {:>0.3f}' .format(           ds_frac), end='\n')
                """
                hist.loc[step, :] = (
                    loss,
                    files_counter_loss,
                    object_loss,
                    class_loss,
                    superclass_loss,
                    grad_argmin,
                    min_grad,
                    max_copy_num,
                    num_bal_files_frac,
                    ds_frac,
                )

                files_counter_old = (
                    files_counter.clone()
                )  # Запоминаем прежнее значение счётчиков
                files_counter += (
                    files_counter_diff  # Применяем   прирощение   к  счётчикам
                )

                # Отсечение всех счётчиков по маскимально допустимому значению, если оно задано:
                if max_file_copy_num is not None:
                    files_counter[files_counter > max_file_copy_num] = max_file_copy_num

                # Сохраняем историю оптимизации, если задан файл для сохранения:
                if history_file is not None:
                    hist.to_csv(history_file)

                # Выходим из цикла итераций если изменение отсутствует:
                if (files_counter == files_counter_old).all():
                    break

                # Выходим из цикла итераций если копия создана уже для кждого файла:
                if (files_counter > 1).all():
                    break

                # Выходим из цикла итераций если коэффициент увеличения ...
                # ... числа семплов в датасете уже превысел заданный предел:
                if (
                    max_ds_increase_frac is not None
                    and files_counter.sum()
                    > files_counter.shape[-1] * max_ds_increase_frac
                ):
                    break

                # Если ф-ия потерь выросла, а не упала, ...
                # ... то сбрасываем все счётчики и ...
                # ... уменьшаем скорость обучения.
                if loss > prev_loss:
                    files_counter[:] = 1.0
                    lr /= 2

                # Сохраняем текущее значение ф-ии потерь для следующей итерации:
                prev_loss = loss

    # Перехватываем прерывание с клавиатуры, останавливая итеративный процесс рассчёта:
    except KeyboardInterrupt:
        print('Итеративный рассчёт числа копий семплов прерван вручную!')

    # Собираем словарь перехода имя_файла -> счётчик_копий:
    files_counter = files_counter.detach().cpu().numpy().astype(int).flatten()
    files2count = dict(zip(files, files_counter))

    return files2count, hist


def make_copy_bal(
    object_file_graphs: 'Список графов связностей',
    img_dir: 'Путь до папки с изображениями',
    steps: 'Максимальное число итераций' = 10000,
    max_file_copy_num: 'Максимальное число копий одного файла, после которого этот файл не копируется' = 500,
    max_ds_increase_frac: 'Во сколько раз можно увеличить размер датасета прежде, чем балансировка прервётся' = 4,
    device: 'Устройство для вычислений' = 'auto',
):
    """Выполняет копирующую балансировку YOLO-датасета на основе списка графов связностей.
    """
    ds_path = os.path.dirname(os.path.dirname(img_dir))  # Путь ко всему датасету
    lbl_dir = os.path.join(
        ds_path, 'labels', os.path.basename(img_dir)
    )  # Путь к разметке
    sts_dir = os.path.join(ds_path, 'statistics')  # Путь к статистике

    # Формируем множество всех изображений:
    files = set(os.listdir(img_dir))

    # Словарь перехода имя_файла -> номер_файла_в_списке:
    file2index = {file: ind for ind, file in enumerate(files)}

    # Объединяем все графы связности в один:
    object_file_graph = pd.concat(object_file_graphs)

    # Проверяем, нет ли у разных графов связностей общих объектов:
    assert len(set(object_file_graph.index)) == len(object_file_graph)

    # Объекты перехода от имени файла к индексу (file2index) и от индекса к имени (files) созданы.

    # Парсинг графов связностей:
    classes_collector = {}  # Инициируем словарь переха расшифровка_классов -> Список списков индексов
    superclass_meaning2class_meaning = {}  # Инициируем словарь переха расшифровка_суперклассов -> Список расшифровок классов
    for file_list, class_meaning, supeerclass_meaning in tqdm(
        object_file_graph.values, desc='Парсинг графов связностей'
    ):
        # Переводим все файлы из списка в соответствующие индексы:
        file_list = list(map(file2index.get, file_list))

        # Добавляем очередной список индексов файлов в список списков для текущей расшифровки классов:
        classes_collector[class_meaning] = classes_collector.get(class_meaning, []) + [
            file_list
        ]

        superclass_meaning2class_meaning[supeerclass_meaning] = (
            superclass_meaning2class_meaning.get(supeerclass_meaning, set())
            | {class_meaning}
        )
    # Т.е. каждый уникальный объект (не путать с классом), может быть представлен не в одном, а в нескольких изображениях. Это ...
    # ... необходимо учитывать при балансировке. Для этого каждому такому объекту ставится в соответствие список номеров индексов ...
    # ... фалов, в которых этот объект появляется. Словарь classes_collector осуществляет переход от расшифровки класса к списку ...
    # ... объектов этого класса, каждый из которых представлен как раз таким списком индексов изображений.

    # Подсчёт числа копий каждого семпла в датасете с помощью PyTorhc:
    history_file = (
        os.path.join(sts_dir, 'copy_bal_hist.csv') if os.path.isdir(sts_dir) else None
    )
    files2count, history = torch_copy_bal(
        files=files,
        file2index=file2index,
        classes_collector=classes_collector,
        superclass_meaning2class_meaning=superclass_meaning2class_meaning,
        steps=steps,
        max_file_copy_num=max_file_copy_num,
        max_ds_increase_frac=max_ds_increase_frac,
        device=device,
        history_file=history_file,
    )

    # Очищаем кеш оперативной памяти GPU после оптимизации, если она вообще доступна:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Выполняем дублирование:
    for file, count in files2count.items():
        # Пропускаем файлы, не требующие дублирования:
        if count == 1:
            continue

        # Определяем имена дублируемых файлов
        basename, ext = os.path.splitext(file)
        source_img_file = os.path.join(img_dir, file)
        source_lbl_file = os.path.join(lbl_dir, basename + '.txt')

        # Делаем нужное количество копий изображений и разметки:
        for num in range(1, count + 1):
            # Имена целевых файлов:
            target_img_file = os.path.join(img_dir, f'{basename}_copy_{num}{ext}')
            target_lbl_file = os.path.join(lbl_dir, f'{basename}_copy_{num}.txt')

            # Создание жёстких ссылок:
            os.link(source_img_file, target_img_file)
            os.link(source_lbl_file, target_lbl_file)

    return history
