import os
import cv2
import numpy as np
from tqdm import tqdm

from utils import (rmpath, mkdirs, mpmap, ImReadBuffer, img_dir2video,
                   draw_contrast_text, AnnotateIt)
from cvat import CVATPoints, sort_tasks_by_len, cvat_backup_task_dir2info


def save_task2segmentation_dataset(task,
                                   inp_path,
                                   out_path,
                                   prv_path,
                                   labels_convertor,
                                   start_sample_ind=0):
    '''
    Выполняет сохранение задачи в файлы семплов для сегментационного датасета.
    '''
    sample_ind = start_sample_ind
    # Инициализация буфера чтения изображений:
    with ImReadBuffer() as buffer:

        # Перебор подзадач:
        for df, file, true_frames in task:

            # Определяем имя проекта и задачи:
            task_info = cvat_backup_task_dir2info(file)
            proj_name, task_name = task_info['proj_name'], task_info['task_name']

            # Используется при формировании превью.

            # Заменяем все классы объектов в датафрейме на номера их
            # суперклассов:
            df = labels_convertor.apply2df(df)

            # Перебираем все кадры:
            for frame, true_frame in true_frames.items():

                # Выбираем метки, соответствующие текущему кадру:
                frame_df = df[df['frame'] == frame]

                # Имена сохраняемых файлов:
                inp_file = os.path.join(inp_path, '%08d.png' % sample_ind)
                out_file = os.path.join(out_path, '%08d.png' % sample_ind)
                prv_file = os.path.join(prv_path, '%08d.jpg' % sample_ind) \
                    if prv_path else None

                sample_ind += 1  # Прирощение номера семпла

                # Получаем все присутствующие типы объектов, содержащихся на
                # изображении:
                superclass_labels = frame_df['label'].unique()

                # Флаг пропуска кадра:
                frame2drop = -2 in superclass_labels
                # Если в кадре есть объект исключённого класса, то кадр надо
                # пропустить!

                # Если в кадре есть объект на исключение а ...
                # ... предпросмотр делать не надо, то пропускаем сразу:
                if prv_file is None and frame2drop:
                    continue

                # Читаем и сохраняем входное изображение:
                prv = buffer(file, true_frame,
                             save2file=None if frame2drop else inp_file)
                # Изображение сохраняется только если нет исключённых объектов!

                # Формируем выходное изображение (ground truth):
                out = np.zeros(prv.shape[:2], prv.dtype)

                # Перебираем все номера суперклассов объектов в порядке
                # убывания, чтобы сегменты объектов, имевщих меньший индекс,
                # орисовывались поверх остальных:
                for label in sorted(superclass_labels, reverse=True):

                    # Перебираем строки только объекты этого суперкласса:
                    for row in frame_df[frame_df['label'] == label].iloc:

                        # Пропускаем объект, если он скрыт:
                        if row['outside']:
                            continue

                        # Создаём соответствующий объект CVATPoints:
                        points = CVATPoints.from_dfrow(row)

                        # Отрисовываем контур на превью:
                        if prv_file:

                            # Задаём цвет описанной фигуры:
                            if label == -1:  # Для неиспользуемых объектов:
                                color = (255, 0, 0)  # Синий
                            elif label > -1:  # Для обычных объектов
                                color = (0, 255, 0)  # Зелёный
                            else:  # Для исключённых объектов
                                color = (0, 0, 255)  # Красный
                            # Если исключения работают корректно, то ...
                            # ... для превью с красным цветом не должны ...
                            # ... существовать соответвствующие входное ...
                            # ... и выходное изображения!

                            # Сама отрисовка фигуры:
                            prv = points.draw(prv, label, color, 3)
                            prv = points.draw(prv, label, color, -1, 0.5)

                        # Если кадр не исключён и класс - не игнорируемый, ...
                        # ... то делаем выходные данные (ground truth):
                        if label >= 0 and not frame2drop:
                            out = points.draw(out, None, int(label) + 1, -1)

                # Наносим текст на превью:
                if prv_file:

                    # Наносим доп.иформацию:
                    text = f'Project: {proj_name}'
                    text += f'\nTask: {task_name}'
                    text += f'\nFrame: {true_frame}'
                    prv = draw_contrast_text(prv, text)
                    # prv = cv2.putText(prv, str(true_frame),
                    #                   (100, 100),
                    #                   cv2.FONT_HERSHEY_COMPLEX,
                    #                   1, 0, 2, cv2.LINE_AA)

                    '''
                    try:
                    except:
                        import json
                        print(frame)
                        print(file[frame])
                        err_file = os.path.join(os.path.dirname(
                            os.path.dirname(file[frame])), 'task.json')
                        with open(err_file, 'r', encoding='utf-8') as f:
                            print(json.load(f)['name'])
                        raise
                    '''

                # Сохраняем все положенные изображения:
                if prv_file      : cv2.imwrite(prv_file, prv)
                if not frame2drop: cv2.imwrite(out_file, out)

    return


def save_tasks2segmentation_dataset(tasks, path, labels_convertor, make_preview=False, desc=None):
    '''
    Создаёт в заданной папке обычный датасет для сегментации.
    '''
    # Определяем и создаём подпапки для сохранения входных и выходных данных:
    inp_path = os.path.join(path, 'inp'); mkdirs(inp_path)
    out_path = os.path.join(path, 'out'); mkdirs(out_path)

    # Повторяем тоже самое для папки с предпросмотром, если надо:
    if make_preview:
        prv_path = os.path.join(path, 'prv')
        mkdirs(prv_path)
    else:
        prv_path = None

    # Сортируем задачи по убыванию числа кадров и пол:
    task_lens, tasks = sort_tasks_by_len(tasks)

    # Получаем список индексов, с которых надо начинать нумерацию семплов в
    # каждой задаче:
    start_sample_inds = [0] + list(np.cumsum(task_lens[:-1]))

    # Сохраняем каждую задачу:
    mpmap(save_task2segmentation_dataset ,
          tasks                          ,
          [inp_path        ] * len(tasks),
          [out_path        ] * len(tasks),
          [prv_path        ] * len(tasks),
          [labels_convertor] * len(tasks),
          start_sample_inds              ,
          desc = desc)

    return


def draw_seg_prev(inp, out, num_classes, out_file=None, return_rzlt=True):
    '''
    Создаёт превью для семпла из датасета сегментации.
    Накладывает на исходное изображение маску цветом, соответствующим классу.
    inp и out должны быть строками с именами файлов, либо RGB-изображениями.
    '''
    # Читаем изображения, если переданы имена файлов:
    name = ''
    if isinstance(inp, str):
        name = os.path.basename(inp)
        inp = cv2.imread(inp)[..., ::-1]
    if isinstance(out, str):
        name = os.path.basename(out)
        out = cv2.imread(out, cv2.IMREAD_GRAYSCALE)

    # Переводим исходное изображение в HSV:
    rzlt = cv2.cvtColor(inp, cv2.COLOR_RGB2HSV)

    # Оттенок берём из номера класса:
    rzlt[..., 0] = (out / num_classes * 180).astype(np.uint8)
    rzlt[..., 1] = 255  # Насыщенностьвыкручиваем на максимум

    # Переводим результат в BGR:
    rzlt = cv2.cvtColor(rzlt, cv2.COLOR_HSV2BGR)

    # Вносим информацию об имени файла:
    rzlt = cv2.putText(rzlt,
                       name,
                       (100, 100),
                       cv2.FONT_HERSHEY_COMPLEX,
                       1, (255, 255, 255),
                       2, cv2.LINE_AA)

    # Сохраняем результат в файл, если он задан:
    if out_file:
        # Если файл уже существует, то объединяем старое превью с новым:
        if os.path.isfile(out_file):
            rzlt = np.hstack([rzlt, cv2.imread(out_file)])
        else:
            raise ValueError(out_file)

        cv2.imwrite(out_file, rzlt)
    else:
        raise ValueError(out_file)

    # Возвращаем результат в RGB, если надо:
    if return_rzlt:
        return rzlt[..., ::-1]


def make_seg_subset_colored_preview(subset_path, num_classes,
                                    make_video=False, desc=None):
    '''
    Создаёт/перезаписывает превью для подвыборки уже созданного датасета
    сегментации. Использует draw_seg_prev. Пишет видео, если надо.
    '''
    # Определяем и создаём подпапки для входных и выходных данных:
    inp_path = os.path.join(subset_path, 'inp')
    out_path = os.path.join(subset_path, 'out')
    prv_path = os.path.join(subset_path, 'prv')

    # Пропускаем обработку, если какой-то из папок не существует:
    inp_path_exists = os.path.isdir(inp_path)
    out_path_exists = os.path.isdir(out_path)
    prv_path_exists = os.path.isdir(prv_path)
    if not (inp_path_exists and out_path_exists and prv_path_exists):
        return

    '''
    # Очищаем папку с превью:
    rmpath(prv_path)
    mkdirs(prv_path)
    '''

    # Составляем списки входных и выходных файлов:
    inp_files = sorted([os.path.join(inp_path, file)
                        for file in os.listdir(inp_path)])
    out_files = sorted([os.path.join(out_path, file)
                        for file in os.listdir(out_path)])
    prv_files = sorted([os.path.join(prv_path, file)
                        for file in os.listdir(inp_path)])
    prv_files = [os.path.splitext(prv_file)[0] + '.jpg'
                 for prv_file in prv_files]
    # Превью должны храниться в JPG.

    # Если папка inp или out пуста, то ничего не делаем:
    if min(map(len, [inp_files, out_files])) == 0:
        return

    mpmap(draw_seg_prev,
          inp_files,
          out_files,
          [num_classes] * len(inp_files),
          prv_files,
          [False] * len(inp_files),
          batch_size=128,
          desc=desc)

    # Создаём видео-превью, если надо:
    if make_video:
        tmp_file = prv_path + '.avi'
        trg_file = prv_path + '.mp4'

        # Собираем временный файл из отдельных кадров:
        img_dir2video(prv_path, tmp_file, fps=30,
                      desc=f'Сборка видео для {desc}')

        # Пересжимаем с использованием межкадрового сжатия:
        with AnnotateIt(f'Пересжатие {desc}...',
                        f'Пересжатие {desc} завершено.') as t:
            if os.system('ffmpeg -y -hide_banner -loglevel quiet ' +
                         f'-i "{tmp_file}" -c:v libx264 -preset slow ' +
                         f'-crf 20 -tune animation "{trg_file}"'):
                os.remove(trg_file)
            else:
                os.remove(tmp_file)


def make_seg_dataset_colored_preview(path, num_classes,
                                     make_video=False, desc='Сборка превью'):
    '''
    Создаёт/перезаписывает превью для уже созданного датасета сегментации.
    Пишет видео, если надо.
    '''
    for subset in ['train', 'val', 'test']:
        make_seg_subset_colored_preview(os.path.join(path, subset),
                                        num_classes, make_video,
                                        desc=f'{desc} ({subset})')


def drop_homogeneous_labeled_frames(dataset_path):
    '''
    Выбрасывает из датасета неразмеченные семплы.
    Признаком отсуствия разметки является принадлежность всех пикселей одному
    и тому же классу.
    '''
    for subset in ['train', 'val', 'test']:
        # Составляем списки входных и выходных файлов:
        path = os.path.join(dataset_path, subset)
        inp_path = os.path.join(path, 'inp')
        out_path = os.path.join(path, 'out')
        prv_path = os.path.join(path, 'prv')
        inp_files = sorted(os.listdir(inp_path))
        out_files = sorted(os.listdir(out_path))
        prv_files = sorted(os.listdir(prv_path))
        prv_files = [os.path.splitext(prv_file)[0] + '.jpg'
                     for prv_file in prv_files]
        # Превью должны храниться в JPG.

        acc = 0
        for file in tqdm(out_files):
            inp_file = os.path.join(inp_path, file)
            out_file = os.path.join(out_path, file)
            prv_file = os.path.join(prv_path, file)
            prv_file = os.path.splitext(prv_file)[0] + '.jpg'

            img = cv2.imread(out_file)
            if len(np.unique(img)) == 1:

                # 2del 1111111111111111!!!!!!!!!!!!!!
                assert os.path.isfile(prv_file)

                acc += 1

                os.remove(prv_file)
                os.remove(inp_file)
                os.remove(out_file)
                if os.path.isfile(prv_file):
                    os.remove(prv_file)

        print('Отброшено', acc, 'семплов!')