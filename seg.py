import os
import cv2
import numpy as np

from utils import mkdirs, mpmap, ImReadBuffer
from cvat import CVATPoints, sort_tasks_by_len


def save_task2segmentation_dataset(task, inp_path, out_path, prv_path, labels_convertor, start_sample_ind=0):
    '''
    Выполняет сохранение задачи в файлы семплов для сегментационного датасета.
    '''
    sample_ind = start_sample_ind
    # Инициализация буфера чтения изображений:
    with ImReadBuffer() as buffer:
        
        # Перебор подзадач:
        for df, file, true_frames in task:
            
            # Заменяем все классы объектов в датафрейме на номера их суперклассов:
            df = labels_convertor.apply2df(df)
            
            # Перебираем все кадры:
            for frame, true_frame in true_frames.items():
                
                # Выбираем метки, соответствующие текущему кадру:
                frame_df = df[df['frame'] == frame]
                
                # Имена сохраняемых файлов:
                inp_file = os.path.join(inp_path, f'%08d.png' % sample_ind)
                out_file = os.path.join(out_path, f'%08d.png' % sample_ind)
                prv_file = os.path.join(prv_path, f'%08d.jpg' % sample_ind) if prv_path else None
                sample_ind += 1 # Прирощение номера семпла
                
                # Получаем все присутствующие типы объектов, содержащихся на изображении:
                superclass_labels = frame_df['label'].unique()
                
                # Флаг пропуска кадра:
                frame2drop = -2 in superclass_labels
                # Если в кадре есть объект исключённого класса, то кадр надо пропустить!
                
                # Если в кадре есть объект на исключение а ...
                # ... предпросмотр делать не надо, то пропускаем сразу:
                if prv_file is None and frame2drop:
                    continue
                
                # Читаем и сохраняем входное изображение:
                prv = buffer(file, true_frame, save2file=None if frame2drop else inp_file)
                # Изображение сохраняется только если нет исключённых объектов!
                
                # Формируем выходное изображение (ground truth):
                out = np.zeros(prv.shape[:2], prv.dtype)
                
                # Перебираем все номера суперклассов объектов в порядке убывания, чтобы  ...
                # ... сегменты объектов, имевщих меньший индекс, орисовывались поверх остальных:
                for label in sorted(superclass_labels, reverse=True):
                    
                    # Перебираем строки только объекты этого суперкласса:
                    for raw in frame_df[frame_df['label'] == label].iloc:
                        
                        # Создаём соответствующий объект CVATPoints:
                        points = CVATPoints.from_dfraw(raw)
                        
                        # Отрисовываем контур на превью:
                        if prv_file:
                            
                            # Задаём цвет описанной фигуры:
                            if  label == -1:
                                color = (255, 0, 0) # Синий   для неиспользуемых объектов
                            elif label > -1:
                                color = (0, 255, 0) # Зелёный для обычных        объектов
                            else:
                                color = (0, 0, 255) # Красный для исключённых    объектов
                            # Если исключения работают корректно, то ...
                            # ... для превью с красным цветом не должны ...
                            # ... существовать соответвствующие входное ...
                            # ... и выходное изображения!
                            
                            # Сама отрисовка контура:
                            try:
                                prv = points.draw(prv, label, color, 3, False)
                            except:
                                import json
                                print(frame)
                                print(file[frame])
                                with open(os.path.join(os.path.dirname(os.path.dirname(file[frame])),'task.json'), 'r', encoding='utf-8') as f:
                                    print(json.load(f)['name'])
                                raise
                        
                        # Если кадр не исключён и класс - не игнорируемый, ...
                        # ... то делаем выходные данные (ground truth):
                        if label >= 0 and not frame2drop:
                            out = points.draw(out, None, int(label) + 1, -1, False)
                
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
    
    # Получаем список индексов, с которых надо начинать нумерацию семплов в каждой задаче:
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

