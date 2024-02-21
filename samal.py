'''
********************************************
*   Использование Segment Anything Model   *
*  (SAM) для автоматической разметки (AL). *
*                                          *
* Применяется для предварительной          *
* разметки изображений в CVAT.             *
*                                          *
*                                          *
* Основные функции и классы:               *
*   SAM - класс, инкапсулирующий Segment   *
*       Anything Model с несколькими       *
*       наиболее востребованными функциями *
*       обработки изображений на её        *
*       основе.                            *
*                                          *
********************************************
'''

import os
import cv2
import torch

import pandas as pd
import numpy  as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from urllib.request import urlretrieve

from utils import mkdirs, AnnotateIt, mpmap, reorder_lists, isint
from cvat import CVATPoints, add_row2df
from cv_utils import Mask, build_masks_JaccardDiceOverlap_matrixs, build_masks_IoU_matrix


class MasksFilter:
    '''
    Суперкласс для фильтров поэлементной обработки списков масок.
    Также может быть использован как декоратор для функций, обрабатывающих одну маску.
    '''
    def __init__(self, mask_filter=None):
        if mask_filter is not None:
            self.mask_filter = mask_filter
            self.__doc__ = mask_filter.__doc__
        else:
            raise ValueError('Функция "mask_filter" должна быть определена!')
    
    # Обработка всего списка масок:
    def __call__(self, masks):
        
        # Инициализация итогового списка масок:
        new_masks = []
        
        # Переборк каждой маски из старого списка:
        for mask in masks:
            
            # Применяем фильтра к маске:
            mask = self.mask_filter(mask)
            
            # Если результат является списком, кортежем или множеством, ...
            # ... то добавляем его элементы в итоговый список:
            if isinstance(mask, (list, tuple, set)):
                new_masks += list(mask)
            # Сделано на случай если в результате фильтрации из одной маски ...
            # ... в результате фильтрации создаётся целый список масок.
            
            # Если маска вообще возвращена, то добавляем её в итоговый список:
            elif mask is not None:
                new_masks.append(mask)
        
        return new_masks

class PrintMasksNum:
    '''
    Выводит текущее количество масок.
    Используется для отладки.
    '''
    def __init__(self, decs='Текущее кол-во масок:'):
        self.desc = decs
    
    def __call__(self, masks):
        print(self.desc, len(masks))
        return masks


class Morphology:
    '''
    Морфология закрытия каждого сегмента.
    
    Полезна для сглаживания контуров в
    сторону их увеличения.
    '''
    def __init__(self, morph_type='close', kernel=5, desc=None):
        
        # Если структурный элемент задан целым числом, то ...
        # ... заранее создаём структурный элемент в виде окружности:
        if isinstance(kernel, int):
            self.kernel = Mask.make_kernel(kernel)
        
        # Если структурный элемент задан вещественным числом, то рассчитывать ...
        # ... его надо каждый раз с учётом размера обробатываемого сегмента:
        elif isinstance(kernel, float):
            self.kernel = kernel
        
        # Если структурный элемент уже задан numpy-массивом:
        elif isinstance(kernel, np.ndarray):
            self.kernel = kernel
        
        # Если ядро не задано целым, вещественнвм и не имеет множества значений:
        else:
            raise ValueError('Тип `kernel` должен быть матрицей, целым или' +
                             f' вещественным числом, а является {type(kernel)}!')
        
        self.desc = desc
        
        # Если тип морфологии задан строкой, то приводим её к нижнему регистру:
        if isinstance(morph_type, str):
            morph_type = morph_type.lower()
        
        # Доопределяем тип преобразования:
        if   morph_type in [cv2.MORPH_DILATE, 'dilate', 'дилатация', 'd', 'д']: self.morph_type = cv2.MORPH_DILATE
        elif morph_type in [cv2.MORPH_ERODE , 'erode' , 'эрозия'   , 'e', 'э']: self.morph_type = cv2.MORPH_ERODE
        elif morph_type in [cv2.MORPH_CLOSE , 'close' , 'закрытие' , 'c', 'з']: self.morph_type = cv2.MORPH_CLOSE
        elif morph_type in [cv2.MORPH_OPEN  , 'open'  , 'открытие' , 'o', 'о']: self.morph_type = cv2.MORPH_OPEN
        else:
            raise ValueError(f'Некорректное значение параметра "morph_type": {morph_type}')

    def do_morphology_on_mask(self, mask):
        return mask.morphology(self.morph_type, self.kernel)
    
    def __call__(self, masks):
        
        # Если размер ядра 1 х 1, предпологаем, ...
        # ... что операцию применять бессмысленно:
        if not isinstance(self.kernel, np.ndarray) and (self.kernel <= 1) and isint(self.kernel) or \
               isinstance(self.kernel, np.ndarray) and (self.kernel.size == 1):
            return masks
        
        num_procs = len(masks) < 2
        return mpmap(self.do_morphology_on_mask, masks, desc=self.desc, num_procs=num_procs)
        '''
        # Инициализируем список итоговых масок:
        masks_ = []
        
        # Перебираем каждую маску исходного списка:
        for mask in tqdm(masks, desc=self.desc, disable=not self.desc):
            
            # Выполняем морфологию и заносим результат в итоговый список:
            masks_.append()
        
        return masks_
        '''

def DropByArea(min_area=None, max_area=None):
    '''
    Создаёт функтор, выбрасывающий из списка масок те из них,
    чьи размеры выходят за рамки диапазона [min_area, max_area].
    Целочисленные значения min_area и max_area должны быть > 1
    и воспринимаются как кол-во пикселей, а вещественные должны
    быть в интервале (0., 1.) и воспринимаются как доли от
    общей площади изображения.
    '''
    # Если оба параметра остались неопределёнными, то выводим ошибку:
    if min_area is None:
        if max_area is None:
            raise ValueError('Должен быть определён хотябы один из двух порогов ("min_area" или "max_area")!')
        
        # Если определён только максимальный размер:
        else:
            
            # Объявляем функцию, отбрасывающую все сегменты больше заданного размера:
            @MasksFilter
            def drop_by_area(mask, max_area=max_area):
                '''
                Исключение сегментов, площади которых больше заданного максимума "{max_area}".
                '''
                # Переводим порог из относительного в абсолютный, если надо:
                if 0. < max_area < 1.: max_area = max_area * mask.size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if mask.area() <= max_area:
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых больше заданного максимума {max_area}.'
    
    else:
        
        # Если объявлен только минимальный размер:
        if max_area is None:
            
            # Объявляем функцию, отбрасывающую все сегменты меньше заданного размера:
            @MasksFilter
            def drop_by_area(mask, min_area=min_area):
                '''
                Исключение сегментов, площади которых меньше заданного минимума "min_area".
                '''
                # Переводим порог из относительного в абсолютный, если надо:
                if 0. < min_area < 1.: min_area = min_area * mask.size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if min_area <= mask.area():
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых меньше заданного минимума {min_area}.'
        
        # Если заданы оба размера:
        else:
            
            # Объявляем функцию, отбрасывающую все сегменты, размеры которых выходят за заданный диапазон:
            @MasksFilter
            def drop_by_area(mask, min_area=min_area, max_area=max_area):
                '''
                Исключение сегментов, площади которых выходят за заданный диапазон ["min_area", "max_area"].
                '''
                # Переводим пороги из относительных в абсолютные, если надо:
                if 0. < min_area < 1.: min_area = min_area * mask.size
                if 0. < max_area < 1.: max_area = max_area * mask.size
                
                # Если площать сегмента вписывается в заданный диапазон, то маска не отбрасывается:
                if min_area <= mask.area() <= max_area:
                    return mask
                
                # Иначе возвращаем None:
                return
            
            # Переопределяем строку описания:
            drop_by_area.__doc__ = f'Исключение сегментов, площади которых выходят за заданный диапазон [{min_area}, {max_area}].'
    
    # Возвращаем построенный функтор:
    return drop_by_area


class DropDuplicates:
    '''
    Функтор исключения дубликатов в списках сегментов.
    
    Решение о совпадении двух сегментов принимается при
    превышении IoU (Intersection-Over-Union, он же
    индекс Жаккара) заданного порогового значения
    max_iou. Из двух сегментов остаётся бOльший, если
    не менять параметр exclude_smaller.
    '''
    def __init__(self, max_iou=0.85, total_parallel_num=64, desc=None):
        self.max_iou = max_iou
        self.total_parallel_num = total_parallel_num
        self.desc = desc
    
    # Флаг сопоставимости площадей:
    def is_comparable_area(self, area1, area2):
        return min(area1, area2) / max(area1, area2) >=  self.max_iou
    
    # Флаг эквивалентности масок:
    def is_equal_masks(self, mask1, mask2, area1, area2):
        if self.is_comparable_area(area1, area2):
            return mask1.Jaccard_with(mask2) >= self.max_iou
        else:
            return False
    
    # 
    def __call__(self, masks):
        # Возвращаем список без изменений, если в нём менее 2-х элементов:
        if len(masks) <= 1: return masks
        
        # Инициализация множества индексов исключённых масок:
        drop_inds = set()
        
        # Площади сегментов:
        areas = [mask.area() for mask in masks]

        # Сортируем маски по площади для большей кучности похожих сегментов:
        #areas, masks = reorder_lists(np.argsort(areas), areas, masks)
                
        # Инициируем список проверенных масок:
        chacked_masks = []
        
        # До тех пор пока список непроверенных масок слишком большой ...
        # Мы выполняем параллельный перебор первой маски со всеми остальными:
        start_len = len(masks)
        if start_len > self.total_parallel_num:
            
            desc = self.desc + ' (предварительное прореживание)' if self.desc else None
            progress = tqdm(total=start_len - self.total_parallel_num, desc=desc, disable=not desc)
            '''
            # Инициируем индекс последнего подходящего по размеру сегмента в списке:
            right_ind = 1 - len(masks)
            # Индекс отсчитывается справа, а не слева чтобы ...
            # ... сохранить инвариантность к сокращению длины списка.
            # Единица добавляется т.к. дальше первый элемент исключается из списка.
            '''
            while len(masks) > self.total_parallel_num:
                
                # Извлекаем первую маску и её площадь из списков непроверенных:
                mask = masks.pop(0)
                area = areas.pop(0)
                
                # Заносим первую маску в список проверенных:
                chacked_masks.append(mask)
                '''
                # Обновляем индекс первого не соответсвующего по площади элемента:
                max_area = area / self.max_iou # Максимально допустимая площадь
                right_ind = max(right_ind, -len(masks))
                for right_ind in range(right_ind, 0):
                    if areas[right_ind] >= max_area:
                        break
                
                # Берём подмножество сегментов, подходящих по площади:
                masks_ = masks[:right_ind]
                areas_ = areas[:right_ind]
                
                # Строим флаги повторов для этого подмножества:
                flags = mpmap(self.is_equal_masks,
                              [mask] * len(masks_), masks_,
                              [area] * len(areas_), areas_,
                              batch_size=len(areas_) // os.cpu_count() // 4)
                '''
                # Фиксируем текущую длину полного списка сегментов:
                delta = len(masks)
                
                # Обновляем списки сегментов, выкидывая отмеченные флагами:
                #masks = [_ for _, flag in zip(masks_, flags) if not flag] + masks[right_ind:]
                #areas = [_ for _, flag in zip(areas_, flags) if not flag] + areas[right_ind:]
                
                #'''
                # Получаем список флагов совпадения первой маски с каждой из оставшихся:
                flags = mpmap(self.is_equal_masks,
                              [mask] * len(masks), masks,
                              [area] * len(areas), areas,
                              batch_size=len(areas) // os.cpu_count() // 4,
                              num_procs=1)
                
                # Выкидываем все совпадения из списка непроверенных масок:
                delta = len(masks)
                masks = [_ for _, flag in zip(masks, flags) if not flag]
                areas = [_ for _, flag in zip(areas, flags) if not flag]
                #'''
                
                delta -= len(masks)        # Число выкинутых сегментов
                progress.update(delta + 1) # Обновляем статусбар
            
            progress.close()
        # При достижении достаточно малого размера списка непроверенных масок ...
        # ... выполняем проверку всех со всеми:
        
        # Строим матрицу связностей:
        j_mat = build_masks_IoU_matrix(masks, desc=self.desc)
        
        # Перебираем маски в качестве первого объекта для сравнения (пропуская исключения):
        for i in range(len(masks) - 1):
            if i in drop_inds: continue
            
            # Перебираем маски в качестве второго объекта для сравнения (пропуская исключения):
            for j in range(i + 1, len(masks)):
                if j in drop_inds: continue
                
                # Если два сегмента почти идентичны (т.е. IoU превышает пороговое значение):
                if j_mat[i, j] > self.max_iou:
                    
                    # Добавляем более позний элемент списка в исключения:
                    drop_inds.add(j)
        
        # Выбрасываем дубликаты:
        masks = [mask for ind, mask in enumerate(masks) if ind not in drop_inds]
        
        return chacked_masks + masks

@MasksFilter
def isolate_segments(mask):
    '''
    Разбиение развалившихся сегментов в отдельные маски.
    Полезно в случае если SAM относит два отдельных
    сегмента к одному объекту (в одну маску).
    '''
    # Разбиение маски на сегменты:
    splitted_masks = mask.split_segments()
    
    # Если найдено не больше одного сегмента, то возвращаем список из одного элемента - оригинальной маски:
    if len(splitted_masks) < 2:
        return [mask]
    
    new_masks = []
    for splitted_mask in splitted_masks:
        
        # Переводим маску в бинарный формат:
        splitted_mask = splitted_mask.astype(bool)
        
        # Вносим обновлённую маску в итоговый список:
        new_masks.append(splitted_mask)
    
    return new_masks


class Decompose:
    '''
    Функтор взрывного увеличения количества вариантов масок
    путём перебора всевозможных вычитаний одной маски из другой.
    
    Если коэффициент наложения (overlap) у двух масок превышает
    заданный порог (если одна маска почти полностью входит в
    другую), то добавляем в список масок результат такого
    вычитания.
    '''
    def __init__(self,
                 min_overlap  : 'Значение наложения при котором считается что один сегмент вписан в другой' = 0.90            ,
                 min_iou      : 'Значение IoU при котором считается, что сегменты пересекаются'             = 0.05            ,
                 max_iou      : 'Значение IoU с которого считается, что сегменты совпадают'                 = 0.99            ,
                 step_postproc: 'Обработка списков после каждой итерации'                                   = isolate_segments,
                 steps        : 'Максимальное число итераций'                                               = 1               ,
                 desc         : 'Название статусбара'                                                       = None            ):
        
        # Проверяем параметры на противоречивость:
        assert max_iou > min_iou
        assert min_iou < min_overlap
        
        # Если функция, то делаем из неё список:
        if hasattr(step_postproc, '__call__'):
            step_postproc = [step_postproc]
        
        self.min_overlap   = min_overlap
        self.min_iou       = min_iou
        self.max_iou       = max_iou
        self.step_postproc = step_postproc
        self.steps         = steps
        self.desc          = desc
    
    # Примерение операции над двумя масками:
    @staticmethod
    def apply_op(arg1, arg2, op='sub'):
        op = op.lower()
        
        if   op in {'sub',        '-'     }: return arg1 - arg2
        elif op in {'or' , 'sum', '+', '|'}: return arg1 | arg2
        elif op in {'and', 'mul', '*', '&'}: return arg1 & arg2
        else: raise ValueError(f'Неизвестный оператор "{op}"!')
    
    def __call__(self, masks):
        
        # Инициируем фильтр, отбрасывающий повторы:
        desc_ = '%s (отбрасывание повторов)' % self.desc if self.desc else None
        drop_duplicates = DropDuplicates(self.max_iou, desc=desc_)
        
        # Перебераем итерации:
        for step in range(self.steps):
            
            # Выводим номер итерации, если надо:
            if self.desc and self.steps > 1:
                print(f'{self.desc} - итерация {step + 1}/{self.steps} - {len(masks)} элементов:')
            
            # Формируем таблицы Дайеса, Жаккара и перекрытия всех пар масок в списке:
            desc_ = '%s (Подсчёт связностей)' % self.desc if self.desc else None
            j_map, d_map, o_map = build_masks_JaccardDiceOverlap_matrixs(masks, desc=desc_)
            
            args1     = [] # Список первых аргументов
            args2     = [] # Список вторых аргументов
            ops       = [] # Список операций
            inds2drop = [] # Список индексов исключаемых масок
            
            # Перебираем первую маску:
            for i, imask in enumerate(masks[:-1]):
                
                # Пропускаем, если сегмент подлежит исключению:
                if i in inds2drop: continue
                
                # Перебираем вторую маску:
                for j, jmask in enumerate(masks[i + 1:], i + 1):
                    
                    # Пропускаем, если сегмент подлежит исключению:
                    if j in inds2drop: continue
                    
                    # Пропускаем сочетание масок, если оно не имеет пересечений:
                    if o_map[i, j] == 0: continue
                    
                    # Исключаем второй сегмент, если он практически совпадает с первым:
                    if j_map[i, j] >= self.max_iou:
                        inds2drop.append(j)
                        continue
                    
                    # Если один сегмент вписан в другой:
                    if o_map[i, j] >= self.min_overlap:
                        
                        # Вычитаем из большего меньший:
                        if imask.area() > jmask.area():
                            args1.append(imask); args2.append(jmask), ops.append('-')
                        else:
                            args1.append(jmask); args2.append(imask), ops.append('-')
                    
                    # Если сегменты пересекаются:
                    elif j_map[i, j] >= self.min_iou:
                        args1.append(imask); args2.append(jmask), ops.append('+') # m1 + m2
                        args1.append(imask); args2.append(jmask), ops.append('*') # m1 * m2
                        args1.append(imask); args2.append(jmask), ops.append('-') # m1 - m2
                        args1.append(jmask); args2.append(imask), ops.append('-') # m2 - m1
            
            # Выкитываем ненужные маски из исходного списка:
            masks = [mask for ind, mask in enumerate(masks) if ind not in inds2drop]
            # Выкидываются повторы, которые вскрылись на этапе взаимного сопоставления масок.
            
            # Получаем новые маски:
            desc_ = '%s (комбинаторное расширение списка сегментов)' % self.desc if self.desc else None
            new_masks = mpmap(self.apply_op, args1, args2, ops, desc=desc_)
            
            # Применяем к новым маскам постобработку:
            for postprocess_filter in self.step_postproc:
                new_masks = postprocess_filter(new_masks)
            
            # Объединяем списки масок:
            masks += new_masks
            
            # Убираем повторы:
            masks = drop_duplicates(masks)
        
        return masks


class MajorSegment:
    '''
    Выбирает одну маску, максимально близкую шаблонной.
    Все входящие в её меньшие сегменты удаляются,а из 
    всех бОльших сегментов, в которые этот сегмент сам
    входит, он вычитается. Т.о. этот объект как бы
    становится главным в заданной области.
    
    Используется, если есть представление о том, что
    какой-то крупный объект занимает значительную область
    кадра и входящие в него подсегменты можно удалить, а
    более крупные области от него отделить.
    '''
    def __init__(self, mask_of_interest, desc=None):
        
        # Если объект не является экземпляром класса Mask, то исправляем это:
        if not isinstance(mask_of_interest, Mask):
            mask_of_interest = Mask(mask_of_interest)
        
        self.original_mask_of_interest = mask_of_interest
        self.         mask_of_interest = mask_of_interest
        self.desc = desc
    
    def __call__(self, masks):
        # Если список масок пуст, то просто возвращаем его:
        if len(masks) == 0:
            return masks
        
        # Ищем сегмент с максимальным IoU с областью интереса ...
        # ... (эту маску и будем считать сегментом жатки):
        major_IoU = 0 # Инициируем максимизируемый параметр (индекс Жаккара)
        for i, mask in enumerate(tqdm(masks, desc='%s (поиск жатки)' % self.desc , disable=not self.desc)):
            
            # Адаптируем размер маски к размеру изображения если надо:
            if self.mask_of_interest.array.shape[:2] != mask.array.shape[:2]:
                mask_of_interest = self.original_mask_of_interest.array.astype(np.uint8)
                mask_of_interest = cv2.resize(mask_of_interest,
                                              mask.array.shape[:2][::-1],
                                              interpolation=cv2.INTER_NEAREST)
                mask_of_interest = mask_of_interest.astype(self.original_mask_of_interest.array.dtype)
                self.mask_of_interest = Mask(mask_of_interest)
            # Подсчитываем индекс Жаккара для совпадения текущего сегмента с областью интереса:
            IoU = mask.Jaccard_with(self.mask_of_interest)
            
            # Если предыдущий рекорд побит, то фиксируем нового лидера:
            if IoU > major_IoU:
                major_IoU      = IoU
                major_mask_ind = i
                major_mask     = mask
        
        if self.desc:
            print(f'IoU главного объекта с областью интереса составляет {max_IoU}')
        
        # Ищем и удаляем все сегменты, входящие в гавный сегмент:
        masks_ = [major_mask] # Инициируем итоговый список сегментов включением в него гланвый объект
        for i, mask in enumerate(tqdm(masks, desc='%s (удаление деталей)' % self.desc, disable=not self.desc)):
            
            # Исключаем из проверки сам главный сегмент:
            if i == major_mask_ind:
                continue
            
            # Если коэффициент перекрытия близок к единице:
            if major_mask.Overlap_with(mask) > 0.999:
                
                # Если главная маска больше текущей маски, ...
                # ... то не включаем её в итоговый список:
                if major_mask.area() > mask.area():
                    continue
                
                # Если маска текущего сегмента болье маски жатки, ...
                # ... то вычитаем из неё маску жатки и сохраняем результат:
                else:
                    masks_.append(mask - major_mask)
                    continue
            
            # Остальные сегменты к главному объекту не относятся ...
            # ... и мы просто переносим их в итоговый список без изменений:
            masks_.append(mask)
        
        return masks_


class SAM:
    '''
    Класс предварительной сегментации с помощью Segment Anything Model.
    '''
    def __init__(self,
                 model_path          = '../sam_vit_h_4b8939.pth',
                 device              = 'auto'                   ,
                 use_tta             = True                     ,
                 postprocess_filters = []                       ):
        
        # Определяем, доступно ли GPU:
        if device == 'auto':
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Качаем модель, если её не оказалось в указанном месте:
        if not os.path.exists(model_path):
            
            # Определяем имя и путь до файла-модели:
            file_dir, file_name = os.path.split(model_path)
            
            # Создаём папку, если её не было:
            mkdirs(file_dir)
            
            # Путь до модели в вебе:
            url = os.path.join('https://dl.fbaipublicfiles.com/segment_anything/', file_name)
            
            # Загрузка:
            with AnnotateIt(f'Загрузка модели {url} в "{file_dir}" ...',
                            f'Загрузка модели {url} в "{file_dir}" завершена!'):
                urlretrieve(url, model_path)
        
        # Загружаем модель (на GPU, если оно доступно)
        sam = sam_model_registry['default'](checkpoint=model_path).to(device=device)
        
        # Создаём метод однократной обработки изображения:
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
        # Использовать ли Test Time Augmentation вместо обычного инференса?
        self.use_tta = use_tta
        
        # Сохраняем список фильтров:
        self.postprocess_filters = postprocess_filters
    
    # Обычное применение модели:
    def inference(self, img):
        return self.mask_generator.generate(img)
    
    # Test Time Augmentation:
    def tta(self, img, leave=False, desc='Инференс для обратимо аугментированных данных'):
        '''
        Отражение по вертикали и горизонтали, а также поворот на
        90 градусов образует 8 вариантов обратимой аугментации.
        '''
        # Инициализация списка масок:
        masks = []
        
        # Оборачиваем вложенные циклы в менеджер контекста статусбара:
        with tqdm(total=8, desc=desc, disable=not desc) as pbar:
            
            # Перебор вариантов применения горизонтального отражения:
            for hflip in [True, False]:
                
                # Применяем горизонтальную аугментацию:
                aug_img_h = np.fliplr(img) if hflip else img
                
                # Перебор вариантов применения вертикального отражения:
                for vflip in [True, False]:
                    
                    # Применяем вертикальную аугментацию:
                    aug_img_hv = np.flipud(aug_img_h) if vflip else aug_img_h
                    
                    # Перебор количества поворотов на 90 градусов:
                    for k in range(2):
                        
                        # Применяем поворот на нужный угол:
                        aug_img_hvk = np.rot90(aug_img_hv, k)
                        
                        # Получаем маски для преобразованного изображения:
                        aug_masks = self.inference(aug_img_hvk)
                        
                        # Выполняем обратные преобразования над полученными масками:
                        for mask in aug_masks:
                            m    = mask['segmentation']
                            bbox = mask['bbox'        ]
                            pc   = mask['point_coords']
                            cb   = mask['crop_box'    ]
                            
                            imsize = m.shape[:2]
                            bbox =  CVATPoints.from_bbox(*bbox, imsize)
                            pc   = [CVATPoints.from_bbox( *_*2, imsize) for _ in pc]
                            cb   =  CVATPoints.from_bbox(  *cb, imsize)
                            
                            # Обращение поворота:
                            if k:
                                m    =   np.rot90(m, -k)
                                bbox = bbox.rot90(-k)
                                pc   = [  _.rot90(-k) for _ in pc]
                                cb   =   cb.rot90(-k)
                            
                            # Обращение вертиеального отражения:
                            if vflip:
                                m    =   np.flipud(m)
                                bbox = bbox.flipud()
                                pc   = [  _.flipud() for _ in pc]
                                cb   =   cb.flipud()
                            
                            # Обращение горизонтального отражения:
                            if hflip:
                                m    =   np.fliplr(m)
                                bbox = bbox.fliplr()
                                pc   = [  _.fliplr() for _ in pc]
                                cb   =   cb.fliplr()
                            
                            # Сборка исправлений в новую маску:
                            mask['segmentation'] = m
                            mask['bbox'        ] = bbox.asbbox()
                            mask['point_coords'] = [  _.asbbox()[:2] for _ in pc]
                            mask['crop_box'    ] =   cb.asbbox()
                            
                            # Внесение маски в список:
                            masks.append(mask)
                        
                        # Обновление статусбара:
                        pbar.update()
        
        return masks
    
    # Постобработка списка масок:
    def postprocess(self, masks):
        
        # Последовательное применение к списку сегментов каждого из фильтров:
        if len(self.postprocess_filters):
            
            # Оборачиваем маски COCO-формата из SAM в экземпляры класса Mask:
            masks = list(map(Mask.from_COCO_annotation, masks))
            
            # Выполняем последовательное приминение ф-ий постобработки:
            for postprocess_filter in self.postprocess_filters:
                masks = postprocess_filter(masks)
            
            # Возвращение из Mask в COCO:
            masks = [mask.as_COCO_annotation() for mask in masks]
        
        return masks
    
    # Применение сети к изображению:
    def img2masks(self, img, *args, **kwargs):
        
        # Применяем разовый инференс или Test Time Augmentation:
        masks = self.tta(img, desc=None) if self.use_tta else self.inference(img)
        
        # Постобработка масок:
        return self.postprocess(masks)
    
    # Конвертация масок в датафрейм формата cvat-задач:
    @staticmethod
    def masks2df(masks, source='SAM', cup_holes=True, **kwarg):
        
        # Список датафреймов для последующего объединения:
        dfs = []
        
        # Перебор всех найденных масок:
        for mask in masks:
            
            # Векторизируем маску:
            points = CVATPoints.from_mask(mask['segmentation']).reducepoly()
            #points = points.re
            #m = points.split_multipoly()
            '''
            for _ in m:
                if isinstance(_, list):
                    print(_)
                    raise
                
                try:
                    m.reducepoly()
                    print('ok')
                except:
                    print('[', end='')
                    for __ in points.flatten():
                        print()
                    print(points.points)
                    points.show()
                    raise
            try:
                points = points.reducepoly()
            except:
                from PIL.Image import fromarray
                fromarray(mask['segmentation'])
                raise
            '''
            # Если надо исключить все внутренние контуры:
            if cup_holes:
                
                # Получаем Расщеплённые контуры:
                multipoly = points.split_multipoly()
                
                # Если контуров действительно больше одного, то оставляем наибольший:
                if len(multipoly) > 0:
                    
                    # Инициируем бОльшую площадь:
                    max_area = 0
                    
                    # Перебираем контуры:
                    for p in multipoly:
                        
                        # Если фигура является точкой, то пропускаем:
                        if len(p) == 2:
                            continue
                        # У точки нулевая площадь.
                        
                        # Подсчитываем площадь обрамляющего прямоугольника:
                        _, _, w, h = p.asbbox()
                        area = w * h
            
                        # Выбираем наибольший сегмент:
                        if area > max_area:
                            max_area = area
                            points = p
            # Предпологается, что сегмент всего один, и все контуры, кроме самого большого являются внутренними (дырами).
            
            # Собираем новый датафрейм из одной строки:
            df = add_row2df(None, points=points.flatten(), type=points.type, rotation=points.rotation, source=source, **kwarg)
            
            # Вносим строку в список:
            dfs.append(df)
        
        # Объединение всех строк в один датафрейм:
        df = pd.concat(dfs)
        
        # Меняем тип метки, если число точек мало для многоугольника:
        df.loc[df['points'].apply(len) == 4, 'type'] = 'polyline' # Если в многоугольнике 2 точки, то это - линия
        df.loc[df['points'].apply(len) == 2, 'type'] = 'points'   # Если в многоугольнике 1 точка, то это - точки
        
        return df
    
    # Формирование разметки в cvat-формате на основе полученного изображения:
    def img2df(self, img):
        
        # Формирование масок:
        masks = self.img2masks(img)
        
        # Перевод масок в датафрейм cvat-формата:
        return self.masks2df(masks)
    
    # Формируем разметку в формате CVAT-подзадачи по входному файлу с изображением:
    def img_file2subtask(self, img_file):
        
        # Читаем исходное изображение в формате RGB:
        img = cv2.imread(img_file)[..., ::-1]
        
        # Применяем модель для получения списка масок:
        masks = self.img2masks(img)
        
        # Переводим маски в датафрейм:
        df = self.masks2df(masks)
        
        # Возвращаем собранную подзадачу:
        return df, img_file, {0:0}
    
    # Файл с изображением -> разметка в формате CVAT-подзадачи:
    __call__ = img2df

