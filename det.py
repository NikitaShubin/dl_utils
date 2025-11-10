'''
********************************************
*   Набор классов и функций для работы с   *
*      обрамляющими прямоугольниками и     *
*                детекцией.                *
********************************************
'''
import cv2

import numpy as np

from matplotlib import pyplot as plt

from cv_utils import (BBox, split_by_attrib, sort_by_attrib,
    build_bboxes_IoU_matrix)


class NMS:
    '''
    Классический Non-maximum Suppression для списка обнаруженных BBox-ов в
    кадре.
    '''

    def __init__(self, minIoU=0.5):
        self.minIoU = 0.5

    def __call__(self, bboxes):
        # Список из менее двух объектов оставляем без изменения:
        if len(bboxes) < 2:
            return bboxes

        # Группируем по классам:
        bboxes_list = split_by_attrib(bboxes)

        # Обрабатываем список для каждого класса отдельно:
        bboxes = []  # Итоговый список объектов
        for label_bboxes in bboxes_list:

            # Если объектов текущего класса меньше двух - переносим его в
            # итоговый без изменений:
            if len(label_bboxes) < 2:
                bboxes += label_bboxes
                continue

            # Упорядочиваем по убыванию уверенности:
            label_bboxes = sort_by_attrib(label_bboxes, nonmarked='raise')

            # Строим матрицу связностей:
            j_mat = build_bboxes_IoU_matrix(label_bboxes)

            # Перебираем все пары:
            excluded_inds = []  # Индексы исключённых объектов
            for i in range(len(label_bboxes) - 1):
                if i in excluded_inds:
                    continue  # Исключённые индексы пропускаем.

                for j in range(i + 1, len(label_bboxes)):
                    if j in excluded_inds:
                        continue  # Исключённые индексы пропускаем.

                    # Если пересечение текущей пары >= порогового, то исключаем
                    # тот, что с меньшей уверенностью:
                    if j_mat[i, j] >= self.minIoU:
                        excluded_inds.append(j)

            # Пополняем итоговый список объектов неисключёнными позициями:
            bboxes += [bbox for ind, bbox in enumerate(label_bboxes)
                       if ind not in excluded_inds]

        return bboxes