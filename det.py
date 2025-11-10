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


