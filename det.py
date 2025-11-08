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

from utils import draw_mask_on_image



class NMS:
    '''
    Классический Non-maximum Suppression для списка обнаруженных BBox-ов в
    кадре.
    '''