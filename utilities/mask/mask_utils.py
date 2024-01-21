import configparser
import copy
import gc
import itertools
import pathlib
import sys
from pathlib import Path

import os
import getpass
import PIL
import cv2
import numpy as np
import skimage
import tifffile
import matplotlib.pyplot as plt
import torch
from scipy.spatial import distance

import pathology_analyzer.utils.log.color
from pathology_analyzer import logger
from pathology_analyzer.utils import file_utils, json_utils
from pathology_analyzer.utils._singleton import Singleton
from pathology_analyzer.utils.poly_utils import alpha_shape, plot_polygon, compute_concave_hull
from pathology_analyzer.utils.log import color

from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = None

def color_code_to_rgb(color_code):
    r = (color_code >> 16) & 0xFF
    g = (color_code >> 8) & 0xFF
    b = color_code & 0xFF
    return r, g, b


def bin_to_color_mask(mask, color=None):
    bin_mask = (mask > 0.5) * 1
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    coords = np.column_stack(np.where(bin_mask > 0))
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    if color is None: color = (255, 255, 255)
    color_mask[x, y] = color
    return color_mask

def is_empty_mask(mask):
    is_all_zero = np.all((mask == 0))
    if np.all(is_all_zero):
        return True
