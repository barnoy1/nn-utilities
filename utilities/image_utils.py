import PIL
import numpy as np
from PIL import Image
import cv2

PIL.Image.MAX_IMAGE_PIXELS = None


def load_image_from_path(image_path):
    return cv2.imread(image_path, cv2.IMAGE_UNCHANGED).astype(np.uint8)


def load_normalized_image_from_path(image_path):
    return cv2.imread(image_path, cv2.IMAGE_UNCHANGED).astype(np.float32) / float(255.0)


def verify_image(image_path):
    try:
        with Image.open(image_path) as im:
            im.verify()
            return True
    except Exception as e:
        return False


def bin_to_color(image, color=None):
    bin_mask = (image > 0.5) * 1
    color_mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    coords = np.column_stack(np.where(bin_mask > 0))
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    if color is None: color = (255, 255, 255)
    color_mask[x, y] = color
    return color_mask


def is_empty(image):
    is_all_zero = np.all((image == 0))
    return is_all_zero


def deserialize_contour_points(point_list):
    points = [tuple(map(float, p.split(','))) for p in point_list.split(';')]
    points = np.array([(int(p[0]), int(p[1])) for p in points])
    points = points.astype(int)
    return points
