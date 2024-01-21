import alphashape as alphashape
import cv2
from matplotlib import pyplot as plt
from matplotlib.pyplot import *

from scipy.spatial import Delaunay
import math
import shapely.geometry as geometry
from descartes import PolygonPatch
from shapely.ops import polygonize, cascaded_union, unary_union

from pathology_analyzer import logger


def plot_polygon(polygon):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    margin = .3

    x_min, y_min, x_max, y_max = polygon.bounds

    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    patch = PolygonPatch(polygon, fc='#ffdede', ec='#ffdede', fill=True, zorder=-1)
    ax.add_patch(patch)
    return fig, patch


def alpha_shape(points, alpha):
    """
        Compute the alpha shape (concave hull) of a set of points.

        @param points: Iterable container of points.
        @param alpha: alpha value to influence the gooeyness of the border. Smaller
                      numbers don't fall inward as much as larger numbers. Too large,
                      and you lose everything!
        """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    coords = np.array([point for point in points])
    # coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    triangles = coords[tri.vertices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < alpha]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(np.concatenate((edge1, edge2, edge3)), axis=0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points


def compute_convex_hull(color_mask, rgb_value):
    gray = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    # Finding contours for the thresholded image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    cont = np.vstack(contours[i] for i in range(len(contours)))
    hull = cv2.convexHull(cont)
    img_contour = color_mask.copy()
    uni_hull = []
    uni_hull.append(hull)  # <- array as first element of list
    rgb_tuple = (int(rgb_value[0]), int(rgb_value[1]), int(rgb_value[2]))
    cv2.drawContours(img_contour, contours=uni_hull, contourIdx=-1, color=rgb_tuple, thickness=-1,
                     lineType=cv2.LINE_AA)
    return img_contour


def compute_concave_hull(color_mask, orig_color_mask, rgb_value, alpha=0):
    gray = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = np.array([list(pt[0]) for ctr in contours for pt in ctr])

    alpha_shape = None
    try:
        alpha_shape = alphashape.alphashape(contour_points, alpha=alpha)
        concave_hull = np.zeros((color_mask.shape[0], color_mask.shape[1], 3), np.uint8)
        points = []
        if alpha_shape.geometryType() == 'Polygon':
            vertices= alpha_shape.boundary.xy
            vx = [int(vertices) for vertices in vertices[0]]
            vy = [int(vertices) for vertices in vertices[1]]
            points = np.vstack([vx, vy]).T
            concave_hull[vy, vx] = rgb_value

        if alpha_shape.geometryType() == 'MultiPolygon':
            vertices = PolygonPatch(alpha_shape).get_verts()
            vertices_arr = np.split(vertices, 2, axis=0)
            vertices_list = vertices_arr[0].T
            vx = [int(v) for v in vertices_list[0]]
            vy = [int(v) for v in vertices_list[1]]
            points = np.vstack([vx, vy]).T
            concave_hull[vy, vx] = rgb_value

        rgb_tuple = (int(rgb_value[0]), int(rgb_value[1]), int(rgb_value[2]))
        cv2.drawContours(concave_hull, [points], -1, rgb_tuple, thickness=-1)

    except Exception as e:
        logger.error(f'failed calculate concave hull. using original component mask ...\n{e}')
        # fig, ax = plt.subplots()
        # ax.scatter(*zip(*contour_points))
        # ax.add_patch(PolygonPatch(alpha_shape, alpha=alpha))
        # plt.show()
        return orig_color_mask

    # convex_hull = compute_convex_hull(color_mask, rgb_value)

    # plt.imshow(convex_hull)
    # plt.imshow(concave_hull)

    return concave_hull

