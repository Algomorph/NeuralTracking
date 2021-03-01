import numpy as np


def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3

    T_opengl_cv = np.array(
        [[1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]]
    )

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


