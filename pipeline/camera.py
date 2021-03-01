import numpy as np
import open3d as o3d
from typing import Tuple


def load_intrinsics_from_text_4x4_matrix_and_first_image(path_matrix: str, path_image: str) -> o3d.camera.PinholeCameraIntrinsic:
    intrinsic_matrix = np.loadtxt(path_matrix)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    image_shape = o3d.io.read_image(path_image).get_max_bound()
    width = int(image_shape[0])
    height = int(image_shape[1])
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def extract_intrinsic_projection_parameters(intrinsics: o3d.camera.PinholeCameraIntrinsic) -> Tuple[float, float, float, float]:
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]
    return fx, fy, cx, cy
