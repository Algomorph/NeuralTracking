#!/usr/bin/python3

import os
import sys
import cv2
import numpy as np
import open3d as o3d
import nnrt
import re

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


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


def count_frames(directory: str, file_regex: re.Pattern) -> int:
    count = 0
    for filename in os.listdir(directory):
        if file_regex.match(filename):
            count += 1

    return count


def set_up(visualizer: o3d.pybind.visualization.VisualizerWithKeyCallback) -> None:
    view_control: o3d.visualization.ViewControl = visualizer.get_view_control()
    view_control.set_up(np.array([0, 0, 1]))


def reset(visualizer: o3d.pybind.visualization.VisualizerWithKeyCallback) -> None:
    view_control: o3d.visualization.ViewControl = visualizer.get_view_control()
    view_control.reset_camera_local_rotate()


def knn_edges_column_to_lines(node_edges: np.ndarray, neighbor_index) -> np.ndarray:
    lines = []
    for node_index in range(0, len(node_edges)):
        neighbor_node_index = node_edges[node_index, neighbor_index]
        if neighbor_node_index != -1:
            lines.append((node_index, neighbor_node_index))
    return np.array(lines)


def make_z_aligned_image_plane(min_pt, max_pt, z, image):
    plane_vertices = [
        [min_pt[0], min_pt[1], z],
        [max_pt[0], min_pt[1], z],
        [max_pt[0], max_pt[1], z],
        [min_pt[0], max_pt[1], z]
    ]
    plane_triangles = [[2, 1, 0],
                       [0, 3, 2]]

    plane_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(plane_vertices), o3d.utility.Vector3iVector(plane_triangles))
    plane_mesh.compute_vertex_normals()

    plane_texture_coordinates = [
        (1, 1), (1, 0), (0, 0),
        (0, 0), (0, 1), (1, 1)
    ]

    plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(plane_texture_coordinates)
    plane_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    plane_mesh.textures = [image]
    return plane_mesh


def draw_knn_graph(node_positions: np.ndarray, node_edges: np.ndarray, background_image: o3d.geometry.Image = None) -> None:
    first_connections = node_edges[:, :1].copy()
    node_indices = np.arange(0, node_positions.shape[0]).reshape(-1, 1)
    lines_0 = np.concatenate((node_indices.copy(), first_connections), axis=1)
    lines_1 = knn_edges_column_to_lines(node_edges, 1)
    lines_2 = knn_edges_column_to_lines(node_edges, 2)
    lines_3 = knn_edges_column_to_lines(node_edges, 3)

    lines = np.concatenate((lines_0, lines_1, lines_2, lines_3), axis=0)

    colors = [[1, 0, 0] for i in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(node_positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    extent_max = node_positions.max(axis=0)
    extent_min = node_positions.min(axis=0)
    plane_z = extent_max[2]

    if background_image is not None:
        plane_mesh = make_z_aligned_image_plane((extent_min[0], extent_min[1]), (extent_max[0], extent_max[1]), plane_z, background_image)
        geometries = [plane_mesh, line_set]
    else:
        geometries = [line_set]

    o3d.visualization.draw_geometries(geometries,
                                      front=[0, 0, -1],
                                      lookat=[0, 0, 1.5],
                                      up=[0, -1.0, 0],
                                      zoom=0.7)


def main():
    # === device configuration ===
    device = o3d.core.Device('cuda:0')

    # === dataset parameters ===
    frames_directory = "/mnt/Data/Reconstruction/real_data/minion/data/"
    depth_intrinsics_path = "/mnt/Data/Reconstruction/real_data/minion/data/depthIntrinsics.txt"
    color_image_filename_mask = frames_directory + "frame-{:06d}.color.png"
    depth_image_filename_mask = frames_directory + "frame-{:06d}.depth.png"
    frame_count = count_frames(frames_directory, re.compile(r'frame-\d{6}\.depth\.png'))
    first_depth_image_path = depth_image_filename_mask.format(0)
    intrinsics = load_intrinsics_from_text_4x4_matrix_and_first_image(depth_intrinsics_path, first_depth_image_path)
    intrinsics_gpu = o3d.core.Tensor(intrinsics.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    extrinsics = np.array([[1.0, 0.0, 0.0, 0],
                           [0.0, 1.0, 0.0, 0],
                           [0.0, 0.0, 1.0, 0],
                           [0.0, 0.0, 0.0, 1.0]])
    extrinsics_gpu = o3d.core.Tensor(extrinsics, o3d.core.Dtype.Float32, device)

    # === volume representation parameters ===
    voxel_size = 0.008  # voxel resolution in meters
    sdf_trunc = 0.04  # truncation distance in meters
    block_resolution = 16  # 16^3 voxel blocks
    initial_block_count = 1000  # initially allocated number of voxel blocks

    volume = o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        block_resolution=block_resolution,
        block_count=initial_block_count,
        device=device)

    previous_color_image = None

    for frame_index in range(0, frame_count):
        print("Processing frame:", frame_index)
        depth_image_path = depth_image_filename_mask.format(frame_index)
        depth_image = o3d.io.read_image(depth_image_path)
        depth_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(depth_image, device=device)

        color_image_path = color_image_filename_mask.format(frame_index)
        color_image = o3d.io.read_image(color_image_path)
        color_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(color_image, device=device)

        if frame_index == 0:
            volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_gpu, extrinsics_gpu, 1000.0, 3.0)
        # else: # __DEBUG
        elif frame_index == 1:
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()
            vertex_positions = np.array(mesh.vertices)
            triangle_vertex_indices = np.array(mesh.triangles)

            erosion_mask = nnrt.get_vertex_erosion_mask(vertex_positions, triangle_vertex_indices, 1, 3)
            node_positions, node_vertex_indices = nnrt.sample_nodes(vertex_positions, erosion_mask, 0.05, False)
            node_edges = nnrt.compute_edges_geodesic(vertex_positions, triangle_vertex_indices, node_vertex_indices, 4, 0.5)
            # __DEBUG
            draw_knn_graph(node_positions, node_edges, previous_color_image)
        else:
            break

        previous_color_image = color_image

    # filtered_depth = nnrt.filter_depth(depth1, 2)

    # point_cloud = nnrt.backproject_depth_ushort(filtered_depth1, fx, fy, cx, cy, 1000.0)

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
