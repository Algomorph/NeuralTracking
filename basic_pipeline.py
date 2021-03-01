#!/usr/bin/python3

import os
import sys
import cv2
import numpy as np
import open3d as o3d
import nnrt
import re
import torch

from pipeline import graph
from pipeline.camera import load_intrinsics_from_text_4x4_matrix_and_first_image

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


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

        deformation_graph = None

        if frame_index == 0:
            volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_gpu, extrinsics_gpu, 1000.0, 3.0)
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            # deformation_graph = graph.build_deformation_graph_from_mesh(mesh)
            deformation_graph = graph.build_deformation_graph_from_depth_image(depth_image, intrinsics)


            # uncomment to visualize KNN graph with background image (no mesh) (deformation graph)
            graph.draw_deformation_graph(deformation_graph, color_image)

            # uncomment to visualize isosurface + KNN graph
            # knn_graph = graph.knn_graph_to_line_set(canonical_node_positions, edges, clusters)
            # o3d.visualization.draw_geometries([mesh, knn_graph],
            #                                   front=[0, 0, -1],
            #                                   lookat=[0, 0, 1.5],
            #                                   up=[0, -1.0, 0],
            #                                   zoom=0.7)
        # else: # __DEBUG
        elif frame_index == 1:
            pass


        else:
            break

        previous_color_image = color_image

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
