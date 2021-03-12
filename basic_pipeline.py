#!/usr/bin/python3

# stdlib
import os
import sys
import re
import enum

# 3rd-party
import numpy as np
import open3d as o3d
import torch

# local
import options as opt
import nnrt
from pipeline import camera
from model import dataset
from utils import image_proc
from model.model import DeformNet
from pipeline import graph

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


def count_frames(directory: str, file_regex: re.Pattern) -> int:
    """
    Count frames of a sequence in the specified directory.
    Only files whose filenames match the file_regex are assumed to be frame images in a single image stream
    within the sequence.
    :param directory: the directory to check
    :param file_regex: regex for files belonging to a single image stream in the sequence
    :return: count of the specified files
    """
    count = 0
    for filename in os.listdir(directory):
        if file_regex.match(filename):
            count += 1

    return count


def reset(visualizer: o3d.pybind.visualization.VisualizerWithKeyCallback) -> None:
    view_control: o3d.visualization.ViewControl = visualizer.get_view_control()
    view_control.reset_camera_local_rotate()


class DatasetPreset(enum.Enum):
    MINION = 1
    RED_SHORTS = 2


def main() -> None:
    #####################################################################################################
    # Options
    #####################################################################################################

    # We will overwrite the default value in options.py / settings.py
    opt.use_mask = True

    #####################################################################################################
    # Load model
    #####################################################################################################

    saved_model = opt.saved_model

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)

    # Construct model
    model = DeformNet().cuda()

    if "chairs_things" in saved_model:
        model.flow_net.load_state_dict(pretrained_dict)
    else:
        if opt.model_module_to_load == "full_model":
            # Load completely model
            model.load_state_dict(pretrained_dict)
        elif opt.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            print(opt.model_module_to_load, "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()

    model.eval()

    #####################################################################################################
    # === device configuration ===
    #####################################################################################################

    device = o3d.core.Device('cuda:0')

    dataset_to_use = DatasetPreset.RED_SHORTS

    #####################################################################################################
    # === dataset parameters ===
    #####################################################################################################

    color_image_filename_mask = None
    depth_image_filename_mask = None
    depth_intrinsics_path = None
    frame_count = 0

    if dataset_to_use is DatasetPreset.MINION:
        # == Minion dataset from VolumeDeform
        frames_directory = "/mnt/Data/Reconstruction/real_data/minion/data/"
        depth_intrinsics_path = "/mnt/Data/Reconstruction/real_data/minion/data/depthIntrinsics.txt"
        color_image_filename_mask = frames_directory + "frame-{:06d}.color.png"
        depth_image_filename_mask = frames_directory + "frame-{:06d}.depth.png"
        frame_count = count_frames(frames_directory, re.compile(r'frame-\d{6}\.depth\.png'))
    elif dataset_to_use is DatasetPreset.RED_SHORTS:
        # == val/seq014 dataset from DeepDeform (red shorts)
        frames_directory = "/mnt/Data/Reconstruction/real_data/deepdeform/val/seq014/"
        depth_intrinsics_path = "/mnt/Data/Reconstruction/real_data/deepdeform/val/seq014/intrinsics.txt"
        color_image_filename_mask = frames_directory + "color/{:06d}.jpg"
        depth_image_filename_mask = frames_directory + "depth/{:06d}.png"
        frame_count = count_frames(os.path.join(frames_directory, "depth"), re.compile(r'\d{6}\.png'))

    first_depth_image_path = depth_image_filename_mask.format(0)
    intrinsics_open3d_cpu = camera.load_intrinsics_from_text_4x4_matrix_and_first_image(depth_intrinsics_path, first_depth_image_path)
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics_open3d_cpu)
    intrinsics_dict = camera.intrinsic_projection_parameters_as_dict(intrinsics_open3d_cpu)
    camera.print_intrinsic_projection_parameters(intrinsics_open3d_cpu)

    intrinsics_open3d_gpu = o3d.core.Tensor(intrinsics_open3d_cpu.intrinsic_matrix, o3d.core.Dtype.Float32, device)

    extrinsics = np.array([[1.0, 0.0, 0.0, 0],
                           [0.0, 1.0, 0.0, 0],
                           [0.0, 0.0, 1.0, 0],
                           [0.0, 0.0, 0.0, 1.0]])
    extrinsics_gpu = o3d.core.Tensor(extrinsics, o3d.core.Dtype.Float32, device)

    #####################################################################################################
    # === volume representation parameters ===
    #####################################################################################################

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

    deformation_graph: graph.DeformationGraph|None = None

    for frame_index in range(0, frame_count):
        #####################################################################################################
        # grab images, transfer to GPU versions for Open3D
        #####################################################################################################
        print("Processing frame:", frame_index)
        depth_image_path = depth_image_filename_mask.format(frame_index)
        print(depth_image_path)
        depth_image_open3d_legacy = o3d.io.read_image(depth_image_path)
        depth_image = np.array(depth_image_open3d_legacy)
        color_image_path = color_image_filename_mask.format(frame_index)
        print(color_image_path)
        color_image_open3d_legacy = o3d.io.read_image(color_image_path)
        color_image = np.array(color_image_open3d_legacy)

        if frame_index == 0:
            depth_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(depth_image_open3d_legacy, device=device)
            color_image_gpu: o3d.t.geometry.Image = o3d.t.geometry.Image.from_legacy_image(color_image_open3d_legacy, device=device)
            volume.integrate(depth_image_gpu, color_image_gpu, intrinsics_open3d_gpu, extrinsics_gpu, 1000.0, 3.0)
            mesh: o3d.geometry.TriangleMesh = volume.extract_surface_mesh(0).to_legacy_triangle_mesh()
            mesh.compute_vertex_normals()

            # === Construct initial deformation graph
            deformation_graph = graph.build_deformation_graph_from_mesh(mesh, 0.05)
            # uncomment to construct from image instead
            # deformation_graph = graph.build_deformation_graph_from_depth_image(point_image, intrinsics, 16)

            # uncomment to visualize KNN graph with background image (no mesh) (deformation graph)
            # graph.draw_deformation_graph(deformation_graph, color_image_open3d_legacy)

            # uncomment to visualize isosurface + KNN graph
            # knn_graph = graph.knn_graph_to_line_set(canonical_node_positions, edges, clusters)
            # o3d.visualization.draw_geometries([mesh, knn_graph],
            #                                   front=[0, 0, -1],
            #                                   lookat=[0, 0, 1.5],
            #                                   up=[0, -1.0, 0],
            #                                   zoom=0.7)
        # else: # __DEBUG
        elif frame_index == 1:
            # TODO: replace source with deformed isosurface render
            source, _, cropper = dataset.DeformDataset.prepare_pytorch_input(
                color_image, depth_image, intrinsics_dict,
                opt.image_height, opt.image_width
            )
            target, target_boundary_mask, _ = dataset.DeformDataset.prepare_pytorch_input(
                color_image, depth_image, intrinsics_dict,
                opt.image_height, opt.image_width,
                cropper=cropper,
                max_boundary_dist=opt.max_boundary_dist,
                compute_boundary_mask=True
            )

            # TODO: this will all need to be replaced somehow by using the isosurface vertices / render process...
            #  before this, it can be optimized -- point_image is already computed inside prepare_pytorch_input,
            #  that can be further split into subroutines that isolate the depth projection
            point_image = nnrt.backproject_depth_ushort(depth_image, fx, fy, cx, cy, 1000.0)

            # TODO: Weights & anchors should probably be computed for isosurface points instead of pixels (?)
            # pixel_anchors = np.zeros(1, dtype=np.int32)
            # pixel_weights = np.zeros(1, dtype=np.float32)
            #
            # nnrt.compute_pixel_anchors_geodesic(
            #     deformation_graph.live_node_positions, deformation_graph.edges,
            #     point_image, 1, 0.05, pixel_anchors, pixel_weights
            # )
            pixel_anchors, pixel_weights = nnrt.compute_pixel_anchors_geodesic(
                deformation_graph.live_node_positions, deformation_graph.edges,
                point_image, 1, 0.05
            )

            pixel_anchors = cropper(pixel_anchors)
            pixel_weights = cropper(pixel_weights)

            fx, fy, cx, cy = image_proc.modify_intrinsics_due_to_cropping(
                intrinsics_dict['fx'], intrinsics_dict['fy'], intrinsics_dict['cx'], intrinsics_dict['cy'],
                opt.image_height, opt.image_width, original_h=cropper.h, original_w=cropper.w
            )
            adjusted_intrinsics_numpy = np.array([fx, fy, cx, cy], dtype=np.float32)

            # TODO: not sure how this is used yet
            num_nodes = np.array(deformation_graph.live_node_positions.shape[0], dtype=np.int64)

            source_cuda = torch.from_numpy(source).cuda().unsqueeze(0)
            target_cuda = torch.from_numpy(target).cuda().unsqueeze(0)
            target_boundary_mask_cuda = torch.from_numpy(target_boundary_mask).cuda().unsqueeze(0)
            graph_nodes_cuda = torch.from_numpy(deformation_graph.live_node_positions).cuda().unsqueeze(0)
            graph_edges_cuda = torch.from_numpy(deformation_graph.edges).cuda().unsqueeze(0)
            # TODO: edge weights are for now all "1.0" since it's not clear how they are computed
            #  (and the default option disables them anyway)
            graph_edges_weights_cuda = torch.from_numpy(deformation_graph.edge_weights).cuda().unsqueeze(0)
            graph_clusters_cuda = torch.from_numpy(deformation_graph.clusters.reshape(-1, 1)).cuda().unsqueeze(0)
            pixel_anchors_cuda = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
            pixel_weights_cuda = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
            intrinsics_cuda = torch.from_numpy(adjusted_intrinsics_numpy).cuda().unsqueeze(0)

            num_nodes_cuda = torch.from_numpy(num_nodes).cuda().unsqueeze(0)



            with torch.no_grad():
                model_data = model(
                    source_cuda, target_cuda,
                    graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                    pixel_anchors_cuda, pixel_weights_cuda,
                    num_nodes_cuda, intrinsics_cuda,
                    evaluate=True, split="test"
                )

                # Get some of the results
            rotations_pred = model_data["node_rotations"].view(num_nodes, 3, 3).cpu().numpy()
            translations_pred = model_data["node_translations"].view(num_nodes, 3).cpu().numpy()

            # mask_pred = model_data["mask_pred"]
            # assert mask_pred is not None, "Make sure use_mask=True in options.py"
            # mask_pred = mask_pred.view(-1, opt.image_height, opt.image_width).cpu().numpy()
        else:  # __DEBUG
            break

        # __DEBUG these will be unneeded once the isosurface deformation & rendering are implemented
        previous_depth_image = depth_image
        previous_color_image = color_image

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
