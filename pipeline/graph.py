import numpy as np
import open3d as o3d
import nnrt
from typing import Tuple
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib import cm
from utils.visualization.primitive import make_z_aligned_image_plane
from pipeline import camera


def knn_edges_column_to_lines(node_edges: np.ndarray, neighbor_index) -> np.ndarray:
    lines = []
    for node_index in range(0, len(node_edges)):
        neighbor_node_index = node_edges[node_index, neighbor_index]
        if neighbor_node_index != -1:
            lines.append((node_index, neighbor_node_index))
    return np.array(lines)


def knn_graph_to_line_set(node_positions: np.ndarray, node_edges: np.ndarray, clusters: np.ndarray = None) -> o3d.geometry.LineSet:
    first_connections = node_edges[:, :1].copy()
    node_indices = np.arange(0, node_positions.shape[0]).reshape(-1, 1)
    lines_0 = np.concatenate((node_indices.copy(), first_connections), axis=1)
    neighbor_line_source_sets = [lines_0]
    for neighbor_index in range(1, node_edges.shape[1]):
        neighbor_line_set = knn_edges_column_to_lines(node_edges, neighbor_index)
        neighbor_line_source_sets.append(neighbor_line_set)

    lines = np.concatenate(neighbor_line_source_sets, axis=0)

    if clusters is None:
        colors = [[1, 0, 0] for _ in range(len(lines))]
    else:
        colors = []
        for line in lines:
            color = cm.tab10(clusters[line[0]])
            colors.append([color[0], color[1], color[2]])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(node_positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_knn_graph(node_positions: np.ndarray, node_edges: np.ndarray, clusters: np.ndarray = None,
                   background_image: o3d.geometry.Image = None) -> None:
    line_set = knn_graph_to_line_set(node_positions, node_edges, clusters)

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


def find_knn_graph_connected_components(knn_edges: np.ndarray) -> np.ndarray:
    source_node_index = 0
    edge_matrix = lil_matrix((knn_edges.shape[0], knn_edges.shape[0]), dtype=int)
    for destination_nodes in knn_edges:
        for destination_node_index in destination_nodes:
            if destination_node_index != -1:
                edge_matrix[source_node_index, destination_node_index] = 1
        source_node_index += 1

    edge_matrix_compressed = csr_matrix(edge_matrix)
    _, labels = connected_components(csgraph=edge_matrix_compressed, directed=False, return_labels=True)
    return labels


class DeformationGraph:
    def __init__(self, canonical_node_positions: np.ndarray, edges: np.ndarray, clusters: np.ndarray):
        self._canonical_node_positions = canonical_node_positions
        self._edges = edges
        self._clusters = clusters
        self._live_node_positions = canonical_node_positions.copy()

    def get_extent_canonical(self):
        return self._canonical_node_positions.max(axis=0), self._canonical_node_positions.min(axis=0)

    def as_line_set_canonical(self):
        return knn_graph_to_line_set(self._canonical_node_positions, self._edges, self._clusters)

    def as_line_set_live(self):
        return knn_graph_to_line_set(self._live_node_positions, self._edges, self._clusters)


def build_deformation_graph_from_mesh(mesh: o3d.geometry.TriangleMesh) -> DeformationGraph:
    vertex_positions = np.array(mesh.vertices)
    triangle_vertex_indices = np.array(mesh.triangles)

    # === Build deformation graph ===

    erosion_mask = nnrt.get_vertex_erosion_mask(vertex_positions, triangle_vertex_indices, 1, 3)
    node_positions, node_vertex_indices = nnrt.sample_nodes(vertex_positions, erosion_mask, 0.05, True)
    edges = nnrt.compute_edges_geodesic(vertex_positions, triangle_vertex_indices, node_vertex_indices, 4, 0.5)
    clusters = find_knn_graph_connected_components(knn_edges=edges)
    return DeformationGraph(node_positions, edges, clusters)


def build_deformation_graph_from_depth_image(depth_image: o3d.geometry.Image, intrinsics: o3d.camera.PinholeCameraIntrinsic) -> DeformationGraph:
    fx, fy, cx, cy = camera.extract_intrinsic_projection_parameters(intrinsics)

    depth_image_numpy = np.array(depth_image)
    point_image = nnrt.backproject_depth_ushort(depth_image_numpy, fx, fy, cx, cy, 1000.0)
    # print(point_image.shape)
    # np.savetxt("point_image.txt", point_image.swapaxes(0, 2).reshape(-1, 3), delimiter=",")

    node_positions, edges, pixel_anchors, pixel_weights = \
        nnrt.construct_regular_graph(point_image,
                                     depth_image_numpy.shape[1] // 2,
                                     depth_image_numpy.shape[0] // 2,
                                     2.0, 0.05, 3.0)
    clusters = find_knn_graph_connected_components(knn_edges=edges)
    return DeformationGraph(node_positions, edges, clusters)


def draw_deformation_graph(deformation_graph: DeformationGraph,
                           background_image: o3d.geometry.Image = None) -> None:
    line_set = deformation_graph.as_line_set_canonical()

    extent_max, extent_min = deformation_graph.get_extent_canonical()
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
