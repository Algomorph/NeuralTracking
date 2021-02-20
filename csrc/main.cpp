#include <torch/extension.h>

#include "cpu/image_proc.h"
#include "cpu/graph_proc.h"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME nnrt
#endif

#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

using namespace pybind11::literals;


int add(int i, int j) {
	return i + j;
}

// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

	m.def("compute_augmented_flow_from_rotation",
	      &image_proc::compute_augmented_flow_from_rotation,
	      "flow_image_rot_sa2so"_a, "flow_image_so2to"_a, "flow_image_rot_to2ta"_a, "height"_a, "width"_a,
	      "Computes an optical flow image that reflects the augmentation applied to the source and target images.");

	//TODO: define what the hell this is and what it's supposed to do or remove if it's garbage
	m.def("count_tp1", &image_proc::count_tp1, "");
	m.def("count_tp2", &image_proc::count_tp2, "");
	m.def("count_tp3", &image_proc::count_tp3, "");

	//TODO: define what the hell this is and what it's supposed to do or remove if it's garbage
	m.def("extend3", &image_proc::extend3, "");


	//[image] --> [ordered point cloud (point image)] and [point image] --> [mesh (vertex positions, vertex colors, face vertex indices)]

	m.def("backproject_depth_ushort", &image_proc::backproject_depth_ushort, "image_in"_a, "point_image_out"_a,
	      "fx"_a, "fy"_a, "cx"_a, "cy"_a, "normalizer"_a, "Back-project depth image into 3D points");

	m.def("backproject_depth_float", &image_proc::backproject_depth_float, "image_in"_a, "point_image_out"_a,
	      "fx"_a, "fy"_a, "cx"_a, "cy"_a, "Back-project depth image into 3D points");

	m.def("compute_mesh_from_depth", &image_proc::compute_mesh_from_depth, "point_image"_a, "max_triangle_edge_distance"_a,
	      "vertex_position"_a, "face_indices"_a, "Computes a mesh using back-projected points and pixel connectivity");

	m.def("compute_mesh_from_depth_and_color", &image_proc::compute_mesh_from_depth_and_color, "point_image"_a, "color_image"_a,
	      "max_triangle_edge_distance"_a, "vertex_positions"_a, "vertex_colors"_a, "face_indices"_a,
	      "Computes a mesh using back-projected points and pixel connectivity. Additionally, extracts colors for each vertex");

	// image filtering
	m.def("filter_depth", py::overload_cast<py::array_t<unsigned short>&, py::array_t<unsigned short>&, int>(&image_proc::filter_depth),
	      "depth_image_in"_a, "depth_image_out"_a, "radius"_a,
	      "Runs a median filter on input depth image, outputs to provided output image. Does not modify the original.");
	m.def("filter_depth", py::overload_cast<py::array_t<unsigned short>&, int>(&image_proc::filter_depth),
	      "depth_image_in"_a, "radius"_a,
	      "Runs a median filter on provided depth image and outputs a new image with the result. Does not modify the original.");

	// warping via bilinear/trilinear interpolation

	m.def("warp_flow", &image_proc::warp_flow, "image"_a, "flow"_a, "mask"_a,
	      "Warps image (RGB) using provided 2D flow inside masked region using bilinear interpolation.\n"
	      "We assume:\n    image shape: (3, h, w)\n    flow shape: (2, h, w)\n    mask shape: (2, h, w)");

	m.def("warp_rigid", &image_proc::warp_rigid, "rgbxyz_image"_a, "rotation"_a, "translation"_a, "fx"_a, "fy"_a, "cx"_a, "cy"_a,
	      "Warps image (concatenated RGB + XYZ ordered point cloud, i.e. 6 channels) using provided depth map and rigid pose. Assumed rotation shape: (9), translation shape: (2).");

	m.def("warp_3d", &image_proc::warp_3d, "Warps image inside masked region using provided warped point cloud and trilinear interpolation).",
	      "We assume:\n    image shape: (6, h, w)\n    flow shape: (3, h, w)\n    mask shape: (h, w)");

	// procedures for deformation graph node sampling from a point-cloud-based mesh

	m.def("get_vertex_erosion_mask", &graph_proc::get_vertex_erosion_mask, "vertex_positions"_a, "face_indices"_a, "iteration_count"_a,
	      "min_neighbors"_a,
	      "Compile a vertex mask that can be used to erode the provided mesh (remove the vertices at surface discontinuities, leave only non-eroded vertices)");

	m.def("sample_nodes", &graph_proc::sample_nodes, "vertex_positions_in"_a, "vertex_erosion_mask_in"_a, "node_positions_out"_a,
	      "node_indices_out"_a, "node_coverage"_a, "use_only_non_eroded_indices"_a, "Samples graph nodes that cover given vertices.");

	// procedures for deformation graph processing

	m.def("compute_edges_geodesic", &graph_proc::compute_edges_geodesic, "vertex_positions"_a, "face_indices"_a, "node_indices"_a,
	      "max_neighbor_count"_a, "max_influence"_a,
	      "Computes geodesic edges between given graph nodes (subsampled vertices on given mesh)\n"
	      " using a priority-queue-based implementation of Djikstra's algorithm.\n"
	      "Output is returned as an array of dimensions (node_count, max_neighbor_count), where row index represents a source node index and\n"
	      " the row's entries, if >=0, represent destination node indices, ordered by geodesic distance between source and destination. \n"
	      "If the source node has no geodesic neighbors, the nearest euclidean neighbor node's index will appear as the first and only entry in the node.");

	m.def("compute_edges_euclidean", &graph_proc::compute_edges_euclidean, "node_positions"_a, "max_neighbor_count"_a,
	      "Computes Euclidean edges between given graph nodes.\n"
	      "The output is returned as an array of (node_count, max_neighbor_count), where row index represents a source node index and\n"
	      "the row's entries, if >=0, represent destination node indices, ordered by euclidean distance between source and destination.");

	m.def("compute_pixel_anchors_geodesic", &graph_proc::compute_pixel_anchors_geodesic,
	      "graph_nodes"_a, "graph_edges"_a, "point_image"_a, "neighborhood_depth"_a, "node_coverage"_a,
	      "pixel_anchors"_a, "pixel_weights"_a,
	      "Computes anchor ids and skinning weights for every pixel using graph connectivity.\n"
	      "Output pixel anchors array (height, width, K) contains indices of K graph nodes that \n"
	      "influence the corresponding point in the point_image. K is currently hard-coded to " STRINGIFY(GRAPH_K) ". \n"
	      "\n The output pixel weights array of the same dimensions contains the corresponding node weights based "
	      "\n on distance d from point to node: weight = e^( -d^(2) / (2*node_coverage^(2)) ).");

	m.def("compute_pixel_anchors_euclidean", &graph_proc::compute_pixel_anchors_euclidean,
	      "graph_nodes"_a, "point_image"_a, "node_coverage"_a, "pixel_anchors"_a, "pixel_weights"_a,
	      "Computes anchor ids and skinning weights for every pixel using Euclidean distances.\n"
	      "Output pixel anchors array (height, width, K) contains indices of K graph nodes that \n"
	      "influence the corresponding point in the point_image. K is currently hard-coded to " STRINGIFY(GRAPH_K) ". \n"
	      "\n The output pixel weights array of the same dimensions contains the corresponding node weights based "
	      "\n on distance d from point to node: weight = e^( -d^(2) / (2*node_coverage^(2)) ).");

	m.def("construct_regular_graph", &graph_proc::construct_regular_graph, "Samples graph uniformly in pixel space, and computes pixel anchors");

	m.def("add", &add, "i"_a, "j"_a, R"pbdoc(
    Add two numbers
    Some other explanation about the add function.
)pbdoc");
}