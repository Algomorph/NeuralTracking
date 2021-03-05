#pragma once


#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define GRAPH_K 4

namespace py = pybind11;

namespace graph_proc {


py::array_t<bool>
get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int iteration_count, int min_neighbors);

/**
 * Samples canonical_node_positions that cover all vertex positions with given node coverage.
 * Nodes are sampled from vertices, resulting node vertex indices are returned.
 */
py::tuple sample_nodes(
		const py::array_t<float>& vertex_positions_in, const py::array_t<bool>& vertex_erosion_mask_in,
		float node_coverage, const bool use_only_non_eroded_indices);


/**
 * Computes the graph edges between canonical_node_positions, connecting nearest canonical_node_positions using geodesic
 * distances.
 */
py::array_t<int> compute_edges_geodesic(
		const py::array_t<float>& vertex_positions,
		const py::array_t<int>& face_indices,
		const py::array_t<int>& node_indices,
		int max_neighbor_count, float max_influence
);


/**
 * Computes the graph edges between canonical_node_positions, connecting nearest canonical_node_positions using Euclidean
 * distances.
 */
py::array_t<int> compute_edges_euclidean(const py::array_t<float>& node_positions, int max_neighbor_count);

/**
 * Compute four nearest anchors and their weights, following graph edges,
 * for each input pixel.
 */
void compute_pixel_anchors_geodesic(
		const py::array_t<float>& graph_nodes,
		const py::array_t<int>& graph_edges,
		const py::array_t<float>& point_image,
		int neighborhood_depth,
		float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
);

py::tuple compute_pixel_anchors_geodesic(
		const py::array_t<float>& graph_nodes,
		const py::array_t<int>& graph_edges,
		const py::array_t<float>& point_image,
		int neighborhood_depth,
		float node_coverage
);


/**
 * For each input pixel it computes 4 nearest anchors, using Euclidean distances.
 * It also compute skinning weights for every pixel.
 */
void compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
);


void construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth,
		py::array_t<float>& graph_nodes,
		py::array_t<int>& graph_edges,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
);

py::tuple construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth
);

} // namespace graph_proc