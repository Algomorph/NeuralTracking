#pragma once

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#define GRAPH_K 4

namespace graph_proc {

    /**
	 * Erode mesh
	 */
	py::array_t<bool> get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices, int iteration_count, int min_neighbors);
	
    /**
	 * Samples nodes that cover all vertex positions with given node coverage.
	 * Nodes are sampled from vertices, resulting node vertex indices are returned.
	 */
	int sample_nodes(
		    const py::array_t<float>& vertex_positions_in, const py::array_t<bool>& vertex_erosion_mask_in,
		    py::array_t<float>& node_positions_out, py::array_t<int>& node_indices_out,
		    float node_coverage, const bool use_only_non_eroded_indices);


	/**
	 * Computes the graph edges between nodes, connecting nearest nodes using geodesic
	 * distances.
	 */
	py::array_t<int> compute_edges_geodesic(
		const py::array_t<float>& vertexPositions,
		const py::array_t<int>& faceIndices, 
		const py::array_t<int>& nodeIndices, 
		int nMaxNeighbors, float maxInfluence
	);


	/**
	 * Computes the graph edges between nodes, connecting nearest nodes using Euclidean
	 * distances.
	 */
	py::array_t<int> compute_edges_euclidean(const py::array_t<float>& nodePositions, int nMaxNeighbors);
	
	/**
	 * For each input pixel it computes 4 nearest anchors, following graph edges. 
	 * It also compute skinning weights for every pixel. 
	 */ 
	void compute_pixel_anchors_geodesic(
        const py::array_t<float>& graphNodes, 
        const py::array_t<int>& graphEdges,
        const py::array_t<float>& pointImage,
        int neighborhoodDepth,
        float nodeCoverage,
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    );


	/**
	 * For each input pixel it computes 4 nearest anchors, using Euclidean distances. 
	 * It also compute skinning weights for every pixel. 
	 */ 
	void compute_pixel_anchors_euclidean(
        const py::array_t<float>& graphNodes, 
        const py::array_t<float>& pointImage,
        float nodeCoverage,
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    );

    /**
	 * It samples graph regularly from the image, using pixel-wise connectivity
     * (connecting each pixel with at most 8 neighbors).
	 */ 
	void construct_regular_graph(
        const py::array_t<float>& pointImage,
        int xNodes, int yNodes,
        float edgeThreshold,
        float maxPointToNodeDistance,
        float maxDepth,
        py::array_t<float>& graphNodes, 
        py::array_t<int>& graphEdges, 
        py::array_t<int>& pixelAnchors, 
        py::array_t<float>& pixelWeights
    );

} // namespace graph_proc