#include "cpu/graph_proc.h"

#include <set>
#include <vector>
#include <numeric> //std::iota
#include <algorithm>
#include <random>
#include <queue>

#include <Eigen/Dense>

using std::vector;

namespace graph_proc {

/**
 * \brief Compile a vertex mask that can be used to erode the provided mesh (iteratively mask out vertices at surface discontinuities, leave only non-eroded vertices)
 * \param vertex_positions array with vertex positions of dimentions (N, 3)
 * \param face_indices
 * \param iteration_count
 * \param min_neighbors minimum number of neighbors considered for erosion
 * \return (N, 1) array containing the erosion mask, such that, if applied to the original vertex postions array, would leave only non-eroded vertices
 */
py::array_t<bool> get_vertex_erosion_mask(const py::array_t<float>& vertex_positions, const py::array_t<int>& face_indices,
                                          int iteration_count, int min_neighbors) {
	int vertex_count = vertex_positions.shape(0);
	int face_count = face_indices.shape(0);

	// Init output
	py::array_t<bool> non_eroded_vertices = py::array_t<bool>({vertex_count, 1});
	std::vector<bool> non_eroded_vertices_vec(vertex_count, false);

	// Init list of eroded face indices with original list
	std::vector<Eigen::Vector3i> non_erorded_face_indices_vec;
	non_erorded_face_indices_vec.reserve(face_count);
	for (int face_idx = 0; face_idx < face_count; ++face_idx) {
		Eigen::Vector3i face(*face_indices.data(face_idx, 0), *face_indices.data(face_idx, 1), *face_indices.data(face_idx, 2));
		non_erorded_face_indices_vec.push_back(face);
	}

	// Erode mesh for a total of iteration_count
	for (int i = 0; i < iteration_count; i++) {
		face_count = non_erorded_face_indices_vec.size();

		// We compute the number of neighboring vertices for each vertex.
		vector<int> vertex_neighbor_counts(vertex_count, 0);
		for (int face_index = 0; face_index < face_count; face_index++) {
			const auto& face = non_erorded_face_indices_vec[face_index];
			vertex_neighbor_counts[face[0]] += 1;
			vertex_neighbor_counts[face[1]] += 1;
			vertex_neighbor_counts[face[2]] += 1;
		}

		std::vector<Eigen::Vector3i> tmp;
		tmp.reserve(face_count);

		for (int face_index = 0; face_index < face_count; face_index++) {
			const auto& face = non_erorded_face_indices_vec[face_index];
			if (vertex_neighbor_counts[face[0]] >= min_neighbors &&
			    vertex_neighbor_counts[face[1]] >= min_neighbors &&
			    vertex_neighbor_counts[face[2]] >= min_neighbors) {
				tmp.push_back(face);
			}
		}

		// We kill the faces with border vertices.
		non_erorded_face_indices_vec.clear();
		non_erorded_face_indices_vec = std::move(tmp);
	}

	// Mark non isolated vertices as not eroded.
	face_count = non_erorded_face_indices_vec.size();

	for (int i = 0; i < face_count; i++) {
		const auto& face = non_erorded_face_indices_vec[i];
		non_eroded_vertices_vec[face[0]] = true;
		non_eroded_vertices_vec[face[1]] = true;
		non_eroded_vertices_vec[face[2]] = true;
	}

	// Store into python array
	for (int i = 0; i < vertex_count; i++) {
		*non_eroded_vertices.mutable_data(i, 0) = non_eroded_vertices_vec[i];
	}

	return non_eroded_vertices;
}

py::tuple sample_nodes(
		const py::array_t<float>& vertex_positions_in, const py::array_t<bool>& vertex_erosion_mask_in,
		float node_coverage, const bool use_only_non_eroded_indices = true
) {
	// assert(vertexPositions.ndim() == 2);
	// assert(vertexPositions.shape(1) == 3);

	float node_coverage_2 = node_coverage * node_coverage;
	int vertex_count = vertex_positions_in.shape(0);

	// create list of shuffled indices
	std::vector<int> shuffled_vertices(vertex_count);
	std::iota(std::begin(shuffled_vertices), std::end(shuffled_vertices), 0);

	std::default_random_engine re{std::random_device{}()};
	std::shuffle(std::begin(shuffled_vertices), std::end(shuffled_vertices), re);

	struct NodeInformation {
		Eigen::Vector3f vertex_position;
		int vertex_index;
	};

	std::vector<NodeInformation> node_information_vector;

	for (int vertex_index : shuffled_vertices) {
		// for (int vertexIdx = 0; vertexIdx < nVertices; ++vertexIdx) {
		Eigen::Vector3f point(*vertex_positions_in.data(vertex_index, 0),
		                      *vertex_positions_in.data(vertex_index, 1),
		                      *vertex_positions_in.data(vertex_index, 2));

		if (use_only_non_eroded_indices && !(*vertex_erosion_mask_in.data(vertex_index))) {
			continue;
		}

		bool is_node = true;
		for (const auto& node_information : node_information_vector) {
			if ((point - node_information.vertex_position).squaredNorm() <= node_coverage_2) {
				is_node = false;
				break;
			}
		}

		if (is_node) {
			node_information_vector.push_back({point, vertex_index});
		}
	}

	py::array_t<float> node_positions_out({static_cast<ssize_t>(node_information_vector.size()), static_cast<ssize_t>(3)});
	py::array_t<int> node_indices_out({static_cast<ssize_t>(node_information_vector.size()), static_cast<ssize_t>(1)});

	int node_index = 0;
	for (const auto& node_information : node_information_vector) {
		*node_positions_out.mutable_data(node_index, 0) = node_information.vertex_position.x();
		*node_positions_out.mutable_data(node_index, 1) = node_information.vertex_position.y();
		*node_positions_out.mutable_data(node_index, 2) = node_information.vertex_position.z();
		*node_indices_out.mutable_data(node_index, 0) = node_information.vertex_index;
		node_index++;
	}

	return py::make_tuple(node_positions_out, node_indices_out);
}

/**
 * Custom comparison operator for geodesic priority queue.
 */
struct CustomCompare {
	bool operator()(const std::pair<int, float>& left, const std::pair<int, float>& right) {
		return left.second > right.second;
	}
};

py::array_t<int> compute_edges_geodesic(
		const py::array_t<float>& vertex_positions,
		const py::array_t<int>& face_indices,
		const py::array_t<int>& node_indices,
		int max_neighbor_count, float max_influence
) {
	int vertex_count = vertex_positions.shape(0);
	int face_count = face_indices.shape(0);
	int node_count = node_indices.shape(0);

	// Preprocess vertex neighbors.
	vector<std::set<int>> vertex_neighbors(vertex_count);
	for (int face_index = 0; face_index < face_count; face_index++) {
		for (int face_vertex_index = 0; face_vertex_index < 3; face_vertex_index++) {
			int vertex_index = *face_indices.data(face_index, face_vertex_index);

			// mark any vertex in the same triangle face as a neighbor of the other two
			for (int neighbor_face_vertex_index = 0; neighbor_face_vertex_index < 3; neighbor_face_vertex_index++) {
				int neighbor_vertex_index = *face_indices.data(face_index, neighbor_face_vertex_index);

				if (vertex_index == neighbor_vertex_index) continue;
				vertex_neighbors[vertex_index].insert(neighbor_vertex_index);
			}
		}
	}

	// Compute inverse vertex -> node relationship.
	vector<int> map_vertex_to_node(vertex_count, -1);

	for (int node_id = 0; node_id < node_count; node_id++) {
		int vertex_index = *node_indices.data(node_id);
		if (vertex_index >= 0) {
			map_vertex_to_node[vertex_index] = node_id;
		}
	}

	// Construct geodesic edges.
	py::array_t<int> graph_edges = py::array_t<int>({node_count, max_neighbor_count});

	//TODO: parallelize or remove pragma
	// #pragma omp parallel for
	for (int node_id = 0; node_id < node_count; node_id++) {
		std::priority_queue<
				std::pair<int, float>,
				vector<std::pair<int, float>>,
				CustomCompare> next_vertices_with_ids;

		std::set<int> visited_vertices;

		// Add node vertex as the first vertex to be visited.
		int node_vertex_idx = *node_indices.data(node_id);
		if (node_vertex_idx < 0) continue;
		next_vertices_with_ids.push(std::make_pair(node_vertex_idx, 0.f));

		// Traverse all neighbors in the monotonically increasing order.
		vector<int> neighbor_node_ids;
		while (!next_vertices_with_ids.empty()) {
			auto next_vertex = next_vertices_with_ids.top();
			next_vertices_with_ids.pop();

			int next_vertex_index = next_vertex.first;
			float next_vertex_distance = next_vertex.second;

			// We skip the vertex, if it was already visited before.
			if (visited_vertices.find(next_vertex_index) != visited_vertices.end()) continue;

			// We check if the vertex is a node.
			int next_node_index = map_vertex_to_node[next_vertex_index];
			if (next_node_index >= 0 && next_node_index != node_id) {
				neighbor_node_ids.push_back(next_node_index);
				if (neighbor_node_ids.size() >= max_neighbor_count) break;
			}

			// Note down the node-vertex distance.
			// *node_to_vertex_distance.mutable_data(next_node_index, next_vertex_index) = next_vertex_distance;

			// We visit the vertex, and check all his neighbors.
			// We add only vertices under a certain distance.
			visited_vertices.insert(next_vertex_index);
			Eigen::Vector3f next_vertex_pos(*vertex_positions.data(next_vertex_index, 0), *vertex_positions.data(next_vertex_index, 1),
			                                *vertex_positions.data(next_vertex_index, 2));

			const auto& next_neighbors = vertex_neighbors[next_vertex_index];
			for (int neighbor_index : next_neighbors) {
				Eigen::Vector3f neighbor_vertex_pos(*vertex_positions.data(neighbor_index, 0), *vertex_positions.data(neighbor_index, 1),
				                                    *vertex_positions.data(neighbor_index, 2));
				float distance = next_vertex_distance + (next_vertex_pos - neighbor_vertex_pos).norm();

				if (distance <= max_influence) {
					next_vertices_with_ids.push(std::make_pair(neighbor_index, distance));
				}
			}
		}

		// If we don't get any geodesic neighbors, we take one nearest Euclidean neighbor,
		// to have a constrained optimization system at non-rigid tracking.
		if (neighbor_node_ids.empty()) {
			float nearest_squared_distance = std::numeric_limits<float>::infinity();
			int nearest_node_id = -1;

			Eigen::Vector3f nodePos(*vertex_positions.data(node_vertex_idx, 0), *vertex_positions.data(node_vertex_idx, 1),
			                        *vertex_positions.data(node_vertex_idx, 2));

			for (int node_index = 0; node_index < node_count; node_index++) {
				int vertex_index = *node_indices.data(node_index);
				if (node_index != node_id && vertex_index >= 0) {
					Eigen::Vector3f neighbor_position(*vertex_positions.data(vertex_index, 0),
					                                  *vertex_positions.data(vertex_index, 1),
					                                  *vertex_positions.data(vertex_index, 2));

					float squared_distance = (neighbor_position - nodePos).squaredNorm();
					if (squared_distance < nearest_squared_distance) {
						nearest_squared_distance = squared_distance;
						nearest_node_id = node_index;
					}
				}
			}

			if (nearest_node_id >= 0) neighbor_node_ids.push_back(nearest_node_id);
		}

		// Store the nearest neighbors.
		int nearest_neighbor_count = neighbor_node_ids.size();

		for (int i = 0; i < nearest_neighbor_count; i++) {
			*graph_edges.mutable_data(node_id, i) = neighbor_node_ids[i];
		}
		for (int i = nearest_neighbor_count; i < max_neighbor_count; i++) {
			*graph_edges.mutable_data(node_id, i) = -1;
		}
	}

	return graph_edges;
}

py::array_t<int> compute_edges_euclidean(const py::array_t<float>& node_positions, int max_neighbor_count) {
	int node_count = node_positions.shape(0);

	py::array_t<int> graph_edges = py::array_t<int>({node_count, max_neighbor_count});

	// Find nearest Euclidean neighbors for each node.
	for (int source_node_index = 0; source_node_index < node_count; source_node_index++) {
		Eigen::Vector3f node_position(*node_positions.data(source_node_index, 0),
		                              *node_positions.data(source_node_index, 1),
		                              *node_positions.data(source_node_index, 2));

		// Keep only the k nearest Euclidean neighbors.
		std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

		for (int neighbor_index = 0; neighbor_index < node_count; neighbor_index++) {
			if (neighbor_index == source_node_index) continue;

			Eigen::Vector3f neighbor_position(*node_positions.data(neighbor_index, 0), *node_positions.data(neighbor_index, 1),
			                                  *node_positions.data(neighbor_index, 2));

			float squared_distance = (node_position - neighbor_position).squaredNorm();
			bool neighbor_inserted = false;
			for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
				// We insert the element at the first position where its distance is smaller than the other
				// element's distance, which enables us to always keep a sorted list of at most k nearest
				// neighbors.
				if (squared_distance <= it->second) {
					it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(neighbor_index, squared_distance));
					neighbor_inserted = true;
					break;
				}
			}

			if (!neighbor_inserted && nearest_nodes_with_squared_distances.size() < max_neighbor_count) {
				nearest_nodes_with_squared_distances.emplace_back(std::make_pair(neighbor_index, squared_distance));
			}

			// We keep only the list of k nearest elements.
			if (neighbor_inserted && nearest_nodes_with_squared_distances.size() > max_neighbor_count) {
				nearest_nodes_with_squared_distances.pop_back();
			}
		}

		// Store nearest neighbor indices.
		int neighbor_index = 0;
		for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
			int destination_node_index = nearest_nodes_with_squared_distance.first;
			*graph_edges.mutable_data(source_node_index, neighbor_index) = destination_node_index;
			neighbor_index++;
		}

		for (neighbor_index = nearest_nodes_with_squared_distances.size(); neighbor_index < max_neighbor_count; neighbor_index++) {
			*graph_edges.mutable_data(source_node_index, neighbor_index) = -1;
		}
	}

	return graph_edges;
}

static inline float compute_anchor_weight(const Eigen::Vector3f& point_position, const Eigen::Vector3f& node_position, float node_coverage) {
	return std::exp(-(node_position - point_position).squaredNorm() / (2.f * node_coverage * node_coverage));
}

void compute_pixel_anchors_geodesic(
		const py::array_t<float>& graph_nodes,
		const py::array_t<int>& graph_edges,
		const py::array_t<float>& point_image,
		int neighborhood_depth,
		float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
) {
	int node_count = graph_nodes.shape(0);
	int neighbor_count = graph_edges.shape(1);
	int width = point_image.shape(1);
	int height = point_image.shape(0);
	assert(point_image.ndim() == 3);
	assert(point_image.shape(2) == 3);

	// Allocate graph node ids and corresponding skinning weights if needed.
	auto resize_if_needed = [&height, &width](auto& output_array) {
		if (output_array.ndim() != 3 || output_array.shape(0) != height
		    || output_array.shape(1) != width || output_array.shape(2) != GRAPH_K) {
			output_array.resize({height, width, GRAPH_K}, false);
		}
	};
	resize_if_needed(pixel_anchors);
	resize_if_needed(pixel_weights);
	// pixel_anchors.resize({height, width, GRAPH_K}, false);
	// pixel_weights.resize({height, width, GRAPH_K}, false);

	// Initialize with invalid anchors.
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < GRAPH_K; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}
		}
	}

	// Compute anchors for every pixel.
#pragma omp parallel for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Query 3d pixel position.
			Eigen::Vector3f pixel_position(
					*point_image.data(y, x, 0),
					*point_image.data(y, x, 1),
					*point_image.data(y, x, 2)
			);
			if (pixel_position.z() <= 0) continue;

			// Find nearest Euclidean graph node.
			float nearest_euclidean_graph_node_distance = std::numeric_limits<float>::infinity();
			int nearest_node_index = -1;
			for (int node_index = 0; node_index < node_count; node_index++) {
				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0), *graph_nodes.data(node_index, 1), *graph_nodes.data(node_index, 2));
				float dist_squared = (node_position - pixel_position).squaredNorm();
				if (dist_squared < nearest_euclidean_graph_node_distance) {
					nearest_euclidean_graph_node_distance = dist_squared;
					nearest_node_index = node_index;
				}
			}

			// Compute the geodesic neighbor candidates.
			std::set<int> neighbors{nearest_node_index};
			std::set<int> new_neighbors = neighbors;

			for (int i = 0; i < neighborhood_depth; i++) {
				// Get all neighbors of the new neighbors.
				std::set<int> current_neighbors;
				for (auto new_neighbor_index : new_neighbors) {
					for (int new_neighbor_neighbor_index = 0; new_neighbor_neighbor_index < neighbor_count; new_neighbor_neighbor_index++) {
						int neighbor_neighbor_index = *graph_edges.data(new_neighbor_index, new_neighbor_neighbor_index);

						// assume -1 means "no neighbor"
						if (neighbor_neighbor_index >= 0) {
							current_neighbors.insert(neighbor_neighbor_index);
						}
					}
				}

				// Check the newly added neighbors (not present in the neighbors set).
				new_neighbors.clear();
				std::set_difference(
						current_neighbors.begin(), current_neighbors.end(),
						neighbors.begin(), neighbors.end(),
						std::inserter(new_neighbors, new_neighbors.begin())
				);

				// Insert the newly added neighbors.
				neighbors.insert(new_neighbors.begin(), new_neighbors.end());
			}

			// Keep only the k nearest geodesic neighbors.
			std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

			for (auto&& neighbor_index : neighbors) {
				Eigen::Vector3f node_position(*graph_nodes.data(neighbor_index, 0),
				                              *graph_nodes.data(neighbor_index, 1),
				                              *graph_nodes.data(neighbor_index, 2));

				float squared_distance = (node_position - pixel_position).squaredNorm();
				bool nodes_inserted = false;
				for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
					// We insert the element at the first position where its distance is smaller than the other
					// element's distance, which enables us to always keep a sorted list of at most k nearest
					// neighbors.
					if (squared_distance <= it->second) {
						it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(neighbor_index, squared_distance));
						nodes_inserted = true;
						break;
					}
				}

				if (!nodes_inserted && nearest_nodes_with_squared_distances.size() < GRAPH_K) {
					nearest_nodes_with_squared_distances.emplace_back(std::make_pair(neighbor_index, squared_distance));
				}

				// We keep only the list of k nearest elements.
				if (nodes_inserted && nearest_nodes_with_squared_distances.size() > GRAPH_K) {
					nearest_nodes_with_squared_distances.pop_back();
				}
			}

			// Compute skinning weights.
			std::vector<int> nearest_geodesic_node_ids;
			nearest_geodesic_node_ids.reserve(nearest_nodes_with_squared_distances.size());

			std::vector<float> skinning_weights;
			skinning_weights.reserve(nearest_nodes_with_squared_distances.size());

			float weight_sum{0.f};
			for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
				int node_index = nearest_nodes_with_squared_distance.first;

				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0), *graph_nodes.data(node_index, 1), *graph_nodes.data(node_index, 2));
				float weight = compute_anchor_weight(pixel_position, node_position, node_coverage);
				weight_sum += weight;

				nearest_geodesic_node_ids.push_back(node_index);
				skinning_weights.push_back(weight);
			}

			// Normalize the skinning weights.
			int anchor_count = nearest_geodesic_node_ids.size();

			if (weight_sum > 0) {
				for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
					skinning_weights[anchor_index] /= weight_sum;
				}
			} else if (anchor_count > 0) {
				for (int anchor_index = 0; anchor_index < anchor_count; anchor_index++) {
					skinning_weights[anchor_index] = 1.f / static_cast<float>(anchor_count);
				}
			}

			// Store the results.
			for (int i = 0; i < anchor_count; i++) {
				*pixel_anchors.mutable_data(y, x, i) = nearest_geodesic_node_ids[i];
				*pixel_weights.mutable_data(y, x, i) = skinning_weights[i];
			}
		}
	}
}

py::tuple compute_pixel_anchors_geodesic(
		const py::array_t<float>& graph_nodes,
		const py::array_t<int>& graph_edges,
		const py::array_t<float>& point_image,
		int neighborhood_depth,
		float node_coverage) {
	py::array_t<int> pixel_anchors;
	py::array_t<float> pixel_weights;
	compute_pixel_anchors_geodesic(graph_nodes, graph_edges, point_image, neighborhood_depth,
	                               node_coverage, pixel_anchors, pixel_weights);
	return py::make_tuple(pixel_anchors, pixel_weights);
}

//TEST
void compute_pixel_anchors_geodesic_old(
		const py::array_t<float>& graphNodes,
		const py::array_t<int>& graphEdges,
		const py::array_t<float>& pointImage,
		int neighborhoodDepth,
		float nodeCoverage,
		py::array_t<int>& pixelAnchors,
		py::array_t<float>& pixelWeights
) {
	int numNodes = graphNodes.shape(0);
	int numNeighbors = graphEdges.shape(1);
	int width = pointImage.shape(2);
	int height = pointImage.shape(1);
	// int nChannels = pointImage.shape(0);

	// Allocate graph node ids and corresponding skinning weights.
	// Initialize with invalid anchors.
	pixelAnchors.resize({height, width, GRAPH_K}, false);
	pixelWeights.resize({height, width, GRAPH_K}, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < GRAPH_K; k++) {
				*pixelAnchors.mutable_data(y, x, k) = -1;
				*pixelWeights.mutable_data(y, x, k) = 0.f;
			}
		}
	}

	// Compute anchors for every pixel.
#pragma omp parallel for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Query 3d pixel position.
			Eigen::Vector3f pixelPos(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
			if (pixelPos.z() <= 0) continue;

			// Find nearest Euclidean graph node.
			float nearestDist2 = std::numeric_limits<float>::infinity();
			int nearestNodeId = -1;
			for (int nodeId = 0; nodeId < numNodes; nodeId++) {
				Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));
				float dist2 = (nodePos - pixelPos).squaredNorm();
				if (dist2 < nearestDist2) {
					nearestDist2 = dist2;
					nearestNodeId = nodeId;
				}
			}

			// Compute the geodesic neighbor candidates.
			std::set<int> neighbors{nearestNodeId};
			std::set<int> newNeighbors = neighbors;

			for (int i = 0; i < neighborhoodDepth; i++) {
				// Get all neighbors of the new neighbors.
				std::set<int> currentNeighbors;
				for (auto neighborId : newNeighbors) {
					for (int k = 0; k < numNeighbors; k++) {
						int currentNeighborId = *graphEdges.data(neighborId, k);

						if (currentNeighborId >= 0) {
							currentNeighbors.insert(currentNeighborId);
						}
					}
				}

				// Check the newly added neighbors (not present in the neighbors set).
				newNeighbors.clear();
				std::set_difference(
						currentNeighbors.begin(), currentNeighbors.end(),
						neighbors.begin(), neighbors.end(),
						std::inserter(newNeighbors, newNeighbors.begin())
				);

				// Insert the newly added neighbors.
				neighbors.insert(newNeighbors.begin(), newNeighbors.end());
			}

			// Keep only the k nearest geodesic neighbors.
			std::list<std::pair<int, float>> nearestNodesWithSquaredDistances;

			for (auto&& neighborId : neighbors) {
				Eigen::Vector3f nodePos(*graphNodes.data(neighborId, 0), *graphNodes.data(neighborId, 1), *graphNodes.data(neighborId, 2));

				float distance2 = (nodePos - pixelPos).squaredNorm();
				bool bInserted = false;
				for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
					// We insert the element at the first position where its distance is smaller than the other
					// element's distance, which enables us to always keep a sorted list of at most k nearest
					// neighbors.
					if (distance2 <= it->second) {
						it = nearestNodesWithSquaredDistances.insert(it, std::make_pair(neighborId, distance2));
						bInserted = true;
						break;
					}
				}

				if (!bInserted && nearestNodesWithSquaredDistances.size() < GRAPH_K) {
					nearestNodesWithSquaredDistances.emplace_back(std::make_pair(neighborId, distance2));
				}

				// We keep only the list of k nearest elements.
				if (bInserted && nearestNodesWithSquaredDistances.size() > GRAPH_K) {
					nearestNodesWithSquaredDistances.pop_back();
				}
			}

			// Compute skinning weights.
			std::vector<int> nearestGeodesicNodeIds;
			nearestGeodesicNodeIds.reserve(nearestNodesWithSquaredDistances.size());

			std::vector<float> skinningWeights;
			skinningWeights.reserve(nearestNodesWithSquaredDistances.size());

			float weightSum{0.f};
			for (auto it = nearestNodesWithSquaredDistances.begin(); it != nearestNodesWithSquaredDistances.end(); ++it) {
				int nodeId = it->first;

				Eigen::Vector3f nodePos(*graphNodes.data(nodeId, 0), *graphNodes.data(nodeId, 1), *graphNodes.data(nodeId, 2));
				float weight = compute_anchor_weight(pixelPos, nodePos, nodeCoverage);
				weightSum += weight;

				nearestGeodesicNodeIds.push_back(nodeId);
				skinningWeights.push_back(weight);
			}

			// Normalize the skinning weights.
			int nAnchors = nearestGeodesicNodeIds.size();

			if (weightSum > 0) {
				for (int i = 0; i < nAnchors; i++) skinningWeights[i] /= weightSum;
			} else if (nAnchors > 0) {
				for (int i = 0; i < nAnchors; i++) skinningWeights[i] = 1.f / nAnchors;
			}

			// Store the results.
			for (int i = 0; i < nAnchors; i++) {
				*pixelAnchors.mutable_data(y, x, i) = nearestGeodesicNodeIds[i];
				*pixelWeights.mutable_data(y, x, i) = skinningWeights[i];
			}
		}
	}
}

void compute_pixel_anchors_euclidean(
		const py::array_t<float>& graph_nodes,
		const py::array_t<float>& point_image,
		float node_coverage,
		py::array_t<int>& pixel_anchors,
		py::array_t<float>& pixel_weights
) {
	int node_count = graph_nodes.shape(0);
	int width = point_image.shape(2);
	int height = point_image.shape(1);
	// int nChannels = pointImage.shape(0);

	// Allocate graph node ids and corresponding skinning weights.
	// Initialize with invalid anchors.
	pixel_anchors.resize({height, width, GRAPH_K}, false);
	pixel_weights.resize({height, width, GRAPH_K}, false);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int k = 0; k < GRAPH_K; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}
		}
	}

	// Compute anchors for every pixel.
#pragma omp parallel for
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Query 3d pixel position.
			Eigen::Vector3f pixel_position(*point_image.data(0, y, x),
			                               *point_image.data(1, y, x),
			                               *point_image.data(2, y, x));
			if (pixel_position.z() <= 0) continue;

			// Keep only the k nearest Euclidean neighbors.
			std::list<std::pair<int, float>> nearest_nodes_with_squared_distances;

			for (int node_index = 0; node_index < node_count; node_index++) {
				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
				                              *graph_nodes.data(node_index, 1),
				                              *graph_nodes.data(node_index, 2));

				float squared_distance = (pixel_position - node_position).squaredNorm();
				bool nodes_inserted = false;
				for (auto it = nearest_nodes_with_squared_distances.begin(); it != nearest_nodes_with_squared_distances.end(); ++it) {
					// We insert the element at the first position where its distance is smaller than the other
					// element's distance, which enables us to always keep a sorted list of at most k nearest
					// neighbors.
					if (squared_distance <= it->second) {
						it = nearest_nodes_with_squared_distances.insert(it, std::make_pair(node_index, squared_distance));
						nodes_inserted = true;
						break;
					}
				}

				if (!nodes_inserted && nearest_nodes_with_squared_distances.size() < GRAPH_K) {
					nearest_nodes_with_squared_distances.emplace_back(std::make_pair(node_index, squared_distance));
				}

				// We keep only the list of k nearest elements.
				if (nodes_inserted && nearest_nodes_with_squared_distances.size() > GRAPH_K) {
					nearest_nodes_with_squared_distances.pop_back();
				}
			}

			// Compute skinning weights.
			std::vector<int> nearest_euclidean_node_indices;
			nearest_euclidean_node_indices.reserve(nearest_nodes_with_squared_distances.size());

			std::vector<float> skinning_weights;
			skinning_weights.reserve(nearest_nodes_with_squared_distances.size());

			float weight_sum{0.f};
			for (auto& nearest_nodes_with_squared_distance : nearest_nodes_with_squared_distances) {
				int node_index = nearest_nodes_with_squared_distance.first;

				Eigen::Vector3f node_position(*graph_nodes.data(node_index, 0),
				                              *graph_nodes.data(node_index, 1),
				                              *graph_nodes.data(node_index, 2));
				float weight = compute_anchor_weight(pixel_position, node_position, node_coverage);
				weight_sum += weight;

				nearest_euclidean_node_indices.push_back(node_index);
				skinning_weights.push_back(weight);
			}

			// Normalize the skinning weights.
			int anchor_count = nearest_euclidean_node_indices.size();

			if (weight_sum > 0) {
				for (int i = 0; i < anchor_count; i++) skinning_weights[i] /= weight_sum;
			} else if (anchor_count > 0) {
				for (int i = 0; i < anchor_count; i++) skinning_weights[i] = 1.f / static_cast<float>(anchor_count);
			}

			// Store the results.
			for (int i = 0; i < anchor_count; i++) {
				*pixel_anchors.mutable_data(y, x, i) = nearest_euclidean_node_indices[i];
				*pixel_weights.mutable_data(y, x, i) = skinning_weights[i];
			}
		}
	}
}

/**
  * Sample graph regularly from the image, using pixel-wise connectivity
  * (connecting each pixel with at most 8 neighbors).
  */
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
) {
	int width = point_image.shape(2);
	int height = point_image.shape(1);

	float x_step = float(width - 1) / static_cast<float>(x_nodes - 1);
	float y_step = float(height - 1) / static_cast<float>(y_nodes - 1);

	// Sample graph canonical_node_positions.
	// We need to maintain the mapping from all -> valid canonical_node_positions ids.
	int node_count = x_nodes * y_nodes;
	std::vector<int> sampled_node_mapping(node_count, -1);

	std::vector<Eigen::Vector3f> node_positions;
	node_positions.reserve(node_count);

	int sampled_node_count = 0;
	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int linear_node_index = y * x_nodes + x;

			// We use nearest neighbor interpolation for node position
			// computation.
			int x_pixel = static_cast<int>(std::round(static_cast<float>(x) * x_step));
			int y_pixel = static_cast<int>(std::round(static_cast<float>(y) * y_step));

			Eigen::Vector3f pixel_position(*point_image.data(0, y_pixel, x_pixel),
			                               *point_image.data(1, y_pixel, x_pixel),
			                               *point_image.data(2, y_pixel, x_pixel));
			if (pixel_position.z() <= 0 || pixel_position.z() > max_depth) continue;

			node_positions.push_back(pixel_position);
			sampled_node_mapping[linear_node_index] = sampled_node_count;
			sampled_node_count++;
		}
	}

	// Compute graph edges using pixel-wise connectivity. Each node
	// is connected with at most 8 neighboring pixels.
	int max_neighbor_count = 8;
	float edge_threshold_squared = edge_threshold * edge_threshold;

	std::vector<int> sampled_node_edges(sampled_node_count * max_neighbor_count, -1);
	std::vector<bool> connected_nodes(sampled_node_count, false);

	int connected_node_count = 0;
	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int node_index = y * x_nodes + x;
			int sampled_node_index = sampled_node_mapping[node_index];

			if (sampled_node_index >= 0) {
				Eigen::Vector3f node_position = node_positions[sampled_node_index];

				int neighbor_count = 0;
				for (int y_delta = -1; y_delta <= 1; y_delta++) {
					for (int x_delta = -1; x_delta <= 1; x_delta++) {
						int x_neighbor = x + x_delta;
						int y_neighbor = y + y_delta;
						if (x_neighbor < 0 || x_neighbor >= x_nodes || y_neighbor < 0 || y_neighbor >= y_nodes)
							continue;

						int neighbor_index = y_neighbor * x_nodes + x_neighbor;

						if (neighbor_index == node_index || neighbor_index < 0)
							continue;

						int sampled_neighbor_index = sampled_node_mapping[neighbor_index];
						if (sampled_neighbor_index >= 0) {
							Eigen::Vector3f neighbor_position = node_positions[sampled_neighbor_index];

							if ((neighbor_position - node_position).squaredNorm() <= edge_threshold_squared) {
								sampled_node_edges[sampled_node_index * max_neighbor_count + neighbor_count] = sampled_neighbor_index;
								neighbor_count++;
							}
						}
					}
				}

				for (int i = neighbor_count; i < max_neighbor_count; i++) {
					sampled_node_edges[sampled_node_index * max_neighbor_count + i] = -1;
				}

				if (neighbor_count > 0) {
					connected_nodes[sampled_node_index] = true;
					connected_node_count += 1;
				}
			}
		}
	}

	// Filter out canonical_node_positions with no edges.
	// After changing node ids the edge ids need to be changed as well.
	std::vector<int> valid_node_mapping(sampled_node_count, -1);
	{
		graph_nodes.resize({connected_node_count, 3}, false);
		graph_edges.resize({connected_node_count, max_neighbor_count}, false);

		int valid_node_index = 0;
		for (int y = 0; y < y_nodes; y++) {
			for (int x = 0; x < x_nodes; x++) {
				int node_index = y * x_nodes + x;
				int sampled_node_index = sampled_node_mapping[node_index];

				if (sampled_node_index >= 0 && connected_nodes[sampled_node_index]) {
					valid_node_mapping[sampled_node_index] = valid_node_index;

					Eigen::Vector3f node_position = node_positions[sampled_node_index];
					*graph_nodes.mutable_data(valid_node_index, 0) = node_position.x();
					*graph_nodes.mutable_data(valid_node_index, 1) = node_position.y();
					*graph_nodes.mutable_data(valid_node_index, 2) = node_position.z();

					valid_node_index++;
				}
			}
		}
	}

	for (int y = 0; y < y_nodes; y++) {
		for (int x = 0; x < x_nodes; x++) {
			int node_index = y * x_nodes + x;
			int sampled_node_index = sampled_node_mapping[node_index];

			if (sampled_node_index >= 0 && connected_nodes[sampled_node_index]) {
				int valid_node_index = valid_node_mapping[sampled_node_index];

				if (valid_node_index >= 0) {
					for (int i = 0; i < max_neighbor_count; i++) {
						int sampledNeighborId = sampled_node_edges[sampled_node_index * max_neighbor_count + i];
						if (sampledNeighborId >= 0) {
							*graph_edges.mutable_data(valid_node_index, i) = valid_node_mapping[sampledNeighborId];
						} else {
							*graph_edges.mutable_data(valid_node_index, i) = -1;
						}
					}
				}
			}
		}
	}

	// Compute pixel anchors and weights.
	pixel_anchors.resize({height, width, 4}, false);
	pixel_weights.resize({height, width, 4}, false);

	float max_point_to_node_distance_squared = max_point_to_node_distance * max_point_to_node_distance;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// Initialize with invalid values.
			for (int k = 0; k < 4; k++) {
				*pixel_anchors.mutable_data(y, x, k) = -1;
				*pixel_weights.mutable_data(y, x, k) = 0.f;
			}

			// Compute 4 nearest canonical_node_positions.
			float x_node = float(x) / x_step;
			float y_node = float(y) / y_step;

			int x0 = std::floor(x_node), x1 = x0 + 1;
			int y0 = std::floor(y_node), y1 = y0 + 1;

			// Check that all neighboring canonical_node_positions are valid.
			if (x0 < 0 || x1 >= x_nodes || y0 < 0 || y1 >= y_nodes)
				continue;

			int sampled_node_00 = sampled_node_mapping[y0 * x_nodes + x0];
			int sampled_node_01 = sampled_node_mapping[y1 * x_nodes + x0];
			int sampled_node_10 = sampled_node_mapping[y0 * x_nodes + x1];
			int sampled_node_11 = sampled_node_mapping[y1 * x_nodes + x1];

			if (sampled_node_00 < 0 || sampled_node_01 < 0 || sampled_node_10 < 0 || sampled_node_11 < 0)
				continue;

			int valid_node_00 = valid_node_mapping[sampled_node_00];
			int valid_node_01 = valid_node_mapping[sampled_node_01];
			int valid_node_10 = valid_node_mapping[sampled_node_10];
			int valid_node_11 = valid_node_mapping[sampled_node_11];

			if (valid_node_00 < 0 || valid_node_01 < 0 || valid_node_10 < 0 || valid_node_11 < 0)
				continue;

			// Check that all canonical_node_positions are close enough to the point.
			Eigen::Vector3f pixel_position(*point_image.data(0, y, x),
			                               *point_image.data(1, y, x),
			                               *point_image.data(2, y, x));
			if (pixel_position.z() <= 0 || pixel_position.z() > max_depth) continue;

			if ((pixel_position - node_positions[sampled_node_00]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_01]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_10]).squaredNorm() > max_point_to_node_distance_squared ||
			    (pixel_position - node_positions[sampled_node_11]).squaredNorm() > max_point_to_node_distance_squared) {
				continue;
			}

			// Compute bilinear weights.
			float dx = x_node - static_cast<float>(x0);
			float dy = y_node - static_cast<float>(y0);

			float w00 = (1 - dx) * (1 - dy);
			float w01 = (1 - dx) * dy;
			float w10 = dx * (1 - dy);
			float w11 = dx * dy;

			*pixel_anchors.mutable_data(y, x, 0) = valid_node_00;
			*pixel_weights.mutable_data(y, x, 0) = w00;
			*pixel_anchors.mutable_data(y, x, 1) = valid_node_01;
			*pixel_weights.mutable_data(y, x, 1) = w01;
			*pixel_anchors.mutable_data(y, x, 2) = valid_node_10;
			*pixel_weights.mutable_data(y, x, 2) = w10;
			*pixel_anchors.mutable_data(y, x, 3) = valid_node_11;
			*pixel_weights.mutable_data(y, x, 3) = w11;
		}
	}
}

py::tuple construct_regular_graph(
		const py::array_t<float>& point_image,
		int x_nodes, int y_nodes,
		float edge_threshold,
		float max_point_to_node_distance,
		float max_depth
) {
	py::array_t<float> graph_nodes;
	py::array_t<int> graph_edges;
	py::array_t<int> pixel_anchors;
	py::array_t<float> pixel_weights;
	construct_regular_graph(point_image, x_nodes, y_nodes, edge_threshold,
	                        max_point_to_node_distance, max_depth,
	                        graph_nodes, graph_edges, pixel_anchors, pixel_weights);
	return py::make_tuple(graph_nodes, graph_edges, pixel_anchors, pixel_weights);
}

} // namespace graph_proc