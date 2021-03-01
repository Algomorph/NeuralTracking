//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/23/21.
//  Copyright (c) 2021 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================
#include "test_main.hpp"

#include <cpu/graph_proc.h>
#include <cpu/image_proc.h>

#include "tests/test_utils/test_utils.hpp"



TEST_CASE("Test Regular Graph Construction") {
	std::string depth_path = test::image_test_data_directory.ToString() + "/frame-000000.depth.png";
	py::array_t<unsigned short> depth = test::load_image<unsigned short>(depth_path);

	float fx = 570.342f;
	float fy = 570.342f;
	float cx = 320.0f;
	float cy = 240.0f;
	float depth_scale = 1000.0f;

	py::array_t<float> point_image = image_proc::backproject_depth_ushort(depth, fx, fy, cx, cy, depth_scale);
	py::tuple output = graph_proc::construct_regular_graph(point_image,
	                                                       depth.shape(1) / 2,
	                                                       depth.shape(0) / 2,
	                                                       2.0f,
	                                                       0.05f,
	                                                       3.0f);

	py::array_t<float> graph_nodes = static_cast<const pybind11::object&>(output[0]);
	py::array_t<int> graph_edges = static_cast<const pybind11::object&>(output[1]);
	py::array_t<int> pixel_anchors = static_cast<const pybind11::object&>(output[2]);
	py::array_t<float> pixel_weights = static_cast<const pybind11::object&>(output[3]);
	py::array_t<unsigned short> graph_edges_ushort = graph_edges;
	py::array_t<unsigned short> pixel_anchors_ushort = pixel_anchors;

	std::string graph_nodes_path = test::image_test_data_directory.ToString() + "/frame-000000.graph_nodes.dng";
	// TODO: need a way to save arbitrary-sized arrays and int arrays, preferably with compression.
	//  Take a look at z_stream dependency / IStreamWrapper / OStreamWrapper in ORUtils of github.com/Algomorph/InfiniTAM
	//  The zlib dependency can be handled the same way as in Open3D
	// std::string pixel_weights_path = test::image_test_data_directory.ToString() + "/frame-000000.pixel_weights.dng";


	py::array_t<float> graph_nodes_gt = test::load_image<float>(graph_nodes_path);
	auto result = test::compare(graph_nodes, graph_nodes_gt, 1e-6f);
	REQUIRE(result.arrays_match);
	// TODO: after GT + more comparison functions are available, compare the graph_edges, pixel_anchors, and pixel_weights
	//  as well


}