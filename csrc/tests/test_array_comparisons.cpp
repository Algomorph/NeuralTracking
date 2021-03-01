//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 3/1/21.
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
#include "tests/test_utils/test_utils.hpp"
namespace py = pybind11;
TEST_CASE("Test Float Array Comparison") {
	py::array_t<float> t1_a({2,3});
	std::array<float, 6> t1_a_values = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
	memcpy(t1_a.mutable_data(), t1_a_values.data(), 6 * sizeof(float));

	py::array_t<float> t1_b({2,3});
	std::array<float, 6> t1_b_values = {0.102, 0.201, 0.311, 0.398, 0.499, 0.602};
	memcpy(t1_b.mutable_data(), t1_b_values.data(), 6 * sizeof(float));

	auto t1_result1 = test::compare(t1_a, t1_b, 0.02f);
	REQUIRE(t1_result1.arrays_match);

	auto t1_result2 = test::compare(t1_a, t1_b, 0.01f);
	REQUIRE(!t1_result2.arrays_match);
	REQUIRE(t1_result2.element_mismatch_information->linear_index == 2);
	std::vector<long> t1_result2_expected_position = {0, 2};
	REQUIRE(t1_result2.element_mismatch_information->position == t1_result2_expected_position);

}