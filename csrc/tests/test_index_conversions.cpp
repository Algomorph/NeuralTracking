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

TEST_CASE("Test Index Unraveling") {
	std::vector<long> dimensions = {2,4,3};
	std::vector<long> position10 = test::unravel_index(10, dimensions);
	std::vector<long> position10_gt = {0, 3, 1};
	REQUIRE(position10 == position10_gt);
	std::vector<long> position14 = test::unravel_index(14, dimensions);
	std::vector<long> position14_gt = {1, 0, 2};
	REQUIRE(position14 == position14_gt);
}