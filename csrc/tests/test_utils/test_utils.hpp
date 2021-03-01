//  ================================================================
//  Created by Gregory Kramida (https://github.com/Algomorph) on 2/28/21.
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
#pragma once

#include "compile_time_string_concatenation.hpp"
#include <test_data_paths.hpp>

#include <memory>
#include <pybind11/numpy.h>



namespace test {
	static constexpr auto image_test_data_directory = StringFactory(STATIC_TEST_DATA_DIRECTORY "images");

	// image persistence
	template<typename TElement>
	pybind11::array_t<TElement> load_image(const std::string& path);
	template<typename TElement>
	void save_image(const pybind11::array_t<TElement>& image, const std::string& path);

	// index conversions
	std::vector<long> unravel_index(long linear_index, const std::vector<long>& dimensions);

	template<typename TElement>
	struct array_element_mismatch_information{
		std::vector<long> position;
		long linear_index;
		TElement element1;
		TElement element2;
		float absolute_tolerance;
	};

	struct array_dimension_mismatch_information{
		std::vector<long> dimensions1;
		std::vector<long> dimensions2;
	};

	template<typename TElement>
	struct array_comparison_result{
		bool arrays_match;
		std::shared_ptr<array_dimension_mismatch_information> dimension_mismatch_information;
		std::shared_ptr<array_element_mismatch_information<TElement>> element_mismatch_information;
	};



	template<typename TElement>
	array_comparison_result<TElement> compare(const pybind11::array_t<TElement>& array1, const pybind11::array_t<TElement>& array2, TElement absolute_tolerance);


} // namespace test