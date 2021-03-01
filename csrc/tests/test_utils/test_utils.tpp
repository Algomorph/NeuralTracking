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

#include "test_utils.hpp"
#include <algorithm>

#include <pybind11/numpy.h>
#include <pybind11/detail/descr.h>

#include <dlib/image_io.h>
#include <dlib/python.h>

namespace py = pybind11;
using namespace dlib;

namespace test {

// loading, general
template<typename TElement>
py::array_t<TElement> load_image(const std::string& path) {
	dlib::numpy_image<unsigned short> image;
	dlib::load_png(image, path);
	return image;
}

// saving, general
template<typename TElement>
void save_image(const py::array_t<TElement>& image, const std::string& path) {
	dlib::numpy_image<float> _image(image);
	dlib::save_png(_image, path);
}

template<typename TElement>
static inline std::vector<long> dimensions_as_std_vector(const pybind11::array_t<TElement>& array1) {
	std::vector<long> dimensions;
	for (int dimension_index = 0; dimension_index < array1.ndim(); dimension_index++) {
		dimensions.push_back(array1.shape(dimension_index));
	}
	return dimensions;
}


// comparison, general
template<typename TElement>
array_comparison_result<TElement>
compare(const pybind11::array_t<TElement>& array1, const pybind11::array_t<TElement>& array2, TElement absolute_tolerance) {
	array_comparison_result<TElement> result;
	result.arrays_match = true;
	std::vector<long> dimensions1 = dimensions_as_std_vector(array1);
	std::vector<long> dimensions2 = dimensions_as_std_vector(array2);
	if (dimensions1 != dimensions2) {
		result.arrays_match = false;
		result.dimension_mismatch_information = std::make_shared<array_dimension_mismatch_information>(
				array_dimension_mismatch_information{dimensions1, dimensions2}
				);
		return result;
	}

	const TElement* data1 = array1.data();
	const TElement* data2 = array2.data();

#pragma omp parallel for default(none) shared(data1, data2, result)
	for (long element_index = 0; element_index < array1.size() && result.arrays_match; element_index++) {
		const TElement& element1 = data1[element_index];
		const TElement& element2 = data2[element_index];
		if (std::abs(element1 - element2) > absolute_tolerance) {
#pragma omp critical
			{
				if(result.arrays_match != false){
					result.arrays_match = false;
					std::vector<long> position = unravel_index(element_index, dimensions1);
					result.element_mismatch_information = std::make_shared<array_element_mismatch_information<TElement>>(
							array_element_mismatch_information<TElement>{position, element_index, element1, element2, absolute_tolerance}
					);
				}
			}
		}
	}
	return result;
}


} // namespace test