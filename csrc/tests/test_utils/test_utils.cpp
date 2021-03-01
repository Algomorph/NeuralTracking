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
#include "test_utils.tpp"

namespace test {

template py::array_t<unsigned short> load_image<unsigned short>(const std::string& path);

// loading specializations
template<>
py::array_t<float> load_image<float>(const std::string& path) {
	dlib::numpy_image<float> image;
	dlib::load_dng(image, path);
	return image;
}


template void save_image<unsigned short>(const pybind11::array_t<unsigned short>& image, const std::string& path);

// saving specializations
template<>
void save_image<float>(const py::array_t<float>& image, const std::string& path) {
	dlib::numpy_image<float> _image(image);
	dlib::save_dng(_image, path);
}

// index conversions
std::vector<long> unravel_index(long linear_index, const std::vector<long>& dimensions) {
	std::vector<long> position;
	long dividend = linear_index;

	for (auto iterator = dimensions.rbegin(); iterator != dimensions.rend(); ++iterator) {
		long dimension = *iterator;
		long remainder = dividend % dimension;
		position.push_back(remainder);
		dividend = dividend / dimension;
	}
	std::reverse(position.begin(), position.end());
	return position;
}

template array_comparison_result<float>
compare(const pybind11::array_t<float>& array1, const pybind11::array_t<float>& array2, float absolute_tolerance);

} // namepspace test
