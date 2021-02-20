#pragma once

#include <iostream>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace image_proc {

    #define IS_IN_RANGE2(x1, y1) ((x1) >= 0 && (x1) < width && (y1) >= 0 && (y1) < height)
    #define INDEX2(dy, dx) (IS_IN_RANGE2((x + dx), (y + dy)) ? *ptr.data(i, 0,  y + dy, x + dx) : 0)
    #define CHECK2() \
            (INDEX2(-1, -1) || INDEX2(-1, 0) || INDEX2(-1, 1) \
            ||	INDEX2(0, -1)  || INDEX2(0, 0)  || INDEX2(0, 1) \
            ||	INDEX2(1, -1)  || INDEX2(1, 0)  || INDEX2(1, 1)) \

    #define IS_IN_RANGE3(x1, y1, z1) ((x1) >= 0 && (x1) < dimx && (y1) >= 0 && (y1) < dimy && (z1) >= 0 && (z1) < dimz)
    #define INDEX3(dz, dy, dx) (IS_IN_RANGE3((x + dx), (y + dy), (z + dz)) ? *ptr.data(i, 0, z + dz,  y + dy, x + dx) : 0)
    #define CHECK3() \
            (INDEX3(-1, -1, -1) || INDEX3(-1, -1, 0) || INDEX3(-1, -1, 1) \
            ||	INDEX3(-1, 0, -1)  || INDEX3(-1, 0, 0)  || INDEX3(-1, 0, 1) \
            ||	INDEX3(-1, 1, -1)  || INDEX3(-1, 1, 0)  || INDEX3(-1, 1, 1) \
            ||	INDEX3(0, -1, -1)  || INDEX3(0, -1, 0)  || INDEX3(0, -1, 1) \
            ||	INDEX3(0, 0,  -1)   || INDEX3(0, 0, 0)   || INDEX3(0, 0, 1) \
            ||	INDEX3(0, 1,  -1)   || INDEX3(0, 1, 0)   || INDEX3(0, 1, 1) \
            ||	INDEX3(1, -1, -1) || INDEX3(1, -1, 0) || INDEX3(1, -1, 1) \
            ||	INDEX3(1, 0,  -1)  || INDEX3(1, 0, 0)  || INDEX3(1, 0, 1) \
            ||	INDEX3(1, 1,  -1)  || INDEX3(1, 1, 0)  || INDEX3(1, 1, 1)) \

    py::array_t<float> compute_augmented_flow_from_rotation(py::array_t<float>& flow_image_rot_sa2so,
                                                            py::array_t<float>& flow_image_so2to,
                                                            py::array_t<float>& flow_image_rot_to2ta,
                                                            const int height, const int width);



    int count_tp1(py::array_t<bool> &p, py::array_t<bool> &gt);    

    int count_tp2(py::array_t<bool> &p, py::array_t<bool> &gt);

    int count_tp3(py::array_t<bool> &p, py::array_t<bool> &gt);

    void extend3(py::array_t<bool> &in, py::array_t<bool> &out);

    void backproject_depth_ushort(py::array_t<unsigned short>& image_in, py::array_t<float>& point_image_out, float fx, float fy, float cx, float cy, float normalizer);
	py::array_t<float> backproject_depth_ushort(py::array_t<unsigned short>& image_in, float fx, float fy, float cx, float cy, float normalizer);

    void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out, float fx, float fy, float cx, float cy);

    void compute_mesh_from_depth(
		    const py::array_t<float>& point_image, float max_triangle_edge_distance,
		    py::array_t<float>& vertex_positions, py::array_t<int>& face_indices
    );

    void compute_mesh_from_depth_and_color(
		    const py::array_t<float>& point_image, const py::array_t<int>& color_image, float max_triangle_edge_distance,
		    py::array_t<float>& vertex_positions, py::array_t<int>& vertex_colors, py::array_t<int>& face_indices
    );

    void filter_depth(py::array_t<unsigned short>& depth_image_in, py::array_t<unsigned short>& depth_image_out, int radius);
	py::array_t<unsigned short> filter_depth(py::array_t<unsigned short>& depth_image_in, int radius);

    py::array_t<float> warp_flow(const py::array_t<float>& image, const py::array_t<float>& flow, const py::array_t<float>& mask);

    py::array_t<float> warp_rigid(const py::array_t<float>& rgbxyz_image, const py::array_t<float>& rotation, const py::array_t<float>& translation, float fx, float fy, float cx, float cy);

    py::array_t<float> warp_3d(const py::array_t<float>& rgbxyz_image, const py::array_t<float>& points, const py::array_t<int>& mask, float fx, float fy, float cx, float cy);

} // namespace image_proc