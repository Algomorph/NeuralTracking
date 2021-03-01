import open3d as o3d


def make_z_aligned_image_plane(min_pt, max_pt, z, image):
    plane_vertices = [
        [min_pt[0], min_pt[1], z],
        [max_pt[0], min_pt[1], z],
        [max_pt[0], max_pt[1], z],
        [min_pt[0], max_pt[1], z]
    ]
    plane_triangles = [[2, 1, 0],
                       [0, 3, 2]]

    plane_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(plane_vertices), o3d.utility.Vector3iVector(plane_triangles))
    plane_mesh.compute_vertex_normals()

    plane_texture_coordinates = [
        (1, 1), (1, 0), (0, 0),
        (0, 0), (0, 1), (1, 1)
    ]

    plane_mesh.triangle_uvs = o3d.utility.Vector2dVector(plane_texture_coordinates)
    plane_mesh.triangle_material_ids = o3d.utility.IntVector([0, 0])
    plane_mesh.textures = [image]
    return plane_mesh
