import open3d as o3d
import numpy as np

# -------------------------------
# 1. Load your point cloud
# -------------------------------
pcd = o3d.io.read_point_cloud("./b342792b-3/point_cloud/iteration_30000/point_cloud.ply")

# -------------------------------
# 2. Estimate normals (needed for mesh reconstruction)
# -------------------------------
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)

# -------------------------------
# 3. Optional: downsample for faster processing
# -------------------------------
# pcd_down = pcd.voxel_down_sample(voxel_size=0.001)
# pcd_down.estimate_normals()
#
# # -------------------------------
# # 4. Convert point cloud to mesh
# #    Using Ball Pivoting method
# # -------------------------------
# distances = pcd_down.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3 * avg_dist
#
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd_down,
#     o3d.utility.DoubleVector([radius, radius * 2])
# )
#
# # Optional: simplify mesh for smaller file
# mesh = mesh.simplify_vertex_clustering(voxel_size=0.001)
#
# # -------------------------------
# # 5. Save mesh as GLB
# # -------------------------------
# o3d.io.write_triangle_mesh("scene.glb", mesh, write_triangle_uvs=True)
#
# print("Mesh saved as scene.glb successfully!")
#

# Downsample
pcd_down = pcd.voxel_down_sample(voxel_size=0.003)
pcd_down.estimate_normals()

# Ball Pivoting mesh
distances = pcd_down.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 2 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd_down,
    o3d.utility.DoubleVector([radius, radius*2])
)

# Save as GLB
o3d.io.write_triangle_mesh("scene.glb", mesh, write_triangle_uvs=True)
print("Mesh saved as scene.glb successfully!")