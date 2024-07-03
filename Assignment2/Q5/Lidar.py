import os
import numpy as np
import open3d as o3d

camera_folder = "camera_images"
camera_image_filenames = [os.path.splitext(file)[0] for file in os.listdir(camera_folder) if file.endswith('.jpeg')]

camera_image_file_paths = [os.path.join(camera_folder, filename + ".jpeg") for filename in camera_image_filenames]

lidar_folder = "lidar_scans"
lidar_scan_file_paths = [os.path.join(lidar_folder, filename + ".pcd") for filename in camera_image_filenames]

def compute_lidar_plane(pcd):
    points = np.asarray(pcd.points)
    
    centroid = np.mean(points, axis=0)
    
    centered_points = points - centroid
    
    _, _, V = np.linalg.svd(centered_points)
    
    normal = V[-1]
    
    offset = -np.dot(normal, centroid)
    
    return normal, offset

lidar_normals = []
offsets = []

for lidar_file_path in lidar_scan_file_paths:

    pcd = o3d.io.read_point_cloud(lidar_file_path)
    
    normal, offset = compute_lidar_plane(pcd)
    
    lidar_normals.append(normal)
    offsets.append(offset)

for i, (normal, offset) in enumerate(zip(lidar_normals, offsets), 1):
    print(f"Plane {i}: Normal = {normal}, Offset = {offset}")