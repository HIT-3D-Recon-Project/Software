import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def preprocess_point_cloud(pcd, voxel_size):
    # 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 估计法线
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 计算FPFH特征
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # ransac_n
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def normalize_point_cloud(pcd):
    # 获取点云的包围盒
    bbox = pcd.get_axis_aligned_bounding_box()
    # 计算包围盒的对角线长度
    diagonal_length = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    # 计算缩放因子，将对角线长度归一化到1
    scale = 1.0 / diagonal_length
    # 获取点云中心
    center = bbox.get_center()
    
    # 创建新的点云副本
    normalized_pcd = o3d.geometry.PointCloud(pcd)
    # 将点云平移到原点
    points = np.asarray(normalized_pcd.points)
    points = points - center
    # 应用缩放
    points = points * scale
    normalized_pcd.points = o3d.utility.Vector3dVector(points)
    
    # 保持颜色信息
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        normalized_pcd.colors = pcd.colors
    
    return normalized_pcd, scale, center

def transfer_colors(source_pcd, target_pcd, k=1):
    """
    将source点云的颜色转移到target点云
    使用K近邻搜索找到最近的有颜色的点
    """
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors)
    target_points = np.asarray(target_pcd.points)
    
    # 使用KNN找到最近邻点
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(source_points)
    distances, indices = nbrs.kneighbors(target_points)
    
    # 为目标点云分配颜色
    target_colors = np.mean(source_colors[indices], axis=1)
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
    
    return target_pcd

# 读取点云文件
source = o3d.io.read_point_cloud("final.ply")  # 有颜色的点云
target = o3d.io.read_point_cloud("point_cloud.ply")  # 没有颜色的点云

# 归一化两个点云
source_normalized, source_scale, source_center = normalize_point_cloud(source)
target_normalized, target_scale, target_center = normalize_point_cloud(target)

print(f"Source scale: {source_scale}, Target scale: {target_scale}")
print(f"Source center: {source_center}")
print(f"Target center: {target_center}")

# 设置不同的体素大小
source_voxel_size = 0.005  # final.ply使用更小的体素大小，保留更多点
target_voxel_size = 0.01   # point_cloud.ply也使用较小的体素大小
registration_voxel_size = 0.02  # 用于配准的体素大小保持不变，以确保配准效果

# 对source(final.ply)进行轻度下采样
source_down_registration, source_fpfh = preprocess_point_cloud(source_normalized, registration_voxel_size)
target_down_registration, target_fpfh = preprocess_point_cloud(target_normalized, registration_voxel_size)

# 执行全局配准
result_ransac = execute_global_registration(source_down_registration, target_down_registration, 
                                          source_fpfh, target_fpfh, registration_voxel_size)
print("Global registration result:")
print(result_ransac)

# 使用ICP进行精细配准
result_icp = refine_registration(source_normalized, target_normalized, result_ransac, registration_voxel_size)
print("ICP refinement result:")
print(result_icp)

# 转换源点云
source_normalized.transform(result_icp.transformation)

# 在配准后，为target点云添加颜色
target_normalized = transfer_colors(source_normalized, target_normalized, k=3)

# 分别对两个点云进行不同程度的下采样
source_down = source_normalized.voxel_down_sample(source_voxel_size)  # 轻度下采样
target_down = target_normalized.voxel_down_sample(target_voxel_size)  # 较强下采样

# 将两个点云合并
combined_pcd = source_down + target_down

# 还原到原始尺度（使用source的尺度作为参考）
points = np.asarray(combined_pcd.points)
points = points / source_scale  # 还原缩放
points = points + source_center  # 还原平移
combined_pcd.points = o3d.utility.Vector3dVector(points)

# 保存结果
o3d.io.write_point_cloud("combined_pointcloud.ply", combined_pcd)

# 可视化结果
o3d.visualization.draw_geometries([combined_pcd])
