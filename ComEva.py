import open3d as o3d
import numpy as np
import os

def preprocess_point_cloud(pcd, voxel_size):
    print("预处理点云...")
    # 下采样点云
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 估计法线
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    # 计算FPFH特征
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def global_registration_with_icp(source, target, voxel_size):
    print("粗配准...")
    # 预处理源点云和目标点云
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # 粗配准
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, voxel_size * 1.5, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

def refine_registration_with_adaptive_threshold(source, target, init_transformation, voxel_size):
    print("精细配准...")
    # 计算点云平均距离来调整距离阈值
    avg_distance = np.mean(np.linalg.norm(np.asarray(source.points)[1:] - np.asarray(source.points)[:-1], axis=1))
    distance_threshold = 2 * avg_distance
    # 限制阈值范围
    distance_threshold = np.clip(distance_threshold, 0.01, 0.05)
    # 精细配准
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500)
    )
    return result

def merge_point_clouds(pcds, output_path):
    print("合并点云...")
    merged_pcd = o3d.geometry.PointCloud()
    # 合并所有点云
    for pcd in pcds:
        merged_pcd += pcd
    # 保存合并后的点云
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"合并点云保存至: {output_path}")
    return merged_pcd

def evaluate_point_clouds(fused_pcd, reference_pcd, voxel_size=0.05):
    print("评估点云...")
    # 计算点云间的距离
    distances = fused_pcd.compute_point_cloud_distance(reference_pcd)
    rmse_value = np.sqrt(np.mean(np.array(distances)**2))
    hausdorff_dist = max(np.max(distances), np.max(reference_pcd.compute_point_cloud_distance(fused_pcd)))
    overlap = overlap_ratio(fused_pcd, reference_pcd, voxel_size)
    print(f"RMSE: {rmse_value:.5f}, Hausdorff 距离: {hausdorff_dist:.5f}, 重叠度: {overlap * 100:.2f}%")

def overlap_ratio(pcd1, pcd2, voxel_size=0.1, distance_threshold=0.1):
    print("计算重叠度...")
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)
    pcd1_tree = o3d.geometry.KDTreeFlann(pcd1_down)
    pcd2_tree = o3d.geometry.KDTreeFlann(pcd2_down)

    overlap_count = 0
    reverse_overlap_count = 0
    total_points_1 = len(pcd1_down.points)
    total_points_2 = len(pcd2_down.points)

    # 根据点云的密度来调整距离阈值
    adjusted_threshold = adjust_threshold_based_on_density(pcd1, distance_threshold)

    # 计算重叠度
    for point in pcd1_down.points:
        [_, idx, distances] = pcd2_tree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and distances[0] < adjusted_threshold:
            overlap_count += 1

    for point in pcd2_down.points:
        [_, idx, distances] = pcd1_tree.search_knn_vector_3d(point, 1)
        if len(idx) > 0 and distances[0] < adjusted_threshold:
            reverse_overlap_count += 1

    total_overlap = (overlap_count + reverse_overlap_count) / (total_points_1 + total_points_2)
    return total_overlap

def adjust_threshold_based_on_density(pcd, base_distance_threshold=0.1):
    # 根据点云密度调整距离阈值
    avg_distance = compute_average_distance(pcd)
    adjusted_threshold = avg_distance * 1.3 if avg_distance > base_distance_threshold else avg_distance * 0.8
    return adjusted_threshold

def compute_average_distance(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    total_distance = 0.0
    num_points = len(pcd.points)

    # 计算点云的平均距离
    for i in range(num_points):
        [k, idx, dist] = pcd_tree.search_knn_vector_3d(pcd.points[i], 2)
        total_distance += np.sqrt(dist[1])

    return total_distance / num_points

def denoise_and_downsample_v2(pcd, voxel_size):
    print("去噪与下采样...")
    # 去除异常点并进行下采样
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    return pcd_downsampled

def main():
    reference_pcd = o3d.io.read_point_cloud(r"D:\depth to cloud\fused_small.ply")
    depth_pcd = o3d.io.read_point_cloud(r"D:\depth to cloud\point_cloud.ply")
    
    # 进行粗配准
    result_global = global_registration_with_icp(depth_pcd, reference_pcd, voxel_size=0.05)
    # 进行精细配准
    result_refined = refine_registration_with_adaptive_threshold(depth_pcd, reference_pcd, result_global.transformation, voxel_size=0.05)

    # 评估配准后的点云
    evaluate_point_clouds(depth_pcd, reference_pcd)

    # 合并多个点云
    pcds_to_merge = [o3d.io.read_point_cloud(path) for path in [
        r"D:\depth to cloud\point_cloud.ply",
        r"D:\depth to cloud\sparse.ply"
    ]]

    output_path = r"D:\depth to cloud\merged_point_cloud_7.ply"
    merged_pcd = merge_point_clouds(pcds_to_merge, output_path)

    voxel_size = 0.05
    # 去噪并下采样
    denoised_pcd = denoise_and_downsample_v2(merged_pcd, voxel_size)

    # 进行粗配准与精细配准
    result_global = global_registration_with_icp(denoised_pcd, reference_pcd, voxel_size)
    result_refined = refine_registration_with_adaptive_threshold(denoised_pcd, reference_pcd, result_global.transformation, voxel_size)

    # 评估去噪后的点云
    evaluate_point_clouds(denoised_pcd, reference_pcd, voxel_size)

    # 应用配准变换
    aligned_pcd = denoised_pcd.transform(result_refined.transformation)
    # 保存对齐后的点云
    o3d.io.write_point_cloud(r"D:\depth to cloud\aligned_point_cloud.ply", aligned_pcd)

if __name__ == "__main__":
    main()