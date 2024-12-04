import open3d as o3d
import numpy as np
from PIL import Image
import os

def read_colmap_cameras(cameras_file):
    """读取COLMAP的相机参数文件"""
    with open(cameras_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过注释行
    for line in lines:
        if not line.startswith('#'):
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = line.strip().split()
            if len(parts) >= 8:  # 确保有足够的参数
                return {
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'fx': float(parts[4]),  # 焦距
                    'cx': float(parts[5]),  # 主点x
                    'cy': float(parts[6]),  # 主点y
                    'k': float(parts[7])    # 径向畸变
                }
    return None

def read_colmap_points(points3d_file):
    """读取COLMAP的3D点云文件以获取深度范围"""
    min_depth = float('inf')
    max_depth = float('-inf')
    
    with open(points3d_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                z = float(parts[3])
                min_depth = min(min_depth, z)
                max_depth = max(max_depth, z)
    
    return min_depth, max_depth

def depth_to_pointcloud(depth_path, output_path, cameras_file=None, points3d_file=None):
    """
    将深度图转换为点云
    :param depth_path: 深度图路径
    :param output_path: 输出点云路径
    :param cameras_file: COLMAP相机参数文件路径
    :param points3d_file: COLMAP点云文件路径
    """
    # 读取深度图
    depth_image = np.array(Image.open(depth_path))
    
    # 如果是RGB图像，转换为灰度图
    if len(depth_image.shape) == 3:
        depth_image = depth_image.mean(axis=2)
    
    # 确保深度图是浮点数类型
    depth_image = depth_image.astype(np.float32)
    
    # MiDaS深度图处理
    # 反转深度值（MiDaS输出中，较大的值表示较近的物体）
    depth_image = depth_image.max() - depth_image
    
    # 使用两次参数的中值
    min_depth = 1.0   # 保持最小深度
    max_depth = 5.25  # 取6.0和4.5的中值
    
    # 非线性深度映射，保持相对深度关系
    # 使用sigmoid函数进行平滑映射
    normalized_depth = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    
    # 使用修改后的sigmoid映射来保持中间范围的细节
    alpha = 3.75  # 取4.0和3.5的中值
    beta = 0.625  # 取0.6和0.65的中值
    sigmoid_depth = 1 / (1 + np.exp(-alpha * (normalized_depth - beta)))
    
    # 映射到实际深度范围
    depth_image = min_depth + (max_depth - min_depth) * sigmoid_depth
    
    # 应用深度校正因子，减小深度变化的极端性
    depth_scale = 0.375  # 取0.4和0.35的中值
    depth_image = depth_image * depth_scale
    
    # 确保深度图是连续的内存布局
    depth_image = np.ascontiguousarray(depth_image)
    
    print("Depth image shape:", depth_image.shape)
    print("Depth image dtype:", depth_image.dtype)
    print("Depth range:", depth_image.min(), "to", depth_image.max())
    
    # 创建相机内参矩阵
    if cameras_file and os.path.exists(cameras_file):
        # 使用COLMAP相机参数
        camera_params = read_colmap_cameras(cameras_file)
        if camera_params:
            width = camera_params['width']
            height = camera_params['height']
            fx = camera_params['fx']
            fy = camera_params['fx']  # SIMPLE_RADIAL模型假设fx=fy
            cx = camera_params['cx']
            cy = camera_params['cy']
            print("Using COLMAP camera parameters")
            print(f"Camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    else:
        # 使用默认参数
        width, height = depth_image.shape[1], depth_image.shape[0]
        fx = 3458.0  # 基于COLMAP数据的观察
        fy = 3458.0
        cx = 1736.0
        cy = 2312.0
        print("Using default camera parameters")
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, fx, fy, cx, cy
    )
    
    # 创建深度图像对象
    depth_o3d = o3d.geometry.Image(depth_image)
    
    # 将深度图转换为点云
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic,
        depth_scale=1.0,
        depth_trunc=max_depth,
        stride=20  # 保持这个值以维持点云密度
    )
    
    # 移除无效点
    pcd = pcd.remove_non_finite_points()
    
    # 更严格的离群点移除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    pcd = pcd.select_by_index(ind)
    
    # 如果点数仍然太多，进行均匀下采样
    if len(pcd.points) > 5000:
        pcd = pcd.voxel_down_sample(voxel_size=0.03)  # 保持这个值以维持点云密度
    
    # 打印点云数量
    print(f"点云中的点数量: {len(pcd.points)}")
    
    # 估计法向量
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)  # 调整搜索半径
    )
    
    # 保存点云
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到: {output_path}")
    
    # 可视化点云
    print("正在显示点云预览...")
    print("操作说明：")
    print("- 按住鼠标左键：旋转视图")
    print("- 按住鼠标右键：平移视图")
    print("- 滚动鼠标滚轮：缩放")
    print("- 按 'H' 键：显示帮助信息")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云预览", width=1280, height=720)
    vis.add_geometry(pcd)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([0, 0, 0])  # 黑色背景
    
    # 设置初始视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 实际的深度图路径
    depth_image_path = "E:/Dev/MiDaS-master/output/depth_map.png"
    output_path = "E:/Dev/MiDaS-master/output/point_cloud.ply"
    cameras_file = "E:/Dev/MiDaS-master/output/cameras.txt"
    points3d_file = "E:/Dev/MiDaS-master/output/points3D.txt"
    
    depth_to_pointcloud(depth_image_path, output_path, cameras_file, points3d_file)