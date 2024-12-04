import sys
import os
import subprocess
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import numpy as np
import open3d as o3d
from PIL import Image

class ReconstructionWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_folder, output_folder=None, use_dense=True, use_texture=True, use_deeplearning=True):
        super().__init__()
        self.input_folder = input_folder
        self.use_dense = use_dense
        self.use_texture = use_texture
        self.use_deeplearning = use_deeplearning
        self.is_running = False
        
        # 设置输出目录
        self.output_folder = output_folder if output_folder else os.path.join(self.input_folder, "reconstruction_output")
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 中间文件目录
        self.temp_dir = os.path.join(self.output_folder, ".temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # COLMAP中间文件目录
        self.colmap_dir = os.path.join(self.temp_dir, "colmap")
        os.makedirs(self.colmap_dir, exist_ok=True)
        self.sparse_dir = os.path.join(self.colmap_dir, "sparse")
        os.makedirs(self.sparse_dir, exist_ok=True)
        self.database_path = os.path.join(self.colmap_dir, "database.db")
        
        # OpenMVS输出目录
        self.openmvs_output = os.path.join(self.temp_dir, "openmvs")
        os.makedirs(self.openmvs_output, exist_ok=True)

        # MiDaS输出目录
        self.midas_output = os.path.join(self.temp_dir, "midas_depth")
        os.makedirs(self.midas_output, exist_ok=True)
        
        # 点云输出目录
        self.pointcloud_dir = os.path.join(self.output_folder, "pointcloud")
        os.makedirs(self.pointcloud_dir, exist_ok=True)

    def setup_midas_environment(self):
        """设置MiDaS环境"""
        try:
            midas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "MiDaS")
            if not os.path.exists(midas_path):
                raise Exception("MiDaS directory not found in tools folder")

            env_file = os.path.join(midas_path, "environment.yaml")
            if not os.path.exists(env_file):
                raise Exception("environment.yaml not found in MiDaS directory")

            # 检查midas-py310环境是否存在
            result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
            if "midas-py310" not in result.stdout:
                self.log.emit("Creating MiDaS conda environment...")
                subprocess.run(["conda", "env", "create", "-f", env_file], 
                             cwd=midas_path, check=True)
                self.log.emit("MiDaS conda environment created successfully")
            else:
                self.log.emit("MiDaS conda environment already exists")

            return midas_path
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to setup MiDaS environment: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during MiDaS setup: {str(e)}")

    def run_midas(self):
        self.log.emit("Running MiDaS depth estimation...")
        try:
            # 设置MiDaS环境
            midas_path = self.setup_midas_environment()
            midas_script = os.path.join(midas_path, "run.py")
            
            if not os.path.exists(midas_script):
                raise Exception("MiDaS run.py not found. Please check the installation.")

            self.log.emit(f"Using MiDaS model: dpt_swin2_large_384")
            
            # 构建运行命令
            # 在Windows上，我们需要使用conda run命令在特定环境中执行Python脚本
            command = [
                "conda", "run", "-n", "midas-py310",
                "python", midas_script,
                "--model_type", "dpt_swin2_large_384",
                "--input_path", self.input_folder,
                "--output_path", self.midas_output
            ]
            
            # 在MiDaS目录下执行命令
            subprocess.run(command, cwd=midas_path, check=True)

            # 验证输出
            if not os.path.exists(self.midas_output) or not os.listdir(self.midas_output):
                raise Exception("MiDaS did not generate any output files")

            self.log.emit("MiDaS depth estimation completed successfully")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"MiDaS processing failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during MiDaS processing: {str(e)}")

    def depth_to_pointcloud(self, depth_path, output_path, cameras_file=None):
        """将深度图转换为点云"""
        self.log.emit(f"Converting depth map to point cloud: {os.path.basename(depth_path)}")
        try:
            # 读取深度图
            depth_image = np.array(Image.open(depth_path))
            if len(depth_image.shape) == 3:
                depth_image = depth_image.mean(axis=2)
            depth_image = depth_image.astype(np.float32)
            
            # MiDaS深度图处理
            depth_image = depth_image.max() - depth_image
            
            # 深度范围映射参数
            min_depth = 1.0
            max_depth = 5.25
            
            # 非线性深度映射
            normalized_depth = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            alpha = 3.75
            beta = 0.625
            sigmoid_depth = 1 / (1 + np.exp(-alpha * (normalized_depth - beta)))
            depth_image = min_depth + (max_depth - min_depth) * sigmoid_depth
            
            # 应用深度校正因子
            depth_scale = 0.375
            depth_image = depth_image * depth_scale
            depth_image = np.ascontiguousarray(depth_image)
            
            # 设置相机参数
            width, height = depth_image.shape[1], depth_image.shape[0]
            if cameras_file and os.path.exists(cameras_file):
                with open(cameras_file, 'r') as f:
                    for line in f:
                        if not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 8:
                                fx = float(parts[4])
                                fy = float(parts[4])
                                cx = float(parts[5])
                                cy = float(parts[6])
                                break
            else:
                fx = fy = 3458.0
                cx = 1736.0
                cy = 2312.0
            
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, fx, fy, cx, cy
            )
            
            # 创建深度图像对象
            depth_o3d = o3d.geometry.Image(depth_image)
            
            # 转换为点云
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic,
                depth_scale=1.0,
                depth_trunc=max_depth,
                stride=20
            )
            
            # 处理点云
            pcd = pcd.remove_non_finite_points()
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
            pcd = pcd.select_by_index(ind)
            
            if len(pcd.points) > 5000:
                pcd = pcd.voxel_down_sample(voxel_size=0.03)
            
            # 估计法向量
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # 保存点云
            o3d.io.write_point_cloud(output_path, pcd)
            self.log.emit(f"Point cloud saved to: {output_path}")
            
        except Exception as e:
            raise Exception(f"Error converting depth map to point cloud: {str(e)}")

    def process_depth_maps(self):
        """处理所有深度图并生成点云"""
        try:
            depth_files = [f for f in os.listdir(self.midas_output) if f.endswith('.png')]
            if not depth_files:
                raise Exception("No depth maps found in MiDaS output directory")
            
            cameras_file = os.path.join(self.colmap_dir, "cameras.txt")
            
            for depth_file in depth_files:
                depth_path = os.path.join(self.midas_output, depth_file)
                output_path = os.path.join(self.pointcloud_dir, f"{os.path.splitext(depth_file)[0]}_cloud.ply")
                self.depth_to_pointcloud(depth_path, output_path, cameras_file)
                
            self.log.emit("All depth maps processed successfully")
            
        except Exception as e:
            raise Exception(f"Error processing depth maps: {str(e)}")

    def run_colmap_feature_extraction(self):
        self.log.emit("Running COLMAP feature extraction...")
        try:
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", self.database_path,
                "--image_path", self.input_folder,
                "--ImageReader.camera_model", "SIMPLE_RADIAL",
                "--ImageReader.single_camera", "1"
            ], check=True)
            self.log.emit("COLMAP feature extraction completed")
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP feature extraction failed: {str(e)}")

    def run_colmap_matcher(self):
        self.log.emit("Running COLMAP matcher...")
        try:
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", self.database_path
            ], check=True)
            self.log.emit("COLMAP matching completed")
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP matching failed: {str(e)}")

    def run_colmap_mapper(self):
        self.log.emit("Running COLMAP mapper...")
        try:
            subprocess.run([
                "colmap", "mapper",
                "--database_path", self.database_path,
                "--image_path", self.input_folder,
                "--output_path", self.sparse_dir
            ], check=True)
            self.log.emit("COLMAP mapping completed")
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP mapping failed: {str(e)}")

    def run_colmap_model_converter(self):
        self.log.emit("Converting COLMAP model...")
        try:
            # 找到最新的重建结果
            reconstruction_paths = [d for d in os.listdir(self.sparse_dir) 
                                 if os.path.isdir(os.path.join(self.sparse_dir, d))]
            if not reconstruction_paths:
                raise Exception("No COLMAP reconstruction found")
            
            latest_reconstruction = os.path.join(self.sparse_dir, sorted(reconstruction_paths)[-1])
            
            # 转换为TXT格式
            subprocess.run([
                "colmap", "model_converter",
                "--input_path", latest_reconstruction,
                "--output_path", self.colmap_dir,
                "--output_type", "TXT"
            ], check=True)
            
            self.log.emit("COLMAP model conversion completed")
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP model conversion failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during model conversion: {str(e)}")

    def convert_to_mvs(self):
        self.log.emit("Converting to MVS format...")
        try:
            # 检查必要的文件是否存在
            required_files = ["cameras.txt", "images.txt", "points3D.txt"]
            for file in required_files:
                if not os.path.exists(os.path.join(self.colmap_dir, file)):
                    raise Exception(f"Required file {file} not found")

            # 转换为MVS格式
            mvs_scene = os.path.join(self.openmvs_output, "scene.mvs")
            subprocess.run([
                "OpenMVS/InterfaceCOLMAP",
                "--input-path", self.colmap_dir,
                "--output-scene", mvs_scene,
                "--working-folder", self.openmvs_output
            ], check=True)
            
            self.log.emit("MVS conversion completed")
            return mvs_scene
        except subprocess.CalledProcessError as e:
            raise Exception(f"MVS conversion failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during MVS conversion: {str(e)}")

    def run(self):
        self.is_running = True
        try:
            self.progress.emit(0)
            self.log.emit("Starting reconstruction process...")

            # 1. 如果启用深度学习，运行MiDaS
            if self.use_deeplearning:
                self.run_midas()
                if not self.is_running:
                    return
                self.progress.emit(15)
                
                # 处理深度图生成点云
                self.process_depth_maps()
                if not self.is_running:
                    return
                self.progress.emit(25)

            # 2. 运行COLMAP特征提取
            self.run_colmap_feature_extraction()
            if not self.is_running:
                return
            self.progress.emit(40)

            # 3. 运行COLMAP特征匹配
            self.run_colmap_matcher()
            if not self.is_running:
                return
            self.progress.emit(55)

            # 4. 运行COLMAP稀疏重建
            self.run_colmap_mapper()
            if not self.is_running:
                return
            self.progress.emit(70)

            # 5. 如果启用密集重建，运行COLMAP密集重建
            if self.use_dense:
                self.run_colmap_dense()
                if not self.is_running:
                    return
                self.progress.emit(85)

            # 6. 运行OpenMVS重建
            self.run_openmvs()
            if not self.is_running:
                return
            self.progress.emit(95)
            
            # 7. 如果启用深度学习，融合点云
            if self.use_deeplearning:
                self.merge_all_point_clouds()
                if not self.is_running:
                    return

            self.progress.emit(100)
            self.log.emit("Reconstruction completed successfully!")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()
        finally:
            self.is_running = False

    def run_colmap_image_undistorter(self):
        self.log.emit("Preparing dense reconstruction input data...")
        try:
            dense_path = os.path.join(self.output_folder, "dense")
            os.makedirs(dense_path, exist_ok=True)
            sparse_path = os.path.join(self.sparse_dir, "0")
            
            subprocess.run([
                "colmap", "image_undistorter",
                "--image_path", self.input_folder,
                "--input_path", sparse_path,
                "--output_path", dense_path,
                "--output_type", "COLMAP"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP image undistorter failed: {str(e)}")

    def run_colmap_patch_match_stereo(self):
        self.log.emit("Running dense point cloud generation...")
        try:
            dense_path = os.path.join(self.output_folder, "dense")
            subprocess.run([
                "colmap", "patch_match_stereo",
                "--workspace_path", dense_path,
                "--workspace_format", "COLMAP",
                "--PatchMatchStereo.geom_consistency", "true"
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP patch match stereo failed: {str(e)}")

    def run_colmap_stereo_fusion(self):
        self.log.emit("Running point cloud fusion...")
        try:
            dense_path = os.path.join(self.output_folder, "dense")
            subprocess.run([
                "colmap", "stereo_fusion",
                "--workspace_path", dense_path,
                "--workspace_format", "COLMAP",
                "--input_type", "geometric",
                "--output_path", os.path.join(dense_path, "fused.ply")
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP stereo fusion failed: {str(e)}")

    def run_colmap_poisson_mesher(self):
        self.log.emit("Generating mesh model...")
        try:
            dense_path = os.path.join(self.output_folder, "dense")
            subprocess.run([
                "colmap", "poisson_mesher",
                "--input_path", os.path.join(dense_path, "fused.ply"),
                "--output_path", os.path.join(dense_path, "meshed.ply")
            ], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"COLMAP poisson mesher failed: {str(e)}")

    def run_openmvs_texturing(self):
        self.log.emit("Starting OpenMVS processing...")
        try:
            # 1. 转换为MVS格式
            self.log.emit("Converting to MVS format...")
            scene_mvs = self.convert_to_mvs()
            
            # 2. 网格重建
            self.log.emit("Running mesh reconstruction...")
            mesh_mvs = os.path.join(self.openmvs_output, "mesh.mvs")
            subprocess.run([
                "OpenMVS/ReconstructMesh",
                "-i", scene_mvs,
                "-o", mesh_mvs,
                "--image-folder", self.input_folder
            ], check=True)

            # 3. 纹理映射
            self.log.emit("Running texture mapping...")
            subprocess.run([
                "OpenMVS/TextureMesh",
                "-i", mesh_mvs
            ], check=True)

            self.log.emit("OpenMVS processing completed successfully")
            
        except subprocess.CalledProcessError as e:
            raise Exception(f"OpenMVS processing failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during OpenMVS processing: {str(e)}")

    def merge_all_point_clouds(self):
        """融合MiDaS和OpenMVS生成的点云"""
        self.log.emit("开始融合点云...")
        try:
            # 获取所有MiDaS生成的点云文件
            midas_clouds = [f for f in os.listdir(self.pointcloud_dir) if f.endswith('_cloud.ply')]
            if not midas_clouds:
                raise Exception("未找到MiDaS生成的点云文件")
            
            # 获取OpenMVS生成的点云文件
            openmvs_cloud = os.path.join(self.openmvs_output, "scene_dense.ply")
            if not os.path.exists(openmvs_cloud):
                raise Exception("未找到OpenMVS生成的点云文件")
            
            # 读取OpenMVS点云
            target_pcd = o3d.io.read_point_cloud(openmvs_cloud)
            if len(target_pcd.points) == 0:
                raise Exception("OpenMVS点云为空")
            
            self.log.emit(f"OpenMVS点云包含 {len(target_pcd.points)} 个点")
            
            # 存储所有配准后的点云
            aligned_clouds = []
            voxel_size = 0.05  # 初始体素大小
            
            # 读取并配准每个MiDaS点云
            for cloud_file in midas_clouds:
                try:
                    source_path = os.path.join(self.pointcloud_dir, cloud_file)
                    source_pcd = o3d.io.read_point_cloud(source_path)
                    if len(source_pcd.points) == 0:
                        self.log.emit(f"跳过空点云: {cloud_file}")
                        continue
                    
                    self.log.emit(f"处理点云 {cloud_file} (包含 {len(source_pcd.points)} 个点)")
                        
                    # 预处理点云
                    source_pcd_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
                    target_pcd_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
                    
                    self.log.emit(f"下采样后点数: {len(source_pcd_down.points)}")
                    
                    # 粗配准
                    result_icp = global_registration_with_icp(source_pcd, target_pcd, voxel_size)
                    
                    # 评估粗配准结果
                    fitness = result_icp.fitness
                    rmse = result_icp.inlier_rmse
                    self.log.emit(f"粗配准结果 - 匹配度: {fitness:.4f}, RMSE: {rmse:.4f}")
                    
                    # 精细配准
                    result_refined = refine_registration_with_adaptive_threshold(
                        source_pcd, target_pcd, result_icp.transformation, voxel_size
                    )
                    
                    # 评估精细配准结果
                    fitness_refined = result_refined.fitness
                    rmse_refined = result_refined.inlier_rmse
                    self.log.emit(f"精细配准结果 - 匹配度: {fitness_refined:.4f}, RMSE: {rmse_refined:.4f}")
                    
                    # 转换点云
                    transformed_pcd = source_pcd.transform(result_refined.transformation)
                    aligned_clouds.append(transformed_pcd)
                    self.log.emit(f"成功配准点云: {cloud_file}")
                    
                except Exception as e:
                    self.log.emit(f"处理点云 {cloud_file} 时出错: {str(e)}")
                    continue
            
            if not aligned_clouds:
                raise Exception("没有可用的配准点云")
            
            # 添加目标点云
            aligned_clouds.append(target_pcd)
            
            # 合并所有点云
            output_path = os.path.join(self.output_folder, "final_merged_cloud.ply")
            merged_pcd = merge_point_clouds(aligned_clouds, output_path)
            self.log.emit(f"合并后点云包含 {len(merged_pcd.points)} 个点")
            
            # 评估融合结果
            evaluation_results = evaluate_point_clouds(merged_pcd, target_pcd, voxel_size)
            
            # 输出评估结果
            self.log.emit("\n点云融合评估结果:")
            self.log.emit(f"RMSE: {evaluation_results['rmse']:.5f}")
            self.log.emit(f"Hausdorff距离: {evaluation_results['hausdorff']:.5f}")
            self.log.emit(f"重叠度: {evaluation_results['overlap']*100:.2f}%")
            self.log.emit(f"点云密度比: {evaluation_results['density_ratio']:.2f}")
            self.log.emit(f"点云分布均匀度: {evaluation_results['uniformity']:.2f}")
            
            # 保存评估报告
            report_path = os.path.join(self.output_folder, "point_cloud_evaluation.txt")
            with open(report_path, 'w') as f:
                f.write("点云融合评估报告\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"OpenMVS点云点数: {len(target_pcd.points)}\n")
                f.write(f"融合后点云点数: {len(merged_pcd.points)}\n")
                f.write(f"RMSE: {evaluation_results['rmse']:.5f}\n")
                f.write(f"Hausdorff距离: {evaluation_results['hausdorff']:.5f}\n")
                f.write(f"重叠度: {evaluation_results['overlap']*100:.2f}%\n")
                f.write(f"点云密度比: {evaluation_results['density_ratio']:.2f}\n")
                f.write(f"点云分布均匀度: {evaluation_results['uniformity']:.2f}\n")
            
            self.log.emit(f"评估报告已保存至: {report_path}")
            self.log.emit("点云融合完成")
            return True
            
        except Exception as e:
            self.error.emit(f"点云融合失败: {str(e)}")
            return False

    def stop(self):
        self.is_running = False
        self.log.emit("Reconstruction process stopped by user.")

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, fpfh

def global_registration_with_icp(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    distance_threshold = voxel_size * 1.5
    result_icp = o3d.registration.registration_icp(
        source_down, target_down, distance_threshold,
        np.eye(4),
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result_icp

def refine_registration_with_adaptive_threshold(source, target, transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    result_refined = o3d.registration.registration_icp(
        source, target, distance_threshold,
        transformation,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result_refined

def merge_point_clouds(aligned_clouds, output_path):
    merged_pcd = aligned_clouds[0]
    for cloud in aligned_clouds[1:]:
        merged_pcd += cloud
    o3d.io.write_point_cloud(output_path, merged_pcd)
    return merged_pcd

def evaluate_point_clouds(merged_pcd, target_pcd, voxel_size):
    """评估点云融合结果"""
    # 计算点云距离
    distances = merged_pcd.compute_point_cloud_distance(target_pcd)
    distances = np.asarray(distances)
    
    # 计算RMSE
    rmse = np.sqrt(np.mean(distances**2))
    
    # 计算Hausdorff距离
    hausdorff = np.max(distances)
    
    # 计算重叠度
    overlap_threshold = voxel_size * 2
    overlap = np.mean(distances < overlap_threshold)
    
    # 计算点云密度比
    density_ratio = len(merged_pcd.points) / len(target_pcd.points)
    
    # 计算点云分布均匀度
    merged_pcd_down = merged_pcd.voxel_down_sample(voxel_size)
    uniformity = len(merged_pcd_down.points) / len(merged_pcd.points)
    
    return {
        'rmse': rmse,
        'hausdorff': hausdorff,
        'overlap': overlap,
        'density_ratio': density_ratio,
        'uniformity': uniformity
    }
