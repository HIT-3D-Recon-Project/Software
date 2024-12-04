import os
import subprocess
from PyQt5.QtCore import QObject, pyqtSignal
import time

class ReconstructionWorker(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_folder, output_folder=None, use_texture=True, use_deeplearning=True):
        super().__init__()
        self.input_folder = input_folder
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
        self.sparse_dir = os.path.join(self.colmap_dir, "sparse")  # COLMAP稀疏重建结果目录
        os.makedirs(self.sparse_dir, exist_ok=True)
        self.database_path = os.path.join(self.colmap_dir, "database.db")  # COLMAP数据库文件
        
        # OpenMVS输出目录
        self.openmvs_output = os.path.join(self.temp_dir, "openmvs")
        os.makedirs(self.openmvs_output, exist_ok=True)

        # MiDaS输出目录
        self.midas_output = os.path.join(self.temp_dir, "midas_depth")
        os.makedirs(self.midas_output, exist_ok=True)

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

            # 2. 运行COLMAP特征提取
            self.run_colmap_feature_extraction()
            if not self.is_running:
                return
            self.progress.emit(30)

            # 3. 运行COLMAP特征匹配
            self.run_colmap_matcher()
            if not self.is_running:
                return
            self.progress.emit(45)

            # 4. 运行COLMAP稀疏重建
            self.run_colmap_mapper()
            if not self.is_running:
                return
            self.progress.emit(55)

            # 4.1 生成TXT文件，为OpenMVS做准备
            self.run_colmap_model_converter()
            if not self.is_running:
                return
            self.progress.emit(60)

            # 5. 运行图像稠密匹配准备
            self.run_colmap_image_undistorter()
            if not self.is_running:
                return
            self.progress.emit(70)

            # 6. 运行密集点云生成
            self.run_colmap_patch_match_stereo()
            if not self.is_running:
                return
            self.progress.emit(80)

            # 7. 运行点云融合
            self.run_colmap_stereo_fusion()
            if not self.is_running:
                return
            self.progress.emit(90)

            # 8. 运行网格生成
            self.run_colmap_poisson_mesher()
            if not self.is_running:
                return
            self.progress.emit(95)

            # 9. 如果启用纹理，运行OpenMVS
            if self.use_texture:
                self.run_openmvs_texturing()
                if not self.is_running:
                    return

            self.progress.emit(100)
            self.log.emit("Reconstruction completed successfully!")
            
        except Exception as e:
            self.error.emit(f"Error during reconstruction: {str(e)}")
        finally:
            self.is_running = False
            self.finished.emit()

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

    def stop(self):
        self.is_running = False
        self.log.emit("Reconstruction process stopped by user.")
