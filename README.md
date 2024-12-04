# 3D重建GUI应用

一个基于PyQt5的Windows桌面应用程序，用于从图像集生成高质量的3D模型。该应用程序集成了COLMAP、OpenMVS和MiDaS等先进的3D重建工具，提供了直观的用户界面和完整的重建流程。

## 功能特点

- 图像处理和3D重建
  * 支持批量图像输入
  * COLMAP特征提取和匹配
  * 稀疏和密集重建
  * OpenMVS网格重建和纹理映射
  * MiDaS深度估计
  * 点云融合和优化

- 深度学习集成
  * MiDaS深度图生成
  * 深度图到点云转换
  * 多源点云融合
  * 点云质量评估

- 用户界面
  * 直观的文件夹选择
  * 实时进度跟踪
  * 详细的处理日志
  * 可配置的重建选项
  * 错误提示和处理

## 系统要求

- 操作系统：Windows 10或更高版本
- Python 3.6+
- CUDA支持的GPU（推荐）
- 足够的磁盘空间用于中间文件

## 依赖项

主要依赖：
```
PyQt5
numpy
open3d
pillow
torch
```

外部工具：
- COLMAP
- OpenMVS
- MiDaS

## 安装说明

1. 克隆仓库：
```bash
git clone [repository-url]
cd 3DGUI
```

2. 安装Python依赖：
```bash
pip install -r requirements.txt
```

3. 安装外部工具：
   - 安装COLMAP并添加到系统PATH
   - 安装OpenMVS并添加到系统PATH
   - MiDaS将在首次运行时自动下载

## 使用方法

1. 启动应用程序：
```bash
python main.py
```

2. 选择输入图像文件夹
3. 选择输出目录（可选）
4. 配置重建选项：
   - 密集重建开关
   - 纹理重建开关
   - 深度学习集成开关
5. 点击"开始重建"按钮

## 输出文件

- `.temp/`：中间文件目录
  * `colmap/`：COLMAP处理文件
  * `openmvs/`：OpenMVS处理文件
  * `midas_depth/`：MiDaS深度图
- `pointcloud/`：点云文件
  * 各视角点云（`*_cloud.ply`）
  * 融合点云（`final_merged_cloud.ply`）
- `point_cloud_evaluation.txt`：点云评估报告
- 最终3D模型和纹理

## 点云评估指标

- RMSE（均方根误差）
- Hausdorff距离
- 点云重叠度
- 点云密度比
- 分布均匀度

## 注意事项

1. 图像要求：
   - 清晰、高质量的图像
   - 足够的重叠度
   - 适当的光照条件

2. 性能考虑：
   - 密集重建需要较大计算资源
   - 建议使用GPU加速
   - 预留足够磁盘空间

3. 最佳实践：
   - 使用20-100张图像
   - 保持图像间30-60%重叠
   - 避免反光和运动模糊

## 高级点云处理

### 点云处理功能

- 深度图转点云转换
- 多阶段点云配准
  * 全局RANSAC配准
  * 精细ICP配准优化
- 智能点云合并
  * 点云归一化
  * 自适应体素下采样
  * 智能颜色转移
- 点云质量评估
  * RMSE评估
  * 点云密度分析
  * 分布均匀性检查

### 点云处理参数

- `voxel_size`: 体素下采样大小
- `distance_threshold`: 配准距离阈值
- `ransac_n`: RANSAC采样点数
- `num_iterations`: ICP迭代次数

## 故障排除

常见问题：
1. COLMAP/OpenMVS未找到
   - 检查PATH环境变量
   - 确认工具安装正确

2. 内存不足
   - 减少输入图像数量
   - 关闭密集重建选项
   - 增加系统虚拟内存

3. 重建质量问题
   - 检查图像质量
   - 调整重建参数
   - 确保图像重叠充分

## 开发说明

主要模块：
- `main.py`：GUI和用户交互
- `reconstruction.py`：重建核心逻辑
- `ComEva.py`：点云评估工具
- `pointcloud_processing.py`：点云处理功能

## 许可证

[许可证信息]

## 贡献指南

欢迎提交问题和改进建议！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 联系方式

[联系信息]
