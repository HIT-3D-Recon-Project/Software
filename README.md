# 3D Reconstruction GUI

一个基于Windows平台的交互式三维重建程序，集成了MiDaS、COLMAP和OpenMVS等工具。

## 功能特点

- 支持从图像文件夹生成三维模型
- 可选的纹理复原功能
- 可选的深度学习增强（使用MiDaS）
- 实时进度显示和日志输出
- 可中断的处理流程

## 系统要求

- Windows 10 或更高版本
- Python 3.6+
- COLMAP
- OpenMVS
- CUDA支持的GPU（推荐）

## 安装步骤

1. 克隆或下载本仓库
2. 安装Python依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 安装并配置外部工具：
   - 安装COLMAP并添加到系统PATH
   - 安装OpenMVS并添加到系统PATH
   - （可选）安装CUDA以获得更好的性能

## 使用方法

1. 运行程序：
   ```bash
   python main.py
   ```
2. 点击"Select Image Folder"选择包含图像的文件夹
3. 根据需要选择是否启用纹理复原和深度学习
4. 点击"Start Reconstruction"开始重建
5. 等待处理完成，结果将保存在输入文件夹的"reconstruction_output"子目录中

## 输出文件

- `database.db`：COLMAP数据库文件
- `sparse/`：稀疏重建结果
- `dense/`：密集重建结果（如果启用）
- `textured/`：带纹理的模型（如果启用）

## 注意事项

- 输入图像应该具有足够的重叠度
- 建议使用清晰、光照均匀的图像
- 处理时间取决于图像数量和分辨率
- 确保有足够的磁盘空间存储中间文件和结果

## 故障排除

如果遇到问题：
1. 检查是否正确安装了所有依赖
2. 确认COLMAP和OpenMVS已添加到系统PATH
3. 查看程序日志了解详细错误信息
4. 确保输入图像格式正确（支持jpg、jpeg、png）

