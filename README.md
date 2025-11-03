# 遥感图像处理作业汇总

本仓库包含四个独立的脚本，分别演示遥感影像的辐射/大气校正（Question1）、图像增强与 PCA 融合（Question2）、植被提取（Question3）以及图像平滑/分割/形态学处理（Question4）。下面给出每个脚本的功能、运行示例、主要参数与输出说明，以及环境依赖和运行建议。

---

## 一、整体依赖（建议使用 conda-forge）
- Python 3.8–3.11（推荐 3.10）
- numpy, scipy, scikit-image, scikit-learn, matplotlib, pillow/imageio
- GDAL（仅 Question1/Question2/Question3 处理 TIF/HDF 时需要）

推荐使用 conda：
```powershell
conda create -n py6s-env python=3.10 -y
conda activate py6s-env
conda install -c conda-forge numpy scipy scikit-image scikit-learn matplotlib pillow imageio -y
conda install -c conda-forge gdal -y  # 若需处理遥感 TIF/HDF
```

---

## 二、脚本概览与运行示例

所有示例均在 Windows PowerShell 下给出。将命令中的路径替换为你的文件路径。

1) `question1.py` — 辐射定标与 6S 大气校正（针对 Landsat）
- 功能：对文件夹中波段进行辐射校正并调用 Py6S 进行大气校正，支持从 MOD04 HDF 提取 AOD 覆盖默认值。
- 主要参数：`--Input_dir`（输入根目录）, `--Output_dir`（输出目录）, `--MOD04`（可选 MOD04 HDF 路径）。
- 运行示例：
```powershell
python .\question1.py --Input_dir "D:\dataset\LC08_folder" --Output_dir "D:\output\question1" --MOD04 "D:\MOD04\MOD04_example.hdf"
```

2) `question2.py` — 图像增强与 PCA 全色锐化（Pansharpening）
- 功能：裁切中心约 5km×5km，线性拉伸、直方图均衡、PCA 计算并用全色波段替换第一主成分以实现融合。
- 输出：多张 PNG（原始全色、线性拉伸、直方图均衡、多光谱 RGB、PC1、PCA 融合、PCA 方差）。
- 运行示例（脚本会自动查找文件夹中的波段）：
```powershell
python .\question2.py
```

3) `question3.py` — 植被提取（NDVI 掩膜）
- 功能：读取 B2/B3/B4/B5，裁剪中心 5km×5km，计算 NDVI（以及 EVI/SAVI 函数），用阈值 NDVI>0.3 生成植被掩膜并保存两张 PNG。
- 运行示例：
```powershell
python .\question3.py
```

4) `question4.py` — 图像平滑 / 分割 / 形态学处理
- 功能：对单张图像进行多种平滑比较（均值、中值、梯度倒数加权、FFT/高斯低通）、多种分割（Otsu、自适应、边缘填充、KMeans、Watershed）及形态学处理，并保存对比图。
- 主要参数：`--input`/`-i`（输入图像），`--out`/`-o`（输出目录，默认 `question4 answer`）。
- 运行示例：
```powershell
python .\question4.py -i .\q4\R.png -o .\question4_output
```

---

## 三、输出文件说明（以 `question4.py` 为例）
- `{base}_smoothing_compare.png`：平滑方法对比网格图。
- `{base}_smooth_mean.png` / `_median.png` / `_gradinv.png` / `_fft.png`：各平滑方法单图。
- `{base}_segmentation_compare.png`：分割方法对比（Otsu、Adaptive、Edges、KMeans、Watershed 等）。
- `{base}_otsu.png`、`{base}_kmeans.png`、`{base}_watershed_mask.png` 等：分割单图。
- `{base}_otsu_opening.png` / `_closing.png` / `_dilate.png` / `_erode.png`：形态学结果。

`question2.py` 和 `question3.py` 也会生成 PNG（展示真彩/掩膜/融合/PC1 等），`question1.py` 输出为经大气校正后的 GeoTIFF（写入指定输出目录）。

---

## 四、验证安装（快捷）
在 PowerShell 中运行：
```powershell
python --version
python -c "import numpy, scipy, skimage, sklearn, matplotlib; print(numpy.__version__, scipy.__version__, skimage.__version__, sklearn.__version__, matplotlib.__version__)"
python -c "from osgeo import gdal; print('GDAL', gdal.__version__)"  # 若安装了 GDAL
```

---

## 五、注意事项与建议
- 若只运行 `question4.py`，不需要安装 GDAL；若需处理 Landsat TIF/HDF，请使用 conda 安装 GDAL（conda-forge）。
- 若运行 `question1.py`，需安装并配置 Py6S（并确认 6S 模型可用）。

---

