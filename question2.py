# ==============================================================================
# 版权所有 (c) 2025 张智文
# 作者: 张智文
#
# 本文件为课程/作业代码，作者保留所有权利。代码可用于学习、教学与研究，
# 但未经作者许可不得用于商业目的或不当传播。若需商用或重新发布，请
# 联系作者取得书面授权。
#
# 许可：在不修改本文件版权声明的前提下，你可以自由复制、学习、研究
# 和改进本代码。如需更宽松的开源许可（如 MIT/BSD），可联系作者更改。
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from osgeo import gdal
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import glob
from skimage.transform import resize

def set_chinese_font():
    zh_fonts = ['SimHei', 'Microsoft YaHei', 'STSong', 'FangSong', 'KaiTi']
    font_found = False
    for font in zh_fonts:
        try:
            font_paths = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            for fp in font_paths:
                if font.lower() in fp.lower():
                    plt.rcParams['font.sans-serif'] = [font]
                    font_found = True
                    print(f"已设置中文字体: {font}")
                    break
            if font_found:
                break
        except Exception:
            continue
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['Arial']
        print("未找到可用中文字体，已设置为Arial。请安装SimHei或微软雅黑字体以支持中文显示。")
    plt.rcParams['axes.unicode_minus'] = False

set_chinese_font()

def image_enhancement():
    """
    第二问：遥感图像增强
    包括线性拉伸、直方图均衡化和主成分变换融合
    在图像中心裁剪5km*5km区域进行处理
    """
    
    # 1. 读取LANDSAT8数据
    landsat_path = "LC81190402021270LGN00"  # 实际文件夹名
    bands = []
    band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    band_files = {}
    # 只查找标准TIF文件，自动跳过.TIF.enp文件
    for band in band_names:
        pattern = os.path.join(landsat_path, f"**/*_{band}.TIF")
        files = glob.glob(pattern, recursive=True)
        if files:
            band_files[band] = files[0]
    for band in band_names:
        if band in band_files:
            ds = gdal.Open(band_files[band])
            if ds:
                band_data = ds.ReadAsArray()
                bands.append(band_data)
    if len(bands) < 8:
        return None

    # 2. 在图像中心裁剪5km×5km区域
    # LANDSAT8空间分辨率约为30米，5km对应约167个像素
    crop_size = 167  # 5000m / 30m ≈ 167 pixels
    
    # 获取图像尺寸
    height, width = bands[0].shape
    
    # 计算中心点坐标
    center_y, center_x = height // 2, width // 2
    half_size = crop_size // 2
    
    # 修正裁剪区域边界，确保尺寸为 crop_size
    start_y = max(0, center_y - half_size)
    end_y = min(height, center_y + half_size)
    if end_y - start_y < crop_size:
        end_y = min(height, start_y + crop_size)
    start_x = max(0, center_x - half_size)
    end_x = min(width, center_x + half_size)
    if end_x - start_x < crop_size:
        end_x = min(width, start_x + crop_size)

    # 裁剪所有波段
    bands_cropped = [band[start_y:end_y, start_x:end_x] for band in bands]
    # 改进归一化显示函数，自动按分位数拉伸，避免全为0或全白
    def norm_rgb(img):
        img = img.astype(float)
        # 按通道分别线性拉伸
        for i in range(img.shape[-1]):
            band = img[..., i]
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)
            if max_val > min_val:
                img[..., i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            else:
                img[..., i] = 0
        # 检查是否全为0
        if np.all(img == 0):
            maxv = np.max(img)
            if maxv > 0:
                img = img / maxv
        return img
    
    # 3. 全色波段线性拉伸和直方图均衡化
    panchromatic = bands_cropped[7]  # B8 - 全色波段
    multispectral_bands = bands_cropped[:7]  # B1-B7
    ms_shape = multispectral_bands[0].shape
    
    # 自动重采样全色波段到多光谱分辨率（如果需要）
    if panchromatic.shape != ms_shape:
        print(f"重采样全色波段：{panchromatic.shape} -> {ms_shape}")
        panchromatic_resampled = resize(
            panchromatic,
            ms_shape,
            order=1,  # 双线性插值
            preserve_range=True,
            anti_aliasing=True
        ).astype(panchromatic.dtype)
    else:
        panchromatic_resampled = panchromatic
    
    # 线性拉伸
    def linear_stretch(image, min_percent=2, max_percent=98):
        min_val = np.percentile(image, min_percent)
        max_val = np.percentile(image, max_percent)
        stretched = (image - min_val) / (max_val - min_val)
        return np.clip(stretched, 0, 1)
    
    pan_linear = linear_stretch(panchromatic_resampled)
    
    # 直方图均衡化
    def histeq_numpy(image):
        # 拉伸到0-255
        img = np.uint8(255 * (image - np.min(image)) / (np.max(image) - np.min(image)))
        # 计算直方图
        hist, bins = np.histogram(img.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_min = cdf_masked.min()
        cdf_max = cdf_masked.max()
        # 均衡化公式
        cdf_eq = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
        cdf_eq = np.ma.filled(cdf_eq, 0).astype('uint8')
        img_eq = cdf_eq[img]
        return img_eq.astype(float) / 255
    
    pan_histeq = histeq_numpy(panchromatic_resampled)
    
    # 4. 主成分变换 (1-7波段)
    ms_stack = np.stack(multispectral_bands, axis=-1)
    original_shape = ms_stack.shape
    
    # 重塑数据用于PCA
    pixels = ms_stack.reshape(-1, 7)
    
    # 标准化
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)
    
    # PCA变换
    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(pixels_scaled)
    
    # 重塑回图像形状
    pc_images = principal_components.reshape(original_shape[0], original_shape[1], 7)
    
    # ...existing code...
    
    # 5. 主成分变换融合 (用全色波段替换第一主成分)
    def pansharpening_pca(multispectral, panchromatic):
        """使用PCA方法进行全色锐化"""
        # 将多光谱数据重塑为2D
        ms_2d = multispectral.reshape(-1, multispectral.shape[2])
        
        # PCA变换
        pca = PCA(n_components=ms_2d.shape[1])
        ms_pca = pca.fit_transform(ms_2d)
        
        # 用全色波段替换第一主成分（需匹配直方图）
        pan_flat = panchromatic.flatten()
        # 直方图匹配
        pan_matched = hist_match(pan_flat, ms_pca[:, 0])
        ms_pca[:, 0] = pan_matched
        
        # 逆PCA变换
        fused_2d = pca.inverse_transform(ms_pca)
        fused = fused_2d.reshape(multispectral.shape)
        
        return fused
    
    def hist_match(source, template):
        """直方图匹配"""
        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()
        
        # 获取值的唯一集合和对应的索引
        s_values, s_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)
        
        # 计算累积分布函数
        s_quantiles = np.cumsum(s_counts).astype(float) / source.size
        t_quantiles = np.cumsum(t_counts).astype(float) / template.size
        
        # 插值
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        
        return interp_t_values[s_idx].reshape(oldshape)
    
    # 执行PCA融合
    # 构建原始多光谱RGB（B4,B3,B2），并修正数据类型
    ms_rgb = np.stack([multispectral_bands[3], multispectral_bands[2], multispectral_bands[1]], axis=-1).astype(float)
    fused_image = pansharpening_pca(ms_stack, panchromatic_resampled)
    fused_rgb = fused_image[:,:,[3,2,1]].astype(float)  # 提取融合后的RGB波段
    
    # 6. 结果单张输出
    output_dir = os.path.join(os.getcwd(), 'question2 answer')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 原始全色波段
    plt.figure(figsize=(6,6))
    plt.imshow(panchromatic_resampled, cmap='gray')
    plt.title('原始全色波段 (5km×5km)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_band8_panchromatic.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 线性拉伸结果
    plt.figure(figsize=(6,6))
    plt.imshow(pan_linear, cmap='gray')
    plt.title('线性拉伸')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_band8_linear.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 直方图均衡化结果
    plt.figure(figsize=(6,6))
    plt.imshow(pan_histeq, cmap='gray')
    plt.title('直方图均衡化')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_band8_histeq.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 原始多光谱图像
    plt.figure(figsize=(6,6))
    plt.imshow(norm_rgb(ms_rgb))
    plt.title('原始多光谱图像')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_multispectral_rgb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 第一主成分
    plt.figure(figsize=(6,6))
    plt.imshow(pc_images[:,:,0], cmap='gray')
    plt.title('第一主成分')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_pca_pc1.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. PCA融合结果
    plt.figure(figsize=(6,6))
    plt.imshow(norm_rgb(fused_rgb))
    plt.title('PCA融合图像')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'output_pca_fused_rgb.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 显示主成分解释方差
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 8), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.savefig(os.path.join(output_dir, 'output_pca_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ...existing code...
    return pan_linear, pan_histeq, pc_images, fused_image

# 运行第二问
if __name__ == "__main__":
    result = image_enhancement()