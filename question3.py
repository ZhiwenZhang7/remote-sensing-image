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
                    break
            if font_found:
                break
        except Exception:
            continue
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
set_chinese_font()
from skimage import filters, morphology, segmentation

def vegetation_extraction():
    """
    第三问：绿色植物信息提取
    从LANDSAT8图像中提取植被信息
    """
    
    # 1. 读取LANDSAT8数据
    landsat_path = "LC81310432021034LGN00"  # 文件夹名
    bands = []
    band_names = ['B2', 'B3', 'B4', 'B5']  # 蓝、绿、红、近红外
    band_files = {}
    import glob
    # 自动递归查找波段文件，支持不同命名格式
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
                bands.append(band_data.astype(float))
    
    if len(bands) < 4:
        print("错误：缺少必要的波段数据，请检查LC81310432021034LGN00目录下是否有B2、B3、B4、B5的TIF文件。")
        return None
    
    blue, green, red, nir = bands
    # 裁剪中心5km×5km区域（30米分辨率，约167像素）
    crop_size = 167
    height, width = blue.shape
    center_y, center_x = height // 2, width // 2
    half_size = crop_size // 2
    start_y = max(0, center_y - half_size)
    end_y = min(height, center_y + half_size)
    if end_y - start_y < crop_size:
        end_y = min(height, start_y + crop_size)
    start_x = max(0, center_x - half_size)
    end_x = min(width, center_x + half_size)
    if end_x - start_x < crop_size:
        end_x = min(width, start_x + crop_size)
    # 裁剪所有波段
    blue = blue[start_y:end_y, start_x:end_x]
    green = green[start_y:end_y, start_x:end_x]
    red = red[start_y:end_y, start_x:end_x]
    nir = nir[start_y:end_y, start_x:end_x]
    
    # 2. 计算植被指数
    def calculate_ndvi(nir, red):
        """计算归一化植被指数"""
        ndvi = (nir - red) / (nir + red + 1e-10)
        return ndvi
    
    def calculate_evi(nir, red, blue):
        """计算增强型植被指数"""
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
        return evi
    
    def calculate_savi(nir, red, L=0.5):
        """计算土壤调节植被指数"""
        savi = (nir - red) / (nir + red + L) * (1 + L)
        return savi
    
    # 2. 计算NDVI
    ndvi = (nir - red) / (nir + red + 1e-10)
    # 3. 植被掩膜（简单阈值法）
    vegetation_mask = ndvi > 0.3
    # 4. 显示原始真彩色和植被掩膜
    rgb_image = np.stack([red, green, blue], axis=-1)
    def norm_rgb(img):
        img = img.astype(float)
        for i in range(img.shape[-1]):
            band = img[..., i]
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)
            if max_val > min_val:
                img[..., i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)
            else:
                img[..., i] = 0
        return img
    plt.figure(figsize=(6,6))
    plt.imshow(norm_rgb(rgb_image))
    plt.title('原始真彩色图像')
    plt.axis('off')
    plt.savefig('output_rgb.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(6,6))
    plt.imshow(vegetation_mask, cmap='Greens')
    plt.title('绿色植物掩膜')
    plt.axis('off')
    plt.savefig('output_vegetation_mask.png', dpi=300, bbox_inches='tight')
    plt.close()
    return vegetation_mask, ndvi


# 运行第三问
if __name__ == "__main__":
    vegetation_mask, ndvi = vegetation_extraction()