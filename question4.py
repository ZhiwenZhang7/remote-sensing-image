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



import os
import argparse
import numpy as np
from skimage import io, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu, sobel
from skimage.segmentation import watershed
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.morphology import opening, closing, dilation, erosion, disk
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter, binary_fill_holes
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

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

def imread(path):
        img = io.imread(path)
        if img.ndim == 2:
                return img
        if img.shape[-1] == 4:
                img = img[..., :3]
        return img

def imwrite(path, img):
        io.imsave(path, img_as_ubyte(img))

# 1. 平滑方法
def mean_blur(img, ksize=5):
    return uniform_filter(img, size=(ksize, ksize, 1) if img.ndim==3 else (ksize, ksize))

def median_blur(img, ksize=5):
    return median_filter(img, size=(ksize, ksize, 1) if img.ndim==3 else (ksize, ksize))

def gradient_inverse_weighted(img, ksize=7, beta=10.0):
    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.copy()
    gray_f = img_as_float(gray)
    gx = sobel(gray_f)
    gy = sobel(gray_f.T).T
    g = np.sqrt(gx*gx + gy*gy)
    g_norm = (g - g.min()) / (g.ptp() + 1e-12)
    w = 1.0 / (1.0 + beta * g_norm)
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            Ic = img[..., c].astype(np.float32)
            num = uniform_filter(Ic * w, size=ksize)
            den = uniform_filter(w, size=ksize)
            out[..., c] = num / (den + 1e-12)
        out = np.clip(out, 0, 255).astype(np.uint8)
    else:
        Ic = img.astype(np.float32)
        num = uniform_filter(Ic * w, size=ksize)
        den = uniform_filter(w, size=ksize)
        out = (num / (den + 1e-12)).clip(0,255).astype(np.uint8)
    return out

def fft_lowpass(img, cutoff=30):
    img = img_as_float(img)
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(3):
            out[..., c] = gaussian_filter(img[..., c], sigma=cutoff)
        return img_as_ubyte(out)
    else:
        return img_as_ubyte(gaussian_filter(img, sigma=cutoff))

# 2. 分割方法
def threshold_segmentation(gray):
        t = threshold_otsu(gray)
        otsu = (gray > t).astype(np.uint8) * 255
        adapt = (gray > uniform_filter(gray, size=25)).astype(np.uint8) * 255
        return otsu, adapt

def edge_contour_segmentation(img_gray):
        edges = canny(img_gray/255.0)
        mask = binary_fill_holes(edges)
        return (edges*255).astype(np.uint8), (mask*255).astype(np.uint8)

def kmeans_segmentation(img, K=3):
        X = img.reshape(-1, img.shape[-1])
        kmeans = KMeans(n_clusters=K, n_init=10).fit(X)
        labels = kmeans.labels_.reshape(img.shape[:2])
        seg = kmeans.cluster_centers_[labels].astype(np.uint8)
        return seg, labels

def watershed_segmentation(img):
        gray = rgb2gray(img)
        t = threshold_otsu(gray)
        markers = np.zeros_like(gray)
        markers[gray < t] = 1
        markers[gray > t] = 2
        ws = watershed(-gaussian_filter(gray, 2), markers)
        mask = (ws == 2).astype(np.uint8) * 255
        boundary = (ws == 0).astype(np.uint8) * 255
        return mask, boundary

# 3. 形态学处理
def morphological_processing(mask):
        kernel3 = disk(1)
        kernel5 = disk(2)
        opening_img = opening(mask, kernel3)
        closing_img = closing(mask, kernel5)
        dilate_img = dilation(mask, kernel3)
        erode_img = erosion(mask, kernel3)
        return opening_img, closing_img, dilate_img, erode_img

# plotting helpers
def save_compare_grid(outpath, titles, images, cmap=None, figsize=(12,8)):
        n = len(images)
        cols = min(3, n)
        rows = int(np.ceil(n/cols))
        plt.figure(figsize=figsize)
        for i, img in enumerate(images):
                plt.subplot(rows, cols, i+1)
                if img.ndim == 2:
                        plt.imshow(img, cmap='gray')
                else:
                        plt.imshow(img)
                plt.title(titles[i], fontsize=9)
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

def process(input_path, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        img = imread(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]

        # make grayscale
        if img.ndim == 3:
                gray = rgb2gray(img)
                gray = img_as_ubyte(gray)
        else:
                gray = img

        # 1. 平滑比较
        mean = mean_blur(img, ksize=5)
        med = median_blur(img, ksize=5)
        grad_inv = gradient_inverse_weighted(img, ksize=7, beta=20.0)
        fft_lp = fft_lowpass(img, cutoff=20)

        save_compare_grid(os.path.join(out_dir, f"{base}_smoothing_compare.png"),
                                         ["原图", "均值", "中值", "梯度倒数加权", "FFT低通"],
                                         [img, mean, med, grad_inv, fft_lp])

        # 保存单个结果
        for name, im in [("mean", mean), ("median", med), ("gradinv", grad_inv), ("fft", fft_lp)]:
                imwrite(os.path.join(out_dir, f"{base}_smooth_{name}.png"), im)

        # 2. 分割比较
        otsu, adapt = threshold_segmentation(gray)
        edges, contour_mask = edge_contour_segmentation(gray)
        kseg_img, klabel = kmeans_segmentation(img, K=3)
        ws_mask, ws_boundary = watershed_segmentation(img)

        save_compare_grid(os.path.join(out_dir, f"{base}_segmentation_compare.png"),
                                         ["原灰度", "Otsu", "Adaptive", "Edges", "Contours Filled", "KMeans", "Watershed Mask"],
                                         [gray, otsu, adapt, edges, contour_mask, kseg_img, ws_mask])

        # 保存分割结果
        imwrite(os.path.join(out_dir, f"{base}_otsu.png"), otsu)
        imwrite(os.path.join(out_dir, f"{base}_adaptive.png"), adapt)
        imwrite(os.path.join(out_dir, f"{base}_edges.png"), edges)
        imwrite(os.path.join(out_dir, f"{base}_contour_mask.png"), contour_mask)
        imwrite(os.path.join(out_dir, f"{base}_kmeans.png"), kseg_img)
        imwrite(os.path.join(out_dir, f"{base}_watershed_mask.png"), ws_mask)
        imwrite(os.path.join(out_dir, f"{base}_watershed_boundary.png"), ws_boundary)

        # 3. 对几个分割结果做形态学处理并保存对比：选用 Otsu 和 KMeans label map 的一个二值版本（选取最大簇）
        unique, counts = np.unique(klabel, return_counts=True)
        largest_label = unique[np.argmax(counts)]
        k_mask = (klabel == largest_label).astype(np.uint8)*255

        morph_otsu = morphological_processing(otsu)
        morph_k = morphological_processing(k_mask)
        # 保存
        names = ["opening", "closing", "dilate", "erode"]
        for i, fn in enumerate(names):
                imwrite(os.path.join(out_dir, f"{base}_otsu_{fn}.png"), morph_otsu[i])
                imwrite(os.path.join(out_dir, f"{base}_k_{fn}.png"), morph_k[i])

        save_compare_grid(os.path.join(out_dir, f"{base}_morphology_compare.png"),
                                         ["原Otsu", *names, "原KMeansMask", *names],
                                         [otsu, *morph_otsu, k_mask, *morph_k],
                                         figsize=(12,8))

        print("处理完成，输出已保存到:", out_dir)

def main():
                parser = argparse.ArgumentParser(description="图像平滑、分割与形态学处理示例")
                parser.add_argument("--input", "-i", required=True, help="输入PNG图像路径")
                parser.add_argument("--out", "-o", default="question4 answer", help="输出目录")
                args = parser.parse_args()
                process(args.input, args.out)

if __name__ == "__main__":
        main()