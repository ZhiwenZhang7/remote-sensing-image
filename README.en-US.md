

# China University of Mining and Technology - Remote Sensing Image Processing Assignment Summary

This repository contains four independent scripts demonstrating radiometric/atmospheric correction (Question1), image enhancement and PCA fusion (Question2), vegetation extraction (Question3), and image smoothing/segmentation/morphological processing (Question4) on remote sensing imagery. Below are the functions, run examples, key parameters, output descriptions for each script, as well as environment dependencies and running recommendations.

---

## 1. Overall Dependencies (conda-forge recommended)
- Python 3.8–3.11 (Python 3.10 recommended)
- numpy, scipy, scikit-image, scikit-learn, matplotlib, pillow/imageio
- GDAL (Only required for Question1/Question2/Question3 when processing TIF/HDF files)

Recommended via conda:
```powershell
conda create -n py6s-env python=3.10 -y
conda activate py6s-env
conda install -c conda-forge numpy scipy scikit-image scikit-learn matplotlib pillow imageio -y
conda install -c conda-forge gdal -y  # Install if remote sensing TIF/HDF processing is needed
```

---

## 2. Script Overview and Run Examples

All examples are provided for Windows PowerShell. Replace the paths in the commands with your actual file paths.

1) `question1.py` — Radiometric Calibration and 6S Atmospheric Correction (for Landsat)
- Function: Performs radiometric correction on bands in the specified folder and calls Py6S for atmospheric correction. Supports extracting AOD from MOD04 HDF to override default values.
- Key parameters: `--Input_dir` (input root directory), `--Output_dir` (output directory), `--MOD04` (optional MOD04 HDF path).
- Run example:
```powershell
python .\question1.py --Input_dir "D:\dataset\LC08_folder" --Output_dir "D:\output\question1" --MOD04 "D:\MOD04\MOD04_example.hdf"
```

2) `question2.py` — Image Enhancement and PCA Pansharpening
- Function: Crops the central ~5km×5km area, applies linear stretching and histogram equalization, performs PCA calculation, and replaces the first principal component with the panchromatic band to achieve image fusion.
- Output: Multiple PNG images (original panchromatic, linear stretch, histogram equalization, multispectral RGB, PC1, PCA-fused, PCA explained variance).
- Run example (the script will automatically search for bands in the folder):
```powershell
python .\question2.py
```

3) `question3.py` — Vegetation Extraction (NDVI Mask)
- Function: Reads B2/B3/B4/B5 bands, crops the central 5km×5km area, calculates NDVI (and functions for EVI/SAVI), generates a vegetation mask using the threshold NDVI>0.3, and saves two PNG images.
- Run example:
```powershell
python .\question3.py
```

4) `question4.py` — Image Smoothing / Segmentation / Morphological Processing
- Function: Applies various smoothing comparisons (mean, median, gradient inverse weighted, FFT/Gaussian low-pass), segmentation methods (Otsu, Adaptive, Edge padding, KMeans, Watershed), and morphological operations on a single image, and saves comparison images.
- Key parameters: `--input`/`-i` (input image), `--out`/`-o` (output directory, default is `question4 answer`).
- Run example:
```powershell
python .\question4.py -i .\q4\R.png -o .\question4_output
```

---

## 3. Output File Descriptions (based on `question4.py`)
- `{base}_smoothing_compare.png`: Grid comparison chart of smoothing methods.
- `{base}_smooth_mean.png` / `_median.png` / `_gradinv.png` / `_fft.png`: Individual images for each smoothing method.
- `{base}_segmentation_compare.png`: Segmentation method comparison (Otsu, Adaptive, Edges, KMeans, Watershed, etc.).
- `{base}_otsu.png`, `{base}_kmeans.png`, `{base}_watershed_mask.png`, etc.: Individual segmentation images.
- `{base}_otsu_opening.png` / `_closing.png` / `_dilate.png` / `_erode.png`: Morphological operation results.

`question2.py` and `question3.py` will also generate PNGs (displaying true color/masks/fusion/PC1, etc.), while `question1.py` outputs atmospheric correction-calibrated GeoTIFFs (written to the specified output directory).

---

## 4. Quick Installation Verification

Run in PowerShell:
```powershell
python --version
python -c "import numpy, scipy, skimage, sklearn, matplotlib; print(numpy.__version__, scipy.__version__, skimage.__version__, sklearn.__version__, matplotlib.__version__)"
python -c "from osgeo import gdal; print('GDAL', gdal.__version__)"  # Only if GDAL is installed
```

---

## 5. Notes and Recommendations
- If you only run `question4.py`, GDAL installation is not required; if you need to process Landsat TIF/HDF files, please install GDAL via conda (conda-forge).
- If you run `question1.py`, you must install and configure Py6S (and ensure the 6S model executable is available in your system PATH).
