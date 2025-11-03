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


import glob
import os
import sys
import tarfile
import re
import numpy
from osgeo import gdal
from Py6S import SixS, Geometry, AtmosProfile, AeroProfile, GroundReflectance, Altitudes, Wavelength, PredefinedWavelengths, AtmosCorr
import pdb
import shutil
import argparse
import tempfile
# 如 base.py 在同目录，改为如下，否则请调整为实际路径
try:
    from base import MeanDEM
except ImportError:
    def MeanDEM(pointUL, pointDR):
        # 占位实现，返回0
        return 0

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--Input_dir', type=str, help='Input dir', default=None)
    parser.add_argument('--Output_dir', type=str, help='Output dir', default=None)
    parser.add_argument('--MOD04', type=str, help='Optional MOD04 HDF file path to use for AOD extraction', default=None)
    return parser.parse_args(argv)

# 逐波段辐射定标
def RadiometricCalibration(BandId, data2, ImgRasterData):
    parameter_OLI = numpy.zeros((9,2))
    # 计算辐射亮度参数，正则健壮性处理
    for i in range(9):
        mult = re.findall(f'RADIANCE_MULT_BAND_{i+1}.+', data2)
        add = re.findall(f'RADIANCE_ADD_BAND_{i+1}.+', data2)
        if mult and add:
            parameter_OLI[i,0] = float(mult[0].split('=')[1])
            parameter_OLI[i,1] = float(add[0].split('=')[1])
        else:
            parameter_OLI[i,0] = 0
            parameter_OLI[i,1] = 0
    Gain = parameter_OLI[int(BandId) - 1,0]
    Bias = parameter_OLI[int(BandId) - 1,1]
    RaCal = numpy.where(ImgRasterData>0 ,Gain * ImgRasterData + Bias,-9999)
    return RaCal

# 6s大气校正
def get_aot_from_mod04(mod04_hdf_path, target_tif_path, log=None):
    """
    从 MOD04 HDF 中提取 AOD（550nm）并重采样到 target_tif_path 的栅格，返回区域代表值（中位数）或 None。
    """
    ds = gdal.Open(mod04_hdf_path)
    if ds is None:
        if log:
            log.write(f"\n无法打开 MOD04 文件: {mod04_hdf_path}")
        return None

    sds = ds.GetSubDatasets()
    if not sds:
        if log:
            log.write(f"\nMOD04 文件没有子数据集: {mod04_hdf_path}")
        return None

    # 选择可能的 AOD 子数据集
    candidate = None
    for name, desc in sds:
        key = desc.lower()
        if 'optical' in key and ('550' in key or 'aod' in key or 'aerosol' in key):
            candidate = name
            break
        if 'aerosol_optical' in key:
            candidate = name
            break
    if candidate is None:
        # 退而求其次，使用第一个子数据集
        candidate = sds[0][0]

    target_ds = gdal.Open(target_tif_path)
    if target_ds is None:
        if log:
            log.write(f"\n无法打开目标文件以匹配投影: {target_tif_path}")
        return None

    gt = target_ds.GetGeoTransform()
    cols = target_ds.RasterXSize
    rows = target_ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * cols
    ymin = ymax + gt[5] * rows

    tmpfile = os.path.join(tempfile.gettempdir(), f"mod04_aod_resampled_{os.path.basename(target_tif_path)}.tif")
    try:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
    except Exception:
        pass

    # 重采样并投影到目标栅格
    try:
        gdal.Warp(tmpfile, candidate, format='GTiff', width=cols, height=rows,
                  dstSRS=target_ds.GetProjection(), outputBounds=(xmin, ymin, xmax, ymax),
                  resampleAlg='bilinear')
    except Exception as e:
        if log:
            log.write(f"\nMOD04 重采样失败: {e}")
        return None

    resampled = gdal.Open(tmpfile)
    if resampled is None:
        return None
    band = resampled.GetRasterBand(1)
    arr = band.ReadAsArray().astype(numpy.float32)

    nodata = band.GetNoDataValue()
    valid = numpy.isfinite(arr)
    if nodata is not None:
        valid &= (arr != nodata)
    # 过滤不合理值（AOD 通常 0-5）
    valid &= (arr > 0)
    valid &= (arr < 5.0)

    if numpy.sum(valid) == 0:
        try:
            resampled = None
            os.remove(tmpfile)
        except Exception:
            pass
        return None

    aod_val = float(numpy.median(arr[valid]))

    try:
        resampled = None
        os.remove(tmpfile)
    except Exception:
        pass

    return aod_val


def AtmosphericCorrection(BandId, data2, override_aot=None):
    # 6S模型
    s = SixS()

    s.geometry = Geometry.User()
    s.geometry.solar_z = 90-float(''.join(re.findall('SUN_ELEVATION.+',data2)).split("=")[1])
    s.geometry.solar_a = float(''.join(re.findall('SUN_AZIMUTH.+',data2)).split("=")[1])
    s.geometry.view_z = 0
    s.geometry.view_a = 0


    # 日期
    Dateparm = ''.join(re.findall('DATE_ACQUIRED.+',data2)).split("=")
    Date = Dateparm[1].split('-')

    s.geometry.month = int(Date[1])
    s.geometry.day = int(Date[2])

    # 中心经纬度
    point1lat = float(''.join(re.findall('CORNER_UL_LAT_PRODUCT.+',data2)).split("=")[1])
    point1lon = float(''.join(re.findall('CORNER_UL_LON_PRODUCT.+',data2)).split("=")[1])
    point2lat = float(''.join(re.findall('CORNER_UR_LAT_PRODUCT.+',data2)).split("=")[1])
    point2lon = float(''.join(re.findall('CORNER_UR_LON_PRODUCT.+',data2)).split("=")[1])
    point3lat = float(''.join(re.findall('CORNER_LL_LAT_PRODUCT.+',data2)).split("=")[1])
    point3lon = float(''.join(re.findall('CORNER_LL_LON_PRODUCT.+',data2)).split("=")[1])
    point4lat = float(''.join(re.findall('CORNER_LR_LAT_PRODUCT.+',data2)).split("=")[1])
    point4lon = float(''.join(re.findall('CORNER_LR_LON_PRODUCT.+',data2)).split("=")[1])

    sLongitude = (point1lon + point2lon + point3lon + point4lon) / 4
    sLatitude = (point1lat + point2lat + point3lat + point4lat) / 4

    # 大气模式类型
    if sLatitude > -15 and sLatitude <= 15:
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.Tropical)

    if sLatitude > 15 and sLatitude <= 45:
        if s.geometry.month > 4 and s.geometry.month <= 9:
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
        else:
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeWinter)

    if sLatitude > 45 and sLatitude <= 60:
        if s.geometry.month > 4 and s.geometry.month <= 9:
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticSummer)
        else:
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.SubarcticWinter)

    # 气溶胶类型大陆
    s.aero_profile = AtmosProfile.PredefinedType(AeroProfile.Continental)

    # 目标地物？？？？？？
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.36)

    # 550nm气溶胶光学厚度,可由外部传入覆盖（override），否则使用默认值
    # 默认值（原脚本中的经验值）
    s.aot550 = 0.14497
    # 如果外部提供了 aot（例如来自 MOD04），则覆盖默认值
    if override_aot is not None:
        try:
            aot_val = float(override_aot)
            # 6S aot550 应为非负且合理
            if aot_val >= 0 and aot_val < 10:
                s.aot550 = aot_val
        except Exception:
            pass

    # 通过研究去区的范围去求DEM高度。
    pointUL = dict()
    pointDR = dict()
    pointUL["lat"] = point1lat
    pointUL["lon"] = point1lon
    pointDR["lat"] = point4lat
    pointDR["lon"] = point2lon
    meanDEM = (MeanDEM(pointUL, pointDR)) * 0.001

    # 研究区海拔、卫星传感器轨道高度
    s.altitudes = Altitudes()
    s.altitudes.set_target_custom_altitude(meanDEM)
    s.altitudes.set_sensor_satellite_level()

    # 校正波段（根据波段名称）
    if BandId == '1':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B1)

    elif BandId == '2':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B2)

    elif BandId == '3':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B3)

    elif BandId == '4':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B4)

    elif BandId == '5':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B5)

    elif BandId == '6':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B6)

    elif BandId == '7':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B7)

    elif BandId == '8':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B8)

    elif BandId == '9':
        s.wavelength = Wavelength(PredefinedWavelengths.LANDSAT_OLI_B9)

    # 下垫面非均一、朗伯体
    s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance(-0.1)

    # 运行6s大气模型
    s.run()

    xa = s.outputs.coef_xa
    xb = s.outputs.coef_xb
    xc = s.outputs.coef_xc
    x = s.outputs.values
    return (xa, xb, xc)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    RootInputPath = args.Input_dir
    RootOutName = args.Output_dir

    LogFile = open(os.path.join(RootOutName, 'log.txt'), 'w')

    for root, dirs, RSFiles in os.walk(RootInputPath):
        if len(dirs) == 0:
            RootInputPathList = RootInputPath.split(os.path.sep)
            RootList = root.split(os.path.sep)
            StartList = len(RootInputPathList)
            EndList = len(RootList)
            outname = RootOutName
            for i in range(StartList, EndList):
                if not os.path.exists(os.path.join(outname, RootList[i])):
                    os.makedirs(os.path.join(outname, RootList[i]))
                outname = os.path.join(outname, RootList[i])

            MeteDatas = glob.glob(os.path.join(root, '*MTL.txt'))
            for MeteData in MeteDatas:
                try:
                    with open(MeteData) as f:
                        data = f.readlines()
                    data2 = ' '.join(data)
                except Exception as e:
                    print(f"读取元数据失败: {MeteData}, {e}")
                    LogFile.write(f'\n{MeteData}元数据读取失败')
                    continue

                shutil.copyfile(MeteData, os.path.join(outname, os.path.basename(MeteData)))

                # 尝试定位 MOD04 文件（优先使用命令行传入）
                mod04_path = None
                if args.MOD04 and os.path.isfile(args.MOD04):
                    mod04_path = args.MOD04
                else:
                    # 在当前根目录、输入根以及脚本目录下查找 MOD04*.hdf
                    candidates = []
                    try:
                        candidates += glob.glob(os.path.join(root, 'MOD04*.hdf'))
                    except Exception:
                        pass
                    try:
                        candidates += glob.glob(os.path.join(RootInputPath, 'MOD04*.hdf'))
                    except Exception:
                        pass
                    try:
                        candidates += glob.glob(os.path.join(os.path.dirname(__file__), 'MOD04*.hdf'))
                    except Exception:
                        pass
                    if candidates:
                        mod04_path = candidates[0]

                if len(os.path.basename(MeteData)) < 10:
                    RSbands = glob.glob(os.path.join(root, "B0[1-8].tiff"))
                else:
                    RSbands = glob.glob(os.path.join(root, "*B[1-8].TIF"))
                print('影像' + root + '开始大气校正')
                print(RSbands)
                for tifFile in RSbands:
                    BandId = (os.path.basename(tifFile).split('.')[0])[-1]
                    try:
                        IDataSet = gdal.Open(tifFile)
                    except Exception as e:
                        print(f"文件{tifFile}打开失败: {e}")
                        LogFile.write(f'\n{tifFile}数据打开失败')
                        continue
                    if IDataSet is None:
                        LogFile.write(f'\n{tifFile}数据集读取为空')
                        continue
                    cols = IDataSet.RasterXSize
                    rows = IDataSet.RasterYSize
                    ImgBand = IDataSet.GetRasterBand(1)
                    ImgRasterData = ImgBand.ReadAsArray(0, 0, cols, rows)
                    if ImgRasterData is None:
                        LogFile.write(f'\n{tifFile}栅格数据为空')
                        continue
                    outFilename = os.path.join(outname, os.path.basename(tifFile))
                    if os.path.isfile(outFilename):
                        print(f"{outFilename}已经完成")
                        continue
                    # 辐射校正
                    RaCalRaster = RadiometricCalibration(BandId, data2, ImgRasterData)
                    # 大气校正（如果找到 MOD04，会尝试从中提取 AOD 并覆盖默认值）
                    aod_override = None
                    if mod04_path is not None:
                        try:
                            aod_override = get_aot_from_mod04(mod04_path, tifFile, log=LogFile)
                        except Exception as e:
                            LogFile.write(f"\n从 MOD04 提取 AOD 失败: {e}")

                    a, b, c = AtmosphericCorrection(BandId, data2, override_aot=aod_override)
                    y = numpy.where(RaCalRaster != -9999, a * RaCalRaster - b, -9999)
                    atc = numpy.where(y != -9999, (y / (1 + y * c)) * 10000, -9999)
                    driver = IDataSet.GetDriver()
                    outDataset = driver.Create(outFilename, cols, rows, 1, gdal.GDT_Int16)
                    geoTransform = IDataSet.GetGeoTransform()
                    outDataset.SetGeoTransform(geoTransform)
                    proj = IDataSet.GetProjection()
                    outDataset.SetProjection(proj)
                    outband = outDataset.GetRasterBand(1)
                    outband.SetNoDataValue(-9999)
                    outband.WriteArray(atc, 0, 0)
                    print(f'第{BandId}波段计算完成')
                print('\n')
    LogFile.close()