import itertools
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pygmt

global Z_MAX
global Z_MIN
Z_MAX = 15
Z_MIN = 6

def main(args):
    input, output, tiles_path = get_paths(args)
    wild_path = f"{input}/*.xml"
    os.makedirs(output, exist_ok=True)
    xml_files = glob.glob(wild_path)
    path_lists = [
        {"input": xml_file, "output": f"{output}/{i}.csv"} 
        for i, xml_file in enumerate(xml_files)]

    csv_path_lists = []
    csv_path_lists_append = csv_path_lists.append
    for path_dict in path_lists:
        if path_dict["input"].endswith(".aux.xml"):
            continue
        csv_path_lists_append(xml2csv(path_dict))

    csv2tiles(csv_path_lists, tiles_path)

    return

def csv2tiles(csv_path_lists, tiles_path):
    tiles_path_dicts = {}
    tiles_path_dicts[Z_MAX] = make_basetiles(csv_path_lists, tiles_path)

def make_basetiles(csv_path_lists, tiles_path):
    grd, tile_corner = csv2grd(csv_path_lists)
    arr = grd2arr(grd)

def grd2arr(grd):
    # 標高タイル用に値を形成
    arr = arrange_grd(grd)
    arr_r, arr_g, arr_b = [arr]*3

    # https://maps.gsi.go.jp/development/demtile.html
    # x = 2^16R + 2^8G + B
    arr_r = np.right_shift(arr_r, 16)
    arr_r = np.bitwise_and(arr_r, 0xff)
    arr_g = np.right_shift(arr_g, 8)
    arr_g = np.bitwise_and(arr_g, 0xff)
    arr_b = np.bitwise_and(arr_b, 0xff)

    # 無効値が[0,0,256]になっているので、[0,0,128]に変更
    arr_bgr = np.dstack([arr_b, arr_g, arr_r])
    mask = np.all(arr_bgr[:,:,:]==[0,0,256], axis=-1)
    mask = np.where(mask==True, 128, 0)
    arr_r = arr_r - mask
    arr_bgr = np.dstack([arr_b, arr_g, arr_r])
    
    # 上下逆転しているので反転
    arr_bgr_flip = np.flipud(arr_bgr)

    return arr_bgr_flip

def arrange_grd(grd):
    # 水深値を100倍して整数化
    grd = grd * 100
    grd = grd // 1

    # 無効値を2^23で埋める
    arr = grd.values
    arr = np.where(arr>math.pow(2,23), math.pow(2,23), arr)
    fillna = math.pow(2, 23)
    arr = np.nan_to_num(arr, nan=fillna)
    
    return arr.astype(int)


# 点群をグリッド化
def csv2grd(csv_path_lists):
    df_lists = [pd.read_csv(csv_path) for csv_path in csv_path_lists]
    df = pd.concat(df_lists)
    # xmin, xmax, ymin, ymax
    deg_corner = get_minmax(df)

    # 緯度経度をピクセル座標に変換
    pix_corner = deg2pix(deg_corner)
    # ピクセル座標をタイル座標に変換
    tile_corner = pix2tile(pix_corner)
    # タイル座標をピクセル座標に変換
    pix_corner = tile2pix(tile_corner)
    # ピクセル座標を緯度経度に変換
    deg_corner = pix2deg(pix_corner)

    # gmtのパラメータ設定
    spacing = get_gmt_resolution(deg_corner,pix_corner)
    region = get_gmt_range(deg_corner, spacing)

    grd = pygmt.xyz2grd(data=df, region=region, spacing=spacing, duplicate="u", header="1")

    return grd, tile_corner

def get_gmt_range(deg_corner, resolutions):
    x_min = deg_corner[0] + resolutions[0]/2
    x_max = deg_corner[1] - resolutions[0]/2
    y_min = deg_corner[2] + resolutions[1]/2
    y_max = deg_corner[3] - resolutions[1]/2

    return [x_min,x_max,y_min,y_max]

def get_gmt_resolution(deg_corner, pix_corner):
    x_resolution = (deg_corner[1] - deg_corner[0]) / (pix_corner[1] - pix_corner[0])
    y_resolution = - (deg_corner[3] - deg_corner[2]) / (pix_corner[3] - deg_corner[2])

    return [x_resolution, y_resolution]

def pix2deg(pix_corner):
    resolution = 180 / math.pow(2, Z_MAX+7)
    pix_left, pix_right, pix_bottom, pix_top = pix_corner
    deg_left = (pix_left * resolution) - 180
    deg_right = (pix_right * resolution) - 180
    deg_bottom = math.atan(math.e ** ((1 - pix_bottom / 180)*math.pi)) + 360/math.pi - 90
    deg_top = math.atan(math.e ** ((1 - pix_top / 180)*math.pi)) + 360/math.pi - 90

    return [deg_left, deg_right, deg_bottom, deg_top]


def tile2pix(tile_corner):
    pix_corner = [tile * 256 for tile in tile_corner]

    return pix_corner

def pix2tile(pix_corner):
    tile_corner = [pix//256 for pix in pix_corner]
    tile_corner[1] = tile_corner[1] + 1
    tile_corner[2] = tile_corner[2] + 1

    return tile_corner

def deg2pix(deg_corner):
    lon_left, lon_right, lat_bottom, lat_top = deg_corner
    resolution = 180 / math.pow(2, Z_MAX+7)
    pix_left = math.floor((lon_left + 180) / resolution)
    pix_right = math.ceil((lon_right + 180) / resolution)
    pix_top = math.ceil(180 / resolution * (1 - 1/math.pi*math.log(math.tan(math.pi/360 * (lat_top+90)))))
    pix_bottom = math.ceil(180 / resolution * (1 - 1/math.pi*math.log(math.tan(math.pi/360 * (lat_bottom+90)))))

    return [pix_left, pix_right, pix_bottom, pix_top]

def get_minmax(df):
    xmin = df["x"].min()
    xmax = df["x"].max()
    ymin = df["y"].min()
    ymax = df["y"].max()

    return xmin, xmax, ymin, ymax

def xml2csv(pathdict):
    tree = ET.parse(pathdict["input"])
    root = tree.getroot()

    # 名前空間の定義
    namespace = {'gml': 'http://www.opengis.net/gml/3.2'}

    range = {}
    # <gml:lowerCorner>を取得
    lower_corner = root.find('.//gml:lowerCorner', namespaces=namespace)
    lower_corner_value = lower_corner.text if lower_corner is not None else "gml:lowerCorner not found."
    print(lower_corner_value)
    range["x_min"], range["y_min"] = [float(deg) for deg in lower_corner_value.split(" ")]

    # <gml:upperCorner>を取得
    upper_corner = root.find('.//gml:upperCorner', namespaces=namespace)
    upper_corner_value = upper_corner.text if upper_corner is not None else "gml:upperCorner not found."
    range["x_max"], range["y_max"] = [float(deg) for deg in upper_corner_value.split(" ")]

    range_num = {}
    # <gml:low>を取得
    low_corner = root.find('.//gml:low', namespaces=namespace)
    low_corner_value = low_corner.text if low_corner is not None else "gml:lowCorner not found."
    range_num["x_num_min"], range_num["y_num_min"] = [int(num) for num in low_corner_value.split(" ")]

    # <gml:high>を取得
    high_corner = root.find('.//gml:high', namespaces=namespace)
    high_corner_value = high_corner.text if high_corner is not None else "gml:highCorner not found."
    range_num["x_num_max"], range_num["y_num_max"] = [int(num) for num in high_corner_value.split(" ")]

    # <gml:tupleList>を取得
    tuple_list = root.find('.//gml:tupleList', namespaces=namespace)
    dem_lists = tuple_list.text.split()
    dem_lists = [float(line.split(",")[1]) for line in dem_lists]
    
    make_grd(range, range_num, dem_lists, pathdict["output"])

    return pathdict["output"]

def get_paths(args):
    if len(args)<3:
        raise("入力先と出力先を設定してください")
    input = args[1]
    print(f"input:{input}")
    output = args[2]
    print(f"output:{output}")
    tiles_path = args[3]
    print(f"tiles_path:{tiles_path}")

    return input, output, tiles_path

def make_grd(range, range_num, dem_lists, output):
    x_resolution = get_resolution(range["x_min"], range["x_max"], range_num["x_num_min"], range_num["x_num_max"])
    y_resolution = get_resolution(range["y_min"], range["y_max"], range_num["y_num_min"], range_num["y_num_max"])
    region = [range["x_min"]+x_resolution/2,
              range["x_max"]-x_resolution/2,
              range["y_min"]+y_resolution/2,
              range["y_max"]-y_resolution/2]
    x_lists, y_lists = np.mgrid[
        range["x_min"]+x_resolution/2:range["x_max"]+x_resolution/2:x_resolution,
        range["y_min"]+y_resolution/2:range["y_max"]+y_resolution/2:y_resolution]
    x = [float(x_list[0]) for x_list in x_lists]
    y = y_lists[0]
    xy = itertools.product(x, y)
  
    xyz_lists = []
    xyz_lists_append = xyz_lists.append
    for (x, y), z in zip(xy, dem_lists):
        xy_dict = {"x":x, "y":y, "z":z}
        xyz_lists_append(xy_dict)

    df = pd.DataFrame(xyz_lists)
    df["x"], df["y"] = df["y"], df["x"]
    df.to_csv(output, index=None)
    # grd = pygmt.xyz2grd(data=df, region=region, spacing=(x_resolution,y_resolution), duplicate="u", header="1", outgrid=path)

    return

def get_resolution(deg_min, deg_max, num_min, num_max):
    deg_range = deg_max - deg_min
    num_range = num_max - num_min + 1
    
    return deg_range/num_range

if __name__ == "__main__":
    main(sys.argv)
    # input: "/mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A"
    # output: "/mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A/csv"
    # tilepath: "/home/shun/work/learnings/python/xml2tiles/src/tiles"
    # cmd
    # python3 app.py /mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A /mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A/csv /home/shun/work/learnings/python/xml2tiles/src/tiles
    