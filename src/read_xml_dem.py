import itertools
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd



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
    tiles_folder = args[3]
    print(f"tiles_folder:{tiles_folder}")

    return input, output, tiles_folder

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

    return

def get_resolution(deg_min, deg_max, num_min, num_max):
    deg_range = deg_max - deg_min
    num_range = num_max - num_min + 1
    
    return deg_range/num_range