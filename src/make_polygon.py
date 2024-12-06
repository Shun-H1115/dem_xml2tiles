import itertools
import json
import math
import os

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, shape
from shapely.ops import unary_union


class MakePolygon():
    def __init__(self):
        pass

    # ファイル名からxyzを取得
    def get_xyz(self, tile_file):
        xyz = {}
        tilepath_lists = tile_file.split("/")

        xyz["x"] = int(tilepath_lists[-2])
        xyz["y"] = int(tilepath_lists[-1].replace(".png", ""))
        xyz["z"] = int(tilepath_lists[-3])

        return xyz


    # タイル座標をピクセル座標に変換
    def tile2pix(self, xyz):
        x = xyz["x"]
        y = xyz["y"]

        return x*256, y*256
    

    # タイル座標を緯度経度に変換
    def tile2deg(self, xyz, h, w):
        pix_left, pix_top = self.tile2pix(xyz)
        pix_right = pix_left + w
        pix_bottom = pix_top + h

        return self.pix2deg(pix_left, pix_right, pix_bottom, pix_top, int(xyz["z"]))
    

    # ピクセル座標を緯度経度に変換
    def pix2deg(self, pix_left, pix_right, pix_bottom, pix_top, z):
        resolution = 180 / math.pow(2, z+7)
        deg_left = (pix_left * resolution) - 180
        deg_right = (pix_right * resolution) - 180
        deg_bottom = math.atan(math.e ** ((1 - pix_bottom / 180)*math.pi)) + 360/math.pi - 90
        deg_top = math.atan(math.e ** ((1 - pix_top / 180)*math.pi)) + 360/math.pi - 90

        return [deg_left, deg_right, deg_bottom, deg_top]
    

    def pix2polygon(self, img, xyz, h, w):
        # 誤差分大きめにポリゴンを作成
        error = 0.000_000_01

        # すべて無効値のタイルはポリゴンを作成しない
        if np.all(img[h,w,:]==[0,0,128]):
            return
        
        # タイルを緯度経度変換
        deg_left, deg_right, deg_bottom, deg_top = self.tile2deg(xyz, h, w)
        polygon = Polygon([
            (deg_left - error, deg_top + error),
            (deg_left - error, deg_bottom - error),
            (deg_right + error, deg_bottom - error),
            (deg_right + error, deg_top + error)])
        
        return {"geometry": polygon}
    

    def simplify_geojson(polygon_file, tolerance=0.0005):
        with open(polygon_file, "r") as f:
            data = json.load(f)

        if "features" in data:
            simplified_features = []
            for feature in data["features"]:
                geometry = feature["geometry"]
                if geometry["type"] == "Polygon":
                    simplified_exterior = LineString(geometry["coordinates"][0]).simplify(
                        tolerance, preserve_topology=True
                    )
                    simplified_interiors = [LineString(interior).simplify(
                        tolerance, preserve_topology=True) for interior in geometry["coordinates"][1:]
                    ]
                    feature["geometry"]["coordinates"] = [simplified_exterior.coords[:]] + [i.coords[:] for i in simplified_interiors]
                elif geometry["type"] == "MultiPolygon":
                    simplified_polygons = []
                    for polygon in geometry["coordinates"]:
                        simplified_exterior = LineString(polygon[0]).simplify(
                            tolerance, preserve_topology=True
                        )
                        simplified_interiors = [LineString(interior).simplify(
                            tolerance, preserve_topology=True) for interior in polygon[1:]
                        ]
                        simplified_polygons.append(
                            [simplified_exterior.coords[:]] + [i.coords[:] for i in simplified_interiors]
                        )
                    feature["geometry"]["coordinates"] = simplified_polygons
                else:
                    raise ("タイル画像が不適切です")
            data["features"] = simplified_features

        with open(polygon_file, "w") as f:
            json.dump(data, f)


    def make_polygon(self, path_dicts):
        tile_file = path_dicts["tile_file"]
        polygon_folder = path_dicts["polygon_folder"]
        xyz = self.get_xyz(tile_file)
        
        # ポリゴンの保存先を作成
        polygon_zx_folder = "{folder}/{z}/{x}".format(folder=polygon_folder, z=xyz["z"], x=xyz["x"])
        os.makedirs(polygon_zx_folder, exist_ok=True)
        polygon_file = "{folder}/{y}.png".format(folder=polygon_zx_folder, y=xyz["y"])

        if not os.path.isfile(tile_file):
            raise (f"{tile_file}タイルが存在しません")
        img = cv2.imread(tile_file)

        # 画像のすべてのピクセルを調べ、無効値でなければピクセル範囲のポリゴン
        H, W, _ = img.shape
        hw_lists = itertools.product(range(H), range(W))
        pix_polygons = [self.pix2polygon(img, xyz, h, w) for h, w in hw_lists]
        shapes = [shape(poly["geometry"]) for poly in pix_polygons if "geometry" in poly]

        # ピクセル単位のポリゴンをマージ
        gdf_shapes = gpd.GeoSeries(shapes)
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(unary_union(gdf_shapes.geometry)))
        gdf.to_file(polygon_file, driver="GeoJSON")
        
        # 直線状の頂点を間引く
        self.simplify_geojson(polygon_file)
