import itertools
import glob
import os

import cv2
import geopandas as gpd
from shapely import LineString, Polygon, MultiPolygon
from shapely.geometry import shape
from shapely.ops import unary_union


class MakePolygon():
    def __init__(self):
        pass

    def make_polygon(self, path_dicts):
        tile_file = path_dicts["tile_file"]
        polygon_folder = path_dicts["polygon_folder"]
        xyz = self.get_xyz(tile_file)
        
        polygon_zx_folder = "{folder}/{z}/{x}".format(folder=polygon_folder, z=xyz["z"], x=xyz["x"])
        os.makedirs(polygon_zx_folder, exist_ok=True)
        polygon_file = "{folder}/{y}.png".format(folder=polygon_zx_folder, y=xyz["y"])

        if not os.path.isfile(tile_file):
            raise (f"{tile_file}タイルが存在しません")
        img = cv2.imread(tile_file)

        H, W, _ = img.shape
        hw_lists = itertools.product(range(H), range(W))
        pix_polygons = [self.pix2polygon(img, xyz, h, w) for h, w in hw_lists]
        shapes = [shape(poly["geometry"]) for poly in pix_polygons if "geometry" in poly]

        gdf_shapes = gpd.GeoSeries(shapes)
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(unary_union(gdf_shapes.geometry)))
