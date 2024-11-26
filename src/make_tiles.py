import itertools
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import pygmt

class MakeTiles():
    def __init__(self, z_base:int, z_min:int, z_max:int):
        if z_base < z_min:
            raise("引数はベースレベル、最小ズームレベル、最大ズームレベルの順に設定してください")
        if z_base > z_max:
            raise("引数はベースレベル、最小ズームレベル、最大ズームレベルの順に設定してください")
        self.z_base = z_base
        self.z_min = z_min
        self.z_max = z_max

    
    def deg2pix_corner(self, df):
        lon_min = df["x"].min()
        lon_max = df["x"].max()
        lat_min = df["y"].min()
        lat_max = df["y"].max()

        return self.deg2pix(lon_min, lon_max, lat_min, lat_max, self.z_base)

    
    def deg2pix(self, lon_min, lon_max, lat_min, lat_max, z:int):
        resolution = 180 / math.pow(2, z+7)
        pix_left = math.floor((lon_min + 180) / resolution)
        pix_right = math.ceil((lon_max + 180) / resolution)
        pix_top = math.ceil(180 / resolution * (1 - 1/math.pi*math.log(math.tan(math.pi/360 * (lat_min+90)))))
        pix_bottom = math.ceil(180 / resolution * (1 - 1/math.pi*math.log(math.tan(math.pi/360 * (lat_max+90)))))

        return [pix_left, pix_right, pix_bottom, pix_top]


    def pix2tile_corner(self, pix_corner):
        tile_corner = [pix//256 for pix in pix_corner]
        tile_corner[1] = tile_corner[1] + 1
        tile_corner[2] = tile_corner[2] + 1

        return tile_corner

        
    def tile2pix_corner(self, tile_corner):
        pix_corner = [tile * 256 for tile in tile_corner]

        return pix_corner

        
    def pix2deg_corner(self, pix_corner, z:int):
        resolution = 180 / math.pow(2, z+7)
        pix_left, pix_right, pix_bottom, pix_top = pix_corner
        deg_left = (pix_left * resolution) - 180
        deg_right = (pix_right * resolution) - 180
        deg_bottom = math.atan(math.e ** ((1 - pix_bottom / 180)*math.pi)) + 360/math.pi - 90
        deg_top = math.atan(math.e ** ((1 - pix_top / 180)*math.pi)) + 360/math.pi - 90

        return [deg_left, deg_right, deg_bottom, deg_top]


    # gmtのパラメータ(spacing)を設定
    def get_gmt_resolution(self, deg_corner, pix_corner):
        x_resolution = (deg_corner[1] - deg_corner[0]) / (pix_corner[1] - pix_corner[0])
        y_resolution = - (deg_corner[3] - deg_corner[2]) / (pix_corner[3] - deg_corner[2])

        return (x_resolution, y_resolution)


    # gmtのパラメータ(region)を設定
    def get_gmt_region(self, deg_corner, resolutions):
        x_min = deg_corner[0] + resolutions[0]/2
        x_max = deg_corner[1] - resolutions[0]/2
        y_min = deg_corner[2] + resolutions[1]/2
        y_max = deg_corner[3] - resolutions[1]/2

        return [x_min,x_max,y_min,y_max]


    # 点群をグリッド化
    def make_grd(self, df, duplicate, is_nearest):
        """
        """
        duplicate_dict = {
            "f": "the first data point",
            "s": "the last data point",
            "l": "the lowest(minimum) value",
            "u": "the upper(maximum) value",
            "d": "the difference between the maximum and minimum",
            "m": "mean value",
            "r": "RMS value",
            "S": "standard deviation",
            "n": "count the number of data points",
            "z": "sum multiple values"
        }
        if not duplicate in duplicate_dict:
            raise ("duplicateを正しく設定してください")
        # 緯度経度をピクセル座標に変換
        pix_corner = self.deg2pix(deg_corner)
        # ピクセル座標をタイル座標に変換
        tile_corner = self.pix2tile_corner(pix_corner)
        # タイル座標をピクセル座標に変換
        pix_corner = self.tile2pix_corner(tile_corner)
        # ピクセル座標を緯度経度に変換
        deg_corner = self.pix2deg_corner(pix_corner)

        # gmtのパラメータ設定
        spacing = self.get_gmt_resolution(deg_corner,pix_corner)
        region = self.get_gmt_region(deg_corner, spacing)

        grd = pygmt.xyz2grd(data=df, region=region, spacing=spacing, duplicate=duplicate, header="1")

        if not is_nearest:
            return grd, tile_corner

        # 補完
        radius = (180 / math.pow(2, self.z_base+7)) * 2
        xyz = pygmt.grd2xyz(grid=grd, skiprows=True, output_type="pandas")
        grd_nearest = pygmt.nearneighbor(data=xyz, spacing=spacing, region=region, search_radius=radius, verbose="w", sectors=1)

        return grd_nearest, tile_corner


class ColorTiles(MakeTiles):
    def __init__(self, z_base, z_min, z_max):
        super().__init__(z_base, z_min, z_max)
        # 無効値タイル
        self.arr_inv = np.zeros((256, 256, 3))
        self.arr_inv[:] = [0, 0, 128]


    def grd2arr(self, grd):
        # 標高タイル用に値を形成
        arr = self.arrange_grd(grd)
        arr_r, arr_g, arr_b = [arr, arr, arr]

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

    def arrange_grd(self, grd):
        # 水深値を100倍して整数化
        grd = grd * 100
        grd = grd // 1

        # 無効値を2^23で埋める
        arr = grd.values
        arr = np.where(arr>math.pow(2,23), math.pow(2,23), arr)
        fillna = math.pow(2, 23)
        arr = np.nan_to_num(arr, nan=fillna)
        
        return arr.astype(int)


    def arr2png(self, arr, tiles_folder, z, x, y):
        try:
            tiles_folder = f"{tiles_folder}/{z}/{x}"
            tiles_file = f"{tiles_folder}/{y}.png"
            os.makedirs(tiles_folder, exist_ok=True)
            cv2.imwrite(tiles_file, arr)

            return True
        
        except Exception as e:
            print(f"タイル画像の出力に失敗しました:{e}")

            return False


    def make_basetile(self, arr, tile_corner, x_num, y_num, x, y, tiles_folder):
        x_split = np.split(arr, x_num, 1)
        xy_split = np.split(x_split[x - tile_corner[0]], y_num, 0)
        tile_arr = xy_split[y - tile_corner[3]]
        if np.allclose(tile_arr, self.arr_inv):
            return {}
        
        # タイル画像出力に成功 → xyz座標を返す
        done_save_png = self.arr2png(tile_arr, tiles_folder, self.z_base, x, y)
        if done_save_png:
            return {"x": x, "y": y, "z": self.z_base}
        
        else:
            raise ("タイル画像の保存に失敗しました")


    def make_basetiles(self, df, tiles_folder, duplicate, is_nearest):
        grd, tile_corner = self.make_grd(df, duplicate, is_nearest)

        arr = self.grd2arr(grd)

        # タイルの枚数をカウント
        x_num = tile_corner[1] - tile_corner[0]
        y_num = - (tile_corner[3] - tile_corner[2])

        xy_lists = itertools.product(range(tile_corner[0], tile_corner[1]),range(tile_corner[3], tile_corner[2]))
        xy_dicts = [self.make_basetile(arr, tile_corner, x_num, y_num, x, y, tiles_folder) for x, y in xy_lists]
        xy_df = pd.DataFrame(xy_dicts).dropna().drop_duplicates()

        return xy_df.to_dict(orient="records")


    def rebin_nan(self, arr):
        # 二次元配列のサイズをH/2,W/2に変換
        shape = [n // 2 for n in arr.shape]
        new_arr = np.empty((shape[0], shape[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                arr_list = np.array([arr[i*2, j*2],arr[i*2+1, j*2],arr[i*2, j*2+1],arr[i*2+1, j*2+1]])
                new_arr[i,j] = np.nanmean(arr_list)

        return new_arr

    def resize_array(self, arr):
        # 無効値[0,0,128]をマスクに設定
        mask = np.all(arr[:,:,:]==[0,0,128], axis=-1)
        # BGR成分に分解
        arr_b = arr[:,:,0]
        arr_g = arr[:,:,1]
        arr_r = arr[:,:,2]
        # 水深値に戻す
        arr_x = arr_r * math.pow(2, 16) + arr_g * math.pow(2, 8) + arr_b
        # 分解能 u = 0.01
        u = 0.01
        arr_h = np.where(arr_x > math.pow(2, 23), (arr_x - math.pow(2, 24)) * u, arr_x * u)
        arr_h = np.where(mask==True, np.nan, arr_h)

        # サイズをH/2,W/2に変換
        arr_h_resize = self.rebin_nan(arr_h)
        # 無効値を2^23で埋める
        arr_h_fillna = np.nan_to_num(arr_h_resize / u, nan=math.pow(2, 23))
        # 画像出力するため、整数値に変換
        arr_h_int = arr_h_fillna.astype(int)
        arr_r, arr_g, arr_b = [arr_h_int, arr_h_int, arr_h_int]

        # ビット演算
        arr_r = np.right_shift(arr_r, 16)
        arr_r = np.bitwise_and(arr_r, 0xff)
        arr_g = np.right_shift(arr_g, 8)
        arr_g = np.bitwise_and(arr_g, 0xff)
        arr_b = np.bitwise_and(arr_b, 0xff)

        return np.dstack([arr_b, arr_g, arr_r])


    def make_downlevel(self, tiles_folder, xyz):
        x = int(xyz["x"])
        y = int(xyz["y"])
        z = int(xyz["z"])

        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2}/{y*2}.png"):
            img_l_t = self.arr_inv
        else:
            img_l_t = cv2.imread(f"{tiles_folder}/{z+1}/{x*2}/{y*2}.png")
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2}.png"):
            img_r_t = self.arr_inv
        else:
            img_r_t = cv2.imread(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2}.png")
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2}/{y*2+1}.png"):
            img_l_b = self.arr_inv
        else:
            img_l_b = cv2.imread(f"{tiles_folder}/{z+1}/{x*2}/{y*2+1}.png")
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2+1}.png"):
            img_r_b = self.arr_inv
        else:
            img_r_b = cv2.imread(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2+1}.png")

        merge_img = np.vstack([
            np.hstack([img_l_t,img_r_t]),
            np.hstack([img_l_b,img_r_b])])
        new_img = self.resize_array(merge_img)
        
        if np.allclose(new_img, self.arr_inv):
            return {}

        can_save_img = self.arr2png(new_img, tiles_folder, z, x, y)
        if not can_save_img:
            raise (f"x{x}, y{y}, z{z}のタイル画像保存に失敗しました")
        
        return xyz


    def make_downleveltiles(self, xyz_dicts, tiles_folder):
        new_xyz_dicts = [{"x":int(xyz_dict["x"]//2), "y":int(xyz_dict["y"]//2), "z":int(xyz_dict["z"]-1)} for xyz_dict in xyz_dicts]
        df = pd.DataFrame(new_xyz_dicts).dropna().drop_duplicates()
        new_xyz_dicts = df.to_dict(orient="records")
        created_xyz_lists = [self.make_downlevel(tiles_folder, xyz) for xyz in new_xyz_dicts]

        z = created_xyz_lists[0]["z"]
        if z == self.z_min:
            return 
        else:
            xy_df = pd.DataFrame(created_xyz_lists).dropna().drop_duplicates()
            created_xyz_lists = xy_df.to_dict(orient="records")

            return self.make_downleveltiles(created_xyz_lists, tiles_folder)


    def make_uplevel(self, tiles_folder, xyz):
        x = int(xyz["x"])
        y = int(xyz["y"])
        z = int(xyz["z"])
        
        tile_file = f"{tiles_folder}/{z}/{x}/{y}.png"
        if not os.path.isfile(tile_file):
            return {}
        
        img_arr = cv2.imread(tile_file)
        H, W, C = img_arr.shape
        img_exp = np.empty((H*2, W*2, C))
        hw_lists = itertools.product(range(H), range(W))
        for i, j in hw_lists:
            img_exp[2*i,2*j,:] = img_arr[i,j,:]
            img_exp[2*i+1,2*j,:] = img_arr[i,j,:]
            img_exp[2*i,2*j+1,:] = img_arr[i,j,:]
            img_exp[2*i+1,2*j+1,:] = img_arr[i,j,:]
        img_lists = [img_exp[H*x:H*(x+1), W*y:W*(y+1)] for x in range(2) for y in range(2)]

        uptiles_dicts = [
            {"x":x*2, "y":y*2, "z":z+1},
            {"x":x*2+1, "y":y*2, "z":z+1},
            {"x":x*2, "y":y*2+1, "z":z+1},
            {"x":x*2+1, "y":y*2+1, "z":z+1}
        ]

        created_xyz_dicts = [
            {"xyz":uptiles_dict, "img":img} for uptiles_dict, img in zip(uptiles_dicts, img_lists) 
            if not np.allclose(img, self.arr_inv)]

        [self.arr2png(d["img"], tiles_folder, d["xyz"]["z"], d["xyz"]["x"], d["xyz"]["y"]) for d in created_xyz_dicts]

        return [d["xyz"] for d in created_xyz_dicts]


    # 拡大レベルのタイル作成
    def make_upleveltiles(self, xyz_dicts, tiles_folder):
        created_xyz_lists = [self.make_uplevel(xyz_dict, tiles_folder) for xyz_dict in xyz_dicts]
        created_xyz_dicts = [d for xyz_dict in created_xyz_lists for d in xyz_dict]

        # 最大ズームレベルを作り終えると終了
        created_z = created_xyz_dicts[0]["z"]
        if created_z==self.z_max:
            return

        # 最大ズームレベルまで再帰処理
        return self.make_upleveltiles(created_xyz_dicts, tiles_folder)

    def make_tiles(self, df, tiles_folder, duplicate="m", is_nearest=None):
        # ベースレベルのタイル作成
        xyz_dicts = self.make_basetiles(df, tiles_folder, duplicate, is_nearest)
        # 縮小レベルのタイル作成
        self.make_downleveltiles(xyz_dicts, tiles_folder)
        # 拡大レベルのタイル作成
        self.make_upleveltiles(xyz_dicts, tiles_folder)

        return


class HillShadeTiles(MakeTiles):
    def __init__(self, z_base, z_min, z_max):
        super().__init__(z_base, z_min, z_max)
        # 無効値タイル
        inv_base = np.repeat(0, 256*256)
        self.arr_a_inv = np.reshape(inv_base, (256,256))
        self.arr_base_bgr = np.zeros((256, 256, 3))
        self.arr_base_bgr[:] = [255, 255, 255]
        self.arr_shade_inv = np.dstack([self.arr_base_bgr, self.arr_a_inv])

        self.exaggeration = 5
        self.light = 292.5
        self.dimensionless = 1 / 60 / 1852
        

    # 陰影図の標準化
    def norm_shade(self, arr):
        f = lambda x: math.atan(math.atan(x))
        function = np.vectorize(f)
        arr_gradient = function(arr)

        max = math.atan(math.pi / 2)
        min = - math.atan(math.pi / 2)
        diff = (max - min) / 255

        f_norm = lambda x: 255 - (x - min) / diff
        function = np.vectorize(f_norm)
        arr_gradient_1d = function(arr_gradient)
        arr_gradient_3d = np.dstack([arr_gradient_1d, arr_gradient_1d, arr_gradient_1d])
        arr_gradient_flip = np.flipud(arr_gradient_3d)

        return np.nan_to_num(arr_gradient_flip, nan=255)


    def grd2arr(self, grd):
        grd = grd * self.exaggeration
        grd_gradient = pygmt.grdgradient(grid=grd, azimuth=self.light)
        # 無次元化
        grd_gradient = grd_gradient * self.dimensionless

        return self.norm_shade(grd_gradient.values)


    def arr2png(self, arr, tiles_folder, z, x, y):
        try:
            tiles_folder = f"{tiles_folder}/{z}/{x}"
            tiles_file = f"{tiles_folder}/{y}.png"
            os.makedirs(tiles_folder, exist_ok=True)
            cv2.imwrite(tiles_file, arr)

            return True
        
        except Exception as e:
            print(f"タイル画像の出力に失敗しました:{e}")

            return False


    def make_basetile(self, arr, tile_corner, x_num, y_num, x, y, tiles_folder):
        x_split = np.split(arr, x_num, 1)
        xy_split = np.split(x_split[x - tile_corner[0]], y_num, 0)
        tile_arr = xy_split[y - tile_corner[3]]
        if np.allclose(tile_arr, self.arr_base_bgr):
            return {}
        
        # タイル画像出力に成功 → xyz座標を返す
        arr_a = np.where(np.all(tile_arr==255, axis=-1), 0, 255)
        arr_gbra = np.dstack([tile_arr, arr_a])
        done_save_png = self.arr2png(arr_gbra, tiles_folder, self.z_base, x, y)
        if done_save_png:
            return {"x": x, "y": y, "z": self.z_base}
        
        else:
            raise ("タイル画像の保存に失敗しました")


    def make_basetiles(self, df, tiles_folder, duplicate, is_nearest):
        grd, tile_corner = self.make_grd(df, duplicate, is_nearest)

        arr = self.grd2arr(grd)

        # タイルの枚数をカウント
        x_num = tile_corner[1] - tile_corner[0]
        y_num = - (tile_corner[3] - tile_corner[2])

        xy_lists = itertools.product(range(tile_corner[0], tile_corner[1]),range(tile_corner[3], tile_corner[2]))
        xy_dicts = [self.make_basetile(arr, tile_corner, x_num, y_num, x, y, tiles_folder) for x, y in xy_lists]
        xy_df = pd.DataFrame(xy_dicts).dropna().drop_duplicates()

        return xy_df.to_dict(orient="records")


    def rebin(self, arr):
        shape = [n // 2 for n in arr.shape]
        new_arr = shape[0], arr.shape[0]//shape[0], shape[1], arr.shape[1]//shape[1]
        new_arr = arr.reshape(new_arr).mean(-1).mean(1)

        return self.arr_check(new_arr)


    # 合成したピクセルに無効値がある場合は除外して計算
    def arr_check(self, arr):
        arr = np.where(arr==256*4, np.nan, arr)
        arr = np.where(arr*4>256*4*3, arr*4-256*4*3, arr)
        arr = np.where(arr*4>256*4*2, arr*4-256*4*2/2, arr)
        arr = np.where(arr*4>256*4, (arr*4-256*4)/3, arr)

        return arr


    def resize_array(self, arr):
        # 無効値を[255,255,255,0]をマスク範囲に設定
        mask_arr = np.all(arr[:,:,:]==[255,255,255,0], axis=-1)
        arr_g = np.where(mask_arr==True, 256*4, arr[:,:,0])
        arr_a = np.where(mask_arr==True, 256*4, arr[:,:,3])
        new_arr_g = self.rebin(arr_g)
        new_arr_a = self.rebin(arr_a)
        new_arr_bgr = np.dstack([new_arr_g, new_arr_g, new_arr_g])
        new_arr_bgra = np.dstack([np.nan_to_num(new_arr_bgr, nan=255), np.nan_to_num(new_arr_a, nan=0)])

        return new_arr_bgra


    def make_downlevel(self, tiles_folder, xyz):
        x = int(xyz["x"])
        y = int(xyz["y"])
        z = int(xyz["z"])

        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2}/{y*2}.png"):
            img_l_t = self.arr_shade_inv
        else:
            img_l_t = cv2.imread(f"{tiles_folder}/{z+1}/{x*2}/{y*2}.png", -1)
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2}.png"):
            img_r_t = self.arr_shade_inv
        else:
            img_r_t = cv2.imread(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2}.png", -1)
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2}/{y*2+1}.png"):
            img_l_b = self.arr_shade_inv
        else:
            img_l_b = cv2.imread(f"{tiles_folder}/{z+1}/{x*2}/{y*2+1}.png", -1)
        if not os.path.isfile(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2+1}.png"):
            img_r_b = self.arr_shade_inv
        else:
            img_r_b = cv2.imread(f"{tiles_folder}/{z+1}/{x*2+1}/{y*2+1}.png", -1)

        merge_img = np.vstack([
            np.hstack([img_l_t,img_r_t]),
            np.hstack([img_l_b,img_r_b])])
        new_img = self.resize_array(merge_img)
        
        if np.allclose(new_img, self.arr_shade_inv):
            return {}

        can_save_img = self.arr2png(new_img, tiles_folder, z, x, y)
        if not can_save_img:
            raise (f"x{x}, y{y}, z{z}のタイル画像保存に失敗しました")
        
        return xyz


    def make_downleveltiles(self, xyz_dicts, tiles_folder):
        new_xyz_dicts = [{"x":int(xyz_dict["x"]//2), "y":int(xyz_dict["y"]//2), "z":int(xyz_dict["z"]-1)} for xyz_dict in xyz_dicts]
        df = pd.DataFrame(new_xyz_dicts).dropna().drop_duplicates()
        new_xyz_dicts = df.to_dict(orient="records")
        created_xyz_lists = [self.make_downlevel(tiles_folder, xyz) for xyz in new_xyz_dicts]

        z = created_xyz_lists[0]["z"]
        if z == self.z_min:
            return 
        else:
            xy_df = pd.DataFrame(created_xyz_lists).dropna().drop_duplicates()
            created_xyz_lists = xy_df.to_dict(orient="records")

            return self.make_downleveltiles(created_xyz_lists, tiles_folder)


    def make_uplevel(self, tiles_folder, xyz):
        x = int(xyz["x"])
        y = int(xyz["y"])
        z = int(xyz["z"])
        
        tile_file = f"{tiles_folder}/{z}/{x}/{y}.png"
        if not os.path.isfile(tile_file):
            return {}
        
        img_arr = cv2.imread(tile_file, -1)
        H, W, C = img_arr.shape
        img_exp = np.empty((H*2, W*2, C))
        hw_lists = itertools.product(range(H), range(W))
        for i, j in hw_lists:
            img_exp[2*i,2*j,:] = img_arr[i,j,:]
            img_exp[2*i+1,2*j,:] = img_arr[i,j,:]
            img_exp[2*i,2*j+1,:] = img_arr[i,j,:]
            img_exp[2*i+1,2*j+1,:] = img_arr[i,j,:]
        img_lists = [img_exp[H*x:H*(x+1), W*y:W*(y+1)] for x in range(2) for y in range(2)]

        uptiles_dicts = [
            {"x":x*2, "y":y*2, "z":z+1},
            {"x":x*2+1, "y":y*2, "z":z+1},
            {"x":x*2, "y":y*2+1, "z":z+1},
            {"x":x*2+1, "y":y*2+1, "z":z+1}
        ]

        created_xyz_dicts = [
            {"xyz":uptiles_dict, "img":img} for uptiles_dict, img in zip(uptiles_dicts, img_lists) 
            if not np.allclose(img, self.arr_shade_inv)]

        [self.arr2png(d["img"], tiles_folder, d["xyz"]["z"], d["xyz"]["x"], d["xyz"]["y"]) for d in created_xyz_dicts]

        return [d["xyz"] for d in created_xyz_dicts]


    # 拡大レベルのタイル作成
    def make_upleveltiles(self, xyz_dicts, tiles_folder):
        created_xyz_lists = [self.make_uplevel(xyz_dict, tiles_folder) for xyz_dict in xyz_dicts]
        created_xyz_dicts = [d for xyz_dict in created_xyz_lists for d in xyz_dict]

        # 最大ズームレベルを作り終えると終了
        created_z = created_xyz_dicts[0]["z"]
        if created_z==self.z_max:
            return

        # 最大ズームレベルまで再帰処理
        return self.make_upleveltiles(created_xyz_dicts, tiles_folder)

    def make_tiles(self, df, tiles_folder, duplicate="m", is_nearest=None):
        # ベースレベルのタイル作成
        xyz_dicts = self.make_basetiles(df, tiles_folder, duplicate, is_nearest)
        # 縮小レベルのタイル作成
        self.make_downleveltiles(xyz_dicts, tiles_folder)
        # 拡大レベルのタイル作成
        self.make_upleveltiles(xyz_dicts, tiles_folder)

        return


def main(args):
    input, output, tiles_folder = get_paths(args)
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

    csv2tiles(csv_path_lists, tiles_folder)

    return

def csv2tiles(csv_path_lists, tiles_folder):
    xyz_dicts = make_basetiles(csv_path_lists, tiles_folder)
    make_downlevel_tiles(xyz_dicts, tiles_folder)
    make_uplevel_tiles(xyz_dicts, tiles_folder)

    return

def make_uplevel_tiles(xyz_dicts, tiles_folder):
    new_xyz_lists = [make_uplevel(tiles_folder, xyz) for xyz in xyz_dicts]

def make_uplevel(tiles_folder, xyz):
    x = xyz["x"]
    y = xyz["y"]
    z = xyz["z"]
    basetile_file = f"{tiles_folder}/{z}/{x}/{y}.png"
    if not os.path.isfile(basetile_file):
        return {}
    
    img = cv2.imread(basetile_file)
    H, W, C = img.shape
    img_exp = np.empty((H*2, W*2, C))










if __name__ == "__main__":
    main(sys.argv)
    # input: "/mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A"
    # output: "/mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A/csv"
    # tilepath: "/home/shun/work/learnings/python/xml2tiles/src/tiles"
    # cmd
    # python3 app.py /mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A /mnt/c/users/shun_/Downloads/PackDLMap/FG-GML-5537-05-DEM5A/csv /home/shun/work/learnings/python/xml2tiles/src/tiles
    