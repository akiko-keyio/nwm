from collections import defaultdict
import numpy as np
import xarray as xr
from joblib import delayed
from metpy.units import units
from scipy.integrate import cumulative_simpson
from loguru import logger
from scipy.interpolate import RectSphereBivariateSpline
from nwm.ztd_met import zhd_saastamoinen, zwd_saastamoinen
from nwm.geoid import GeoidHeight
from nwm.height_convert import geopotential_to_geometric
from nwm.Interprepter import (
    LinearInterpolator,
    LogCubicSplineInterpolator,
    LogLinearInterpolator,
)
from tqdm_joblib import ParallelPbar


class ZTDNWMGenerator:
    def __init__(
            self,
            nwm_path,
            location=None,
            egm_type="egm96-5",#"egm96-5"
            vertical_level="pressure_level",
            gravity_variation="latitude",
            refractive_index="mode2",
            compute_e_mode="mode2",
            p_interp_step=None,
            swap_interp_step=None,
            n_jobs=-3,
            batch_size=1000,
    ):

        self.nwm_path = nwm_path
        self.location = location.copy()
        self.egm_type = egm_type
        self.vertical_dimension = vertical_level
        self.gravity_variation = gravity_variation
        self.refractive_index = refractive_index
        self.compute_e_mode = compute_e_mode
        self.p_interp_step = p_interp_step
        self.swap_interp_step = swap_interp_step
        self.n_jobs = n_jobs
        self.ds = None
        self.ds_site = None
        self.ds_sap = None
        self.batch_size = batch_size


    def read_met_file(self):

        self.ds = xr.open_dataset(self.nwm_path)    # ← 关闭时不要写回)

        lon = self.ds.coords["longitude"]
        if (lon > 180).any():
            logger.info("Check Longitude contain (0,360), transform to (-180,180]")
            self.ds = self.ds.assign_coords(
                longitude=((lon + 180) % 360) - 180
            ).sortby("longitude")

        # self.ds=xr.open_dataset(self.nwm_path, chunks='auto')
        rename_map = {
            "level": "pressure_level",
            "isobaricInhPa": "pressure_level",
            "valid_time": "time"
        }
        # 过滤出实际存在的键
        existing = {k: v for k, v in rename_map.items() if k in self.ds}

        # 先重命名
        if existing:
            logger.info(f"Renaming dimensions: {existing}")
            self.ds = self.ds.rename(existing)

        try:
            self.ds = self.ds.expand_dims("time")
        except:
            pass

        try:
            self.ds = self.ds.expand_dims("number")
        except:
            pass

    def horizental_interpolate(self):
        if self.location is None:
            self.ds = self.ds.stack(site_index=("latitude", "longitude"))
            self.ds["site"] = self.ds.site_index
            self.ds["alt"] = 0
            return

        self.ds = self.ds.sortby("longitude")
        self.ds = self.ds.sortby("latitude")

        lon = self.ds.longitude.values
        lon_step = lon[1] - lon[0]
        is_global = (np.isclose(lon.min(), 0)
                     and np.isclose(360 % lon_step, 0)
                     and np.isclose(lon.max() + lon_step, 360))

        if is_global:
            # 用 pad(mode="wrap") 在 longitude 轴尾部复制第一个切片
            # pad_width=(0,1) 表示在前端不扩，后端 +1
            logger.info("Padding Coordinates")
            self.ds = self.ds.pad(longitude=(0, 1), mode="wrap")
            # 重新设置坐标：原有 lon 加上 360
            new_lon = np.append(lon, 360.0)
            self.ds = self.ds.assign_coords(longitude=new_lon)

        self.location["site_index"] = self.location.index
        lon = xr.DataArray(
            self.location["lon"],
            dims="site_index",
            coords={"site_index": self.location["site_index"]},
        )
        lat = xr.DataArray(
            self.location["lat"],
            dims="site_index",
            coords={"site_index": self.location["site_index"]},
        )
        alt = xr.DataArray(
            self.location["alt"],
            dims="site_index",
            coords={"site_index": self.location["site_index"]},
        )
        site = xr.DataArray(
            self.location["site"],
            dims="site_index",
            coords={"site_index": self.location["site_index"]},
        )
        logger.info("Interpolate to site")
        self.ds = self.ds.interp(longitude=lon, latitude=lat, method="linear",assume_sorted =True)
        self.ds = self.ds.rename({"longitude": "lon", "latitude": "lat"})
        self.ds = xr.merge([self.ds, alt, site])
        self.ds.coords["alt"] = self.ds.alt



    def quantify_met_parameters(self):
        self.ds["p"] = self.ds.pressure_level * units.hPa
        self.ds = self.ds.metpy.quantify()

    def geopotential_to_orthometric(self):

        if self.gravity_variation == "ignore":
            # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-SpatialreferencesystemsandEarthmodel
            Re = 6371.2229e3 * units.meters
            G = 9.80665 * units.meters / units.second ** 2
            geopotential_height = self.ds["z"] / G
            self.ds["h"] = (geopotential_height * Re) / (Re - geopotential_height)

        elif self.gravity_variation == "latitude":
            G = 9.80665 * units.meters / units.second ** 2
            geopotential_height = self.ds.z / G
            if len(self.ds.lat.shape) == 1:
                lat = np.expand_dims(self.ds.lat, axis=(0, 1))
            else:
                lat = self.ds.lat
            h = geopotential_to_geometric(
                latitude=lat, geopotential_height=geopotential_height
            )

            self.ds["h"] = h
        else:
            raise ValueError("Gravity variation must be either ignore or latitude")

    def orthometric_to_ellipsoidal0(self):
        geoid = GeoidHeight(egm_type=self.egm_type)

        def get_geoid_height(lat, lon):
            return geoid.get(lat, lon)

        gravity_anomaly = (
                xr.apply_ufunc(
                    get_geoid_height,
                    self.ds["lat"],
                    self.ds["lon"],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )
                * units.meters
        )

        self.ds["h"] = self.ds["h"] + gravity_anomaly

    def orthometric_to_ellipsoidal(self):
        geoid = GeoidHeight(egm_type=self.egm_type)

        # 原 ds 在水平插值后维度为 (time, vertical, site_index)
        # ds.lat/ ds.lon 只有 site_index 维度
        site_indices = self.ds.site_index.values
        lat_vals = self.ds["lat"].values
        lon_vals = self.ds["lon"].values

        # 1. 对每个站点单独调用 geoid.get，保证 ix/iy 都是整数
        anom_list = []
        for lat, lon in zip(lat_vals, lon_vals):
            # 转成 Python float，万无一失
            anom_list.append(geoid.get(float(lat), float(lon)))
        # 转 numpy array，加上单位
        anomaly = np.array(anom_list) * units.meters

        # 2. 构造一个只带 site_index 维度的 DataArray
        anom_da = xr.DataArray(
            anomaly,
            dims=("site_index",),
            coords={"site_index": site_indices},
            name="gravity_anomaly",
        )

        # 3. 自动广播到 (time, vertical, site_index) 并相加
        self.ds["h"] = self.ds["h"] + anom_da

        return

    def compute_e(self):
        # 水汽压 / 湿空气的分压
        if self.compute_e_mode == "mode1":
            self.ds["e"] = (self.ds["q"] * self.ds["p"]) / 0.622
        elif self.compute_e_mode == "mode2":
            self.ds["e"] = (self.ds["q"] * self.ds["p"]) / (
                    0.622 + 0.378 * self.ds["q"]
            )

    def resample_to_ellipsoidal(self):
        if self.vertical_dimension == "pressure_level":
            return
        ds = self.ds
        location = self.location
        h_max = ds.sel(pressure_level=ds.pressure_level.min()).h.min()

        target_h = np.arange(location.alt.min(), float(h_max), 50) * units.meters

        def interpolate_for_site_time(site_index, time, param_name, interpolator):
            site_zenith_data = ds.sel(site_index=site_index, time=time)
            x = np.flip(site_zenith_data.h.values)
            y = np.flip(site_zenith_data[param_name].values)
            interpolated_values = interpolator(x, y).interpolate(target_h)
            return site_index, time, param_name, interpolated_values

        # parallelize the computation
        results = ParallelPbar("Resample to Ellipsoidal")(n_jobs=self.n_jobs)(
            delayed(interpolate_for_site_time)(
                site_index, time, param_name, interpolator
            )
            for site_index in ds.site_index.values
            for time in ds.time.values
            for param_name, interpolator in [
                ("e", LogLinearInterpolator),
                ("p", LogLinearInterpolator),
                ("t", LinearInterpolator),
            ]
        )

        # save values
        results_dict = defaultdict(list)
        for r in results:
            results_dict[r[2]].append((r[0], r[1], r[3]))

        ds_swap = xr.Dataset(
            coords={"site_index": ds.site_index, "time": ds.time, "h": target_h}
        )
        for param_name in ["e", "p", "t"]:
            da = xr.DataArray(
                dims=["site_index", "time", "h"],
                coords={"site_index": ds.site_index, "time": ds.time, "h": target_h},
            )

            interp_array = np.full(
                (len(ds.site_index), len(ds.time), len(target_h)), np.nan
            )
            site_index_map = {
                site: idx for idx, site in enumerate(ds.site_index.values)
            }
            time_map = {time: idx for idx, time in enumerate(ds.time.values)}

            for site_index, time, interp_values in results_dict[param_name]:
                site_idx = site_index_map[site_index]
                time_idx = time_map[time]
                interp_array[site_idx, time_idx, :] = interp_values

            da.loc[dict(site_index=ds.site_index, time=ds.time, h=target_h)] = (
                interp_array
            )
            ds_swap[param_name] = da * ds[param_name].metpy.units

        ds_swap["lon"] = ds["lon"]
        ds_swap["lat"] = ds["lat"]
        ds_swap["site"] = ds["site"]
        ds_swap = ds_swap.sortby("h", ascending=False)
        self.ds = ds_swap

    def compute_refractive_index(self):

        # # GNSS定位定时中的对流层延迟模型优化研究_苏行
        # k1 = 77.604 * units.kelvin / units.hPa
        # k2 = 69.4 * units.kelvin / units.hPa
        # k3 = 370100 * units.kelvin ** 2 / units.hPa

        # Establishing a high‑precision real‑time ZTD model of China with GPS  and ERA5 historical datas and its
        # application in PPP
        # k1 = 77.6 * units.kelvin / units.hPa
        # k2 = 70.4 * units.kelvin / units.hPa
        # k3 = 373900 * units.kelvin ** 2 / units.hPa

        # # Real-time precise point positioning augmented with high-resolution numerical weather prediction model
        # k1 = 77.689 * units.kelvin / units.hPa
        # k2 = 71.2952 * units.kelvin / units.hPa
        # k3 = 375463 * units.kelvin ** 2 / units.hPa

        t = self.ds.t  # temperature
        e = self.ds.e  # partial pressure of water vapor
        p = self.ds.p  # total pressure of (moist) air
        p_d = p - e  # partial pressure of dry air

        # Troposphere modeling and filtering for precise gps leveling
        #     delft, the netherlands, 2004, equation (4.36)
        k1 = 77.689 * units.kelvin / units.hPa
        k2 = 71.2952 * units.kelvin / units.hPa
        k3 = 375463 * units.kelvin ** 2 / units.hPa
        r_d = (
                287.06 * units.joule / (units.kilogram * units.kelvin)
        )  # specific gas constant of dry air
        r_v = (
                461.525 * units.joule / (units.kilogram * units.kelvin)
        )  # specific gas constant of water vapor
        k2_ = k2 - k1 * r_d / r_v

        # Divides the refractivity into dry and vapour part
        # inverse compressibility factor of dry air
        z_d_inv = 1 + p_d * (
                57.90e-8 / units.hPa
                - (9.4581e-4 * units.kelvin / units.hPa) * t ** -1
                + (0.25844 * units.kelvin ** 2 / units.hPa) * t ** -2
        )
        # inverse compressibility factor of water vapor
        z_v_inv = 1 + e * (1 + (3.7e-4 / units.hPa) * e) * (
                -2.37321e-3 / units.hPa
                + (2.23366 * units.kelvin / units.hPa) * t ** -1
                - (710.792 * units.kelvin ** 2 / units.hPa) * t ** -2
                + (7.75141e4 * units.kelvin ** 3 / units.hPa) * t ** -3
        )
        n_d = z_d_inv * k1 * p_d / t  # refractivity of dry air
        n_v = z_v_inv * (k2 * e / t + k3 * e / t ** 2)  # refractivity of water vapour

        # Divides the refractivity into hydrostatic and a non-hydrostatic part
        rho_d = p_d / (r_d * t)  # density of dry air
        rho_v = e / (r_v * t)  # density of water vapor
        rho_m = rho_d + rho_v  # density of moist air
        n_h = k1 * r_d * rho_m  # hydrostatic refractivity.
        n_w = (
                      k2_ * e / t  # non-hydrostatic refractivity (wet refractivity)
                      + k3 * e / t ** 2
              ) * z_v_inv

        if self.refractive_index == "mode2":
            self.ds["n"] = n_w + n_h
            self.ds["n_w"] = n_w
            self.ds["n_h"] = n_h
        elif self.refractive_index == "mode1":
            self.ds["n"] = n_d + n_v
            self.ds["n_w"] = n_w
            self.ds["n_h"] = n_d + n_v - n_w
        else:
            raise ValueError("refractive_index must be mode1 or mode2")

        self.ds["n_d"] = n_d
        self.ds["n_v"] = n_v
        self.ds["z_d_inv"] = z_d_inv
        self.ds["z_v_inv"] = z_v_inv

    def compute_top_level_delay(self):
        ds = self.ds
        if self.vertical_dimension == "pressure_level":
            top_level = ds.sel(pressure_level=ds.pressure_level.min())
        elif self.vertical_dimension == "h":
            top_level = ds.sel(h=ds.h.max())
        else:
            raise ValueError(
                "vertical_dimension must be either pressure_level or ellipsoidal_height"
            )

        top_level = top_level.transpose("number", "time", "site_index")
        top_level["zwd"] = units.meters * zwd_saastamoinen(
            e=top_level.e.metpy.dequantify(), t=top_level.t.metpy.dequantify()
        )
        top_level["zhd"] = units.meters * zhd_saastamoinen(
            p=top_level.p.metpy.dequantify(),
            lat=top_level.lat.metpy.dequantify(),
            alt=top_level.h.metpy.dequantify(),
        )
        top_level["ztd"] = top_level["zwd"] + top_level["zhd"]
        self.top_level = top_level

    def simpson_numerical_integration(self):
        ds = self.ds
        if self.vertical_dimension == "pressure_level":
            ds = ds.sortby("pressure_level")

        x = -ds.transpose("number", "time", "site_index", self.vertical_dimension).h

        def intergration(y):
            y = y.transpose("number", "time", "site_index", self.vertical_dimension)
            return cumulative_simpson(y=y, x=x, axis=-1, initial=0)

        # tropospheric delay intergration
        for zxd, n in [("ztd", "n"), ("zwd", "n_w"), ("zhd", "n_h")]:
            zxd_value = intergration(y=ds[n] * 1.0e-6) * units.meters

            ds[f"{zxd}_simpson"] = xr.DataArray(
                data=zxd_value, dims=["number", "time", "site_index", self.vertical_dimension]
            )
            ds[f"{zxd}_simpson"] = ds[f"{zxd}_simpson"] + self.top_level[zxd]

            # # np.expand_dims(B, axis=-1)
            # zxd_value = zxd_value + np.expand_dims(self.top_level[zxd].values, axis=-1)#.values#.reshape(1, -1)
            # self.ds[f'{zxd}_simpson'] = xr.DataArray(data=zxd_value, dims=['time', 'site_index','pressure_level'])
        self.ds = ds
        # refractive grid intergration
        # k1_grad = intergration((self.ds['p'] - self.ds['e']) / self.ds['t'])
        # k2_grad = intergration(self.ds['e'] / self.ds['t'])
        # k3_grad = intergration(self.ds['e'] / (self.ds['t'] ** 2))
        # self.ds[f'k1_grad'] = xr.DataArray(data=k1_grad, dims=['time', 'site_index', 'pressure_level'])
        # self.ds[f'k2_grad'] = xr.DataArray(data=k2_grad, dims=['time', 'site_index', 'pressure_level'])
        # self.ds[f'k3_grad'] = xr.DataArray(data=k3_grad, dims=['time', 'site_index', 'pressure_level'])

    def vertical_interpolate_to_site0(self):
        """对 ztd_simpson 做竖直插值；支持 number 维；按 time 分批并行。"""
        ds = self.ds

        # ------------ 内部工具：插值单个  (number, site_index, time_batch) ----------
        def interp_one_batch(ds_site, number, site_index, times, param):
            target_alt = ds_site.alt.values  # shape (1,)

            # 2-D 数组： (time, vertical)
            x_all = ds_site.h.sel(number=number).values
            y_all = ds_site[param].sel(number=number).values

            out = []
            for tidx, t in enumerate(times):
                # 1-D 剖面插值 → 单值
                interp_val = LogCubicSplineInterpolator(
                    x_all[tidx], y_all[tidx]
                ).interpolate(target_alt)
                out.append((number, site_index, t, param, interp_val))
            return out

        # ------------ 并行调度 ----------------------------------------
        param_names = ["ztd_simpson"]
        numbers = ds.number.values if "number" in ds.dims else np.array([0])
        site_indices = ds.site_index.values
        times = ds.time.values

        # 按批切 time
        tbsz = self.batch_size
        time_batches = [times[i:i + tbsz] for i in range(0, len(times), tbsz)]

        tasks = (
            delayed(interp_one_batch)(
                ds.sel(site_index=si),  # 传进整站数据 (含全部 time & number)
                mem, si, t_batch, pnm
            )
            for si in site_indices
            for mem in numbers
            for pnm in param_names
            for t_batch in time_batches
        )

        results = ParallelPbar("Vertical Interpolate to site")(n_jobs=self.n_jobs,prefer='processes')(tasks)
        results = [item for sub in results for item in sub]  # flatten

        # ------------ 写回 Dataset -----------------------------------
        ds_site = xr.Dataset(
            coords={
                "number": numbers,
                "site_index": site_indices,
                "time": times,
            }
        )
        ds_site["site"] = ds.site

        # 映射表
        midx = {m: i for i, m in enumerate(numbers)}
        sidx = {s: i for i, s in enumerate(site_indices)}
        tidx = {t: i for i, t in enumerate(times)}

        # 收集并写入
        for pnm in param_names:
            data = np.full((len(numbers), len(site_indices), len(times), 1), np.nan)
            for mem, si, t, _, val in results:
                data[midx[mem], sidx[si], tidx[t], 0] = val

            da = xr.DataArray(
                data=data,
                dims=["number", "site_index", "time", "h"],
                coords={
                    "number": numbers,
                    "site_index": site_indices,
                    "time": times,
                    # h 用原剖面最高层；这里只是形状占位，值无所谓
                    "h": ds.h.isel({self.vertical_dimension: -1}),
                },
            ) * ds[pnm].metpy.units * 1000.0  # 同原代码

            ds_site[pnm] = da

        self.ds_site = ds_site
        return ds_site

    def vertical_interpolate_to_site(self):
        """
        对 ztd_simpson 做竖直插值；支持 number 维；
        先按 time 再按 member 分块并行，任务维度：site × mem_block × time_block
        """
        import numpy as np
        import xarray as xr
        from joblib import delayed

        ds = self.ds

        # ---------- 常量：硬编码分块大小 ----------
        TIME_CHUNK = 10  # 每个任务最多处理 200 个 time step
        MEM_CHUNK = 50  # 每个任务最多处理 10 个 ensemble member

        # ---------- 内部工具：插值单个 mem_block × time_block ----------
        def interp_one_batch(ds_site, mem_block, site_index, times_block, param):
            """
            返回 [(mem, site_index, time, param, value), ...]
            """
            target_alt = ds_site.alt.values  # shape (1,)

            results = []
            for mem in mem_block:
                x_all = ds_site.h.sel(number=mem).values
                y_all = ds_site[param].sel(number=mem).values

                for tidx, t in enumerate(times_block):
                    interp_val = LogCubicSplineInterpolator(
                        x_all[tidx], y_all[tidx]
                    ).interpolate(target_alt)
                    results.append((mem, site_index, t, param, interp_val))
            return results

        # ---------- 并行调度 ----------
        param_names = ["ztd_simpson"]
        numbers = ds.number.values if "number" in ds.dims else np.array([0])
        site_indices = ds.site_index.values
        times = ds.time.values

        # time / member 双重分块
        time_blocks = [times[i: i + TIME_CHUNK] for i in range(0, len(times), TIME_CHUNK)]
        mem_blocks = [numbers[i: i + MEM_CHUNK] for i in range(0, len(numbers), MEM_CHUNK)]

        tasks = (
            delayed(interp_one_batch)(
                ds.sel(site_index=si),
                mem_block, si, t_block, pnm
            )
            for si in site_indices
            for mem_block in mem_blocks
            for pnm in param_names
            for t_block in time_blocks
        )

        # 运行并 flatten
        results = ParallelPbar("Vertical Interpolate to site")(
            n_jobs=self.n_jobs, prefer='processes'
        )(tasks)
        results = [item for sub in results for item in sub]

        # ---------- 写回 Dataset ----------
        ds_site = xr.Dataset(
            coords={
                "number": numbers,
                "site_index": site_indices,
                "time": times,
            }
        )
        ds_site["site"] = ds.site

        # 映射表
        midx = {m: i for i, m in enumerate(numbers)}
        sidx = {s: i for i, s in enumerate(site_indices)}
        tidx = {t: i for i, t in enumerate(times)}

        for pnm in param_names:
            data = np.full((len(numbers), len(site_indices), len(times), 1), np.nan)
            for mem, si, t, _, val in results:
                data[midx[mem], sidx[si], tidx[t], 0] = val

            da = xr.DataArray(
                data=data,
                dims=["number", "site_index", "time", "h"],
                coords={
                    "number": numbers,
                    "site_index": site_indices,
                    "time": times,
                    # h 用原剖面最高层；这里只是形状占位
                    "h": ds.h.isel({self.vertical_dimension: -1}),
                },
            ) * ds[pnm].metpy.units * 1000.0  # 单位与原实现一致

            ds_site[pnm] = da

        self.ds_site = ds_site
        return ds_site

    def run(self, time_select=None):
        logger.info(f"Start ZTD computation (vertical_dimension={self.vertical_dimension})")

        logger.info("1/11: Reading meteorological file")
        self.read_met_file()

        logger.info("2/11: Performing horizontal interpolation")
        self.horizental_interpolate()

        logger.info("3/11: Quantifying meteorological parameters (units)")
        self.quantify_met_parameters()

        logger.info("4/11: Converting geopotential to orthometric height")
        self.geopotential_to_orthometric()

        logger.info("5/11: Converting orthometric to ellipsoidal height")
        self.orthometric_to_ellipsoidal()

        logger.info("6/11: Computing water vapor pressure (e)")
        self.compute_e()

        if self.vertical_dimension != "pressure_level":
            logger.info("7/11: Resampling to ellipsoidal vertical grid")
            self.resample_to_ellipsoidal()
        else:
            logger.info("7/11: Skipping resampling (using pressure levels)")

        logger.info("8/11: Computing refractive index (n, n_h, n_w)")
        self.compute_refractive_index()

        logger.info("9/11: Computing top-level delays (zwd, zhd, ztd)")
        self.compute_top_level_delay()

        logger.info("10/11: Performing Simpson integration and vertical interpolation")
        self.simpson_numerical_integration()

        logger.info("11/11: Vertical interpolation")
        ds_site = self.vertical_interpolate_to_site()

        logger.info("ZTD computation finished; returning DataFrame")
        return ds_site.to_dataframe().reset_index().drop(columns=['site_index','h'])
