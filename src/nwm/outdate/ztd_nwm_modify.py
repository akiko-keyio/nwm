"""Fast ZTD generator based on NWM / ERA-like 3-D meteorological files.

This module implements a high performance workflow for deriving Zenith
Tropospheric Delay (ZTD) from numerical weather prediction data.  The
computation follows the eleven steps originally implemented in
``ztd_nwm.py``:

1. Read and pre-process the meteorological file.
2. Format dataset dimensions and longitude.
3. Perform horizontal interpolation to station coordinates.
4. Quantify meteorological parameters with proper units.
5. Convert geopotential height to orthometric height.  The formulae are
   taken from the ERA5 documentation_.
6. Transform orthometric height to ellipsoidal height using the selected
   geoid model.
7. Compute water vapour pressure ``e``.
8. Optionally resample to an ellipsoidal vertical grid.
9. Derive refractive indices.  Recommended constants come from
   "Troposphere modeling and filtering for precise GPS leveling" [2004].
10. Compute top level hydrostatic and wet delays.
11. Integrate refractivity using Simpson's rule.
12. Interpolate ZTD to the exact station altitude.

References
----------
`ERA5 data documentation`_
"GNSS定位定时中的对流层延迟模型优化研究", 苏行
"Establishing a high‑precision real‑time ZTD model of China with GPS and
ERA5 historical datas and its application in PPP"
"Real-time precise point positioning augmented with high-resolution
numerical weather prediction model"

.. _ERA5 data documentation: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#heading-SpatialreferencesystemsandEarthmodel
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from metpy.units import units
from scipy.integrate import cumulative_simpson
from scipy.interpolate import (
    CubicSpline,
    RegularGridInterpolator,
    RectSphereBivariateSpline,
)
from tqdm_joblib import ParallelPbar

from nwm.Interprepter import LinearInterpolator, LogLinearInterpolator
from nwm.geoid import GeoidHeight
from nwm.height_convert import geopotential_to_geometric
from nwm.ztd_met import zhd_saastamoinen, zwd_saastamoinen


class ZTDNWMGenerator:
    # ----------------------------- 初始化 ---------------------------- #
    def __init__(
        self,
        nwm_path: str | Path,
        location: xr.Dataset | None = None,
        egm_type: str = "egm96-5",
        vertical_level: str = "pressure_level",
        gravity_variation: str = "lat",
        refractive_index: str = "mode2",
        compute_e_mode: str = "mode2",
        p_interp_step: int | None = None,
        swap_interp_step: int | None = None,
        n_jobs: int = -1,
        batch_size: int = 100_000,
        load_method: str = "auto",
        horizental_interpolation_method: str = "linear",
    ):
        self.nwm_path = Path(nwm_path)
        self.location = location.copy() if location is not None else None
        self.egm_type = egm_type
        self.vertical_dimension = vertical_level
        self.gravity_variation = gravity_variation
        self.refractive_index = refractive_index
        self.compute_e_mode = compute_e_mode
        self.p_interp_step = p_interp_step
        self.swap_interp_step = swap_interp_step
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.load_method = load_method
        self.horizental_interpolation_method = horizental_interpolation_method
        self.merge_ground=True

        self.ds: xr.Dataset | None = None
        self.ds_site: xr.Dataset | None = None
        self.top_level: xr.Dataset | None = None

    # -------------------------- 1 读文件 -------------------------- #
    def read_met_file(self) -> None:
        t0 = time.perf_counter()

        def _load_stream() -> xr.Dataset:
            import fsspec

            mem_url = "memory://temp.nc"
            with (
                self.nwm_path.open("rb") as fsrc,
                fsspec.open(mem_url, "wb") as fdst,
            ):
                shutil.copyfileobj(fsrc, fdst, length=1024 << 20)
            with fsspec.open(mem_url, "rb") as f:
                ds = xr.open_dataset(f)
                ds.load()
            return ds

        def _is_hdf5(path: Path, blocksize: int = 8) -> bool:
            """Check whether file header matches the HDF5 signature."""

            with path.open("rb") as f:
                header = f.read(blocksize)
            return header == b"\x89HDF\r\n\x1a\n"

        def _load_memory() -> xr.Dataset:
            engine = "h5netcdf" if _is_hdf5(self.nwm_path) else None
            try:
                ds = xr.open_dataset(
                    self.nwm_path,
                    engine=engine,
                    chunks="auto",
                    mask_and_scale=True,
                    decode_times=True,
                ).load()
            except Exception as e:
                logger.warning(
                    f"{engine or 'default'} backend failed ({e}); retry with default engine"
                )
                ds = xr.open_dataset(self.nwm_path).load()
            return ds

        def _load_lazy() -> xr.Dataset:
            return xr.open_dataset(self.nwm_path, chunks="auto")

        def _load_zarr() -> xr.Dataset:
            return xr.open_dataset(self.nwm_path, engine="zarr").load()
        method = self.load_method
        loaders = {
            "stream": _load_stream,
            "memory": _load_memory,
            "lazy": _load_lazy,
            "zarr":_load_zarr
        }

        if method == "auto":
            for name in ("stream", "memory", "lazy"):
                try:
                    self.ds = loaders[name]()
                    logger.info(f"Loaded {name}")
                    break
                except Exception as e:  # pragma: no cover - log only
                    logger.warning(f"{name} loading failed ({e})")
            else:
                raise RuntimeError("Failed to load dataset")
        else:
            if method not in loaders:
                raise ValueError(f"Unknown load_method: {method}")
            self.ds = loaders[method]()
        logger.info(
            f"1/12: Reading meteorological file done in {time.perf_counter() - t0:.2f}s"
        )

    def format_dataset(self):
        t0 = time.perf_counter()
        # === 后续维度重命名与补维保持不变 ===
        rename_map = {
            "level": "pressure_level",
            "isobaricInhPa": "pressure_level",
            "valid_time": "time",
            "longitude": "lon",
            "latitude": "lat",
        }
        exist = {k: v for k, v in rename_map.items() if k in self.ds}
        if exist:
            logger.info(f"Renaming dimensions: {exist}")
            self.ds = self.ds.rename(exist)

        if "time" not in self.ds.dims:
            logger.info(f"Expanding Time")
            self.ds = self.ds.expand_dims("time")
        if "number" not in self.ds.dims:
            logger.info(f"Expanding Number")
            self.ds = self.ds.expand_dims("number")

        def normalize_longitude(
            ds: xr.Dataset, *, drop_duplicate: bool = True, roll: bool = True
        ) -> xr.Dataset:
            if drop_duplicate and "lon" in ds.data_vars:
                ds = ds.drop_vars("lon")
            lon = ds.lon
            if not (lon > 180).any():
                return ds
            logger.info("Normalizing longitude")
            new_lon = ((lon + 180) % 360) - 180
            if roll:
                shift = int((new_lon < 0).sum())
                ds = ds.roll(lon=shift, roll_coords=True)
                new_lon = new_lon.roll(lon=shift)

            ds = ds.assign_coords(lon=("lon", new_lon.values))
            return ds

        self.ds = normalize_longitude(self.ds)

        def expand_p_dimension(self):
            ds = self.ds
            if self.merge_ground:
                ds['p'] = ds.pressure_level.expand_dims({
                    'lat': ds.lat,  # size 721
                    'lon': ds.lon,  # size 1440
                    'number': ds.number,  # size 1
                    'time': ds.time,  # size 1
                }).transpose('number', 'time', 'pressure_level', 'lat', 'lon')
            else:
                ds['p'] = ds.pressure_level

        expand_p_dimension(self)
        logger.info(f"2/12: Format file done in {time.perf_counter() - t0:.2f}s")

    # ---------------------- 2  水平插值 --------------------------- #
    def horizental_interpolate(self) -> None:
        """
        Horizontal interpolation for NWM fields.

        Parameters
        ----------
        method : {"linear", "sphere_spline"}
            "linear"         —— RegularGridInterpolator (平面线性，默认)
            "sphere_spline"  —— RectSphereBivariateSpline (球面样条)
        """

        t0 = time.perf_counter()
        method = self.horizental_interpolation_method

        # -------- 0. 目标为空：直接 stack --------
        if self.location is None:
            self.ds = self.ds.stack(site_index=("lat", "lon"))
            self.ds["site"] = self.ds.site_index
            # self.ds["alt"] = 0
            logger.info("No target sites: stacked original grid.")
            logger.info(
                f"3/12: Horizontal interpolation done in {time.perf_counter() - t0:.2f}s"
            )
            return

        # -------- 1. 通用准备 --------
        ds, loc = self.ds, self.location
        lat_grid = ds.lat.values  # 1-D
        lon_grid = ds.lon.values  # 1-D
        pts_lat = loc["lat"].values  # 目标站列表
        pts_lon = loc["lon"].values

        if method == "linear":
            query_pts = np.column_stack((pts_lat, pts_lon))

        elif method == "sphere_spline":
            # θ = π/2 − lat（必须递增且 ∈ (0,π)），φ wrap 到 [0,2π)
            theta_q = np.deg2rad(90.0 - pts_lat)
            phi_q = np.deg2rad(np.mod(pts_lon, 360.0))

            theta = np.deg2rad(90.0 - lat_grid)
            phi = np.deg2rad(np.mod(lon_grid, 360.0))
            theta_sort_idx = np.argsort(theta)
            phi_sort_idx = np.argsort(phi)
            theta = theta[theta_sort_idx]
            phi = phi[phi_sort_idx]

            eps = 1e-6
            theta = np.clip(theta, eps, np.pi - eps)
        else:
            raise ValueError(f"Unknown method '{method}'")

        new_vars = {}

        # -------- 2. 变量循环 --------
        for vn, da in ds.data_vars.items():
            logger.info(f"Interpolating {vn} with '{method}'")
            dims = da.dims

            if {"lat", "lon"} <= set(dims):
                lat_ax, lon_ax = dims.index("lat"), dims.index("lon")
                other_axes = [i for i in range(da.ndim) if i not in (lat_ax, lon_ax)]

                if method == "linear":
                    # ---- 2-A 线性 ----
                    arr2 = np.moveaxis(
                        da.values,
                        (lat_ax, lon_ax) + tuple(other_axes),
                        (0, 1) + tuple(2 + np.arange(len(other_axes))),
                    )
                    other_shape = arr2.shape[2:]  # ★ 修正点
                    interp = RegularGridInterpolator(
                        (lat_grid, lon_grid),
                        arr2,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    res = interp(query_pts)

                else:
                    # ---- 2-B 球面样条 ----
                    arr = np.moveaxis(da.values, (lat_ax, lon_ax), (0, 1))
                    arr = arr[theta_sort_idx][:, phi_sort_idx]
                    other_shape = arr.shape[2:]  # ★ 与线性统一
                    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)

                    res_list = []
                    for k in range(flat.shape[-1]):
                        spline = RectSphereBivariateSpline(
                            theta, phi, flat[..., k], s=0.0, pole_continuity=True
                        )
                        try:
                            res_list.append(spline.ev(theta_q, phi_q))
                        except ValueError:
                            res_list.append(np.full_like(theta_q, np.nan))
                    res = np.stack(res_list, axis=1)  # (nsite, n_flat)

                # ---- reshape & moveaxis ----
                if res.ndim > 1:
                    res2 = res.reshape(len(pts_lat), *other_shape)
                    res2 = np.moveaxis(res2, range(1, 1 + len(other_axes)), other_axes)
                else:
                    res2 = res[:, None]

                new_dims = tuple(
                    d for d in dims if d not in ("lat", "lon")
                ) + ("site_index",)
                coords = {d: ds.coords[d] for d in new_dims if d != "site_index"}
                coords["site_index"] = loc.index
                new_vars[vn] = xr.DataArray(res2, dims=new_dims, coords=coords, name=vn)
            else:
                new_vars[vn] = da  # 无 lat/lon，不动

        # -------- 3. 打包 Dataset --------
        ds2 = xr.Dataset(new_vars)
        ds2["lat"] = ("site_index", pts_lat)
        ds2["lon"] = ("site_index", pts_lon)
        ds2["alt"] = ("site_index", loc["alt"].values)
        ds2["site"] = ("site_index", loc["site"].values)
        ds2.coords["alt"] = ds2.alt

        self.ds = ds2
        logger.info(
            f"3/12: Horizontal interpolation done in {time.perf_counter() - t0:.2f}s"
        )



    # --------------------------- 3 量纲 --------------------------- #
    def quantify_met_parameters(self) -> None:
        t0 = time.perf_counter()

        # 1. 解析 CF 元数据并尝试自动量化
        ds = self.ds.metpy.parse_cf()
        try:
            ds = ds.metpy.quantify()
        except Exception as e:
            # 如果整体量化失败，记录一下，但不影响后续手工转换
            logger.warning(f"metpy.quantify() failed: {e}")

        # 2. 定义手工转换映射
        conversions = {
            "p": ("p", lambda da: da * units.hPa),
            "z": ("z", lambda da: da * units.meters ** 2 / units.second ** 2),
            "t": ("t", lambda da: da * units.kelvin),
        }

        # 3. 过滤出：在 ds 中存在且自动量化后无单位（或 dimensionless）的那些变量
        to_convert = {
            src: (dst, fn)
            for src, (dst, fn) in conversions.items()
            if src in ds and (
                    getattr(ds[src].metpy, "units", None) is None
                    or getattr(ds[src].metpy.units, "is_dimensionless", True)
            )
        }
        logger.info(f"Converting units: {to_convert.keys()}")

        # 4. 对过滤后的字典一次性做手工转换
        for src, (dst, fn) in to_convert.items():
            ds[dst] = fn(self.ds[src])

        # 5. 回写并结束
        self.ds = ds
        elapsed = time.perf_counter() - t0
        # print(f"quantify_met_parameters 用时 {elapsed:.3f}s")

    # --------------- 4 Geopotential → Orthometric ---------------- #
    def geopotential_to_orthometric(self) -> None:
        t0 = time.perf_counter()
        if self.gravity_variation == "ignore":
            Re = 6371.2229e3 * units.meter
            G0 = 9.80665 * units.meter / units.second**2
            geop_height = self.ds["z"] / G0
            self.ds["h"] = geop_height * Re / (Re - geop_height)
        elif self.gravity_variation == "lat":
            G0 = 9.80665 * units.meter / units.second**2
            geop_height = self.ds.z / G0
            lat_vals = self.ds.lat
            if lat_vals.ndim == 1:
                lat_vals = np.expand_dims(lat_vals, axis=(0, 1))

            self.ds["h"] = geopotential_to_geometric(
                    latitude=lat_vals, geopotential_height=geop_height
                )


        else:
            raise ValueError("gravity_variation must be ignore or latitude")
        logger.info(
            f"5/12: Geopotential→Orthometric done in {time.perf_counter() - t0:.2f}s"
        )

    # ------------- 5 Orthometric → Ellipsoidal ------------------- #
    def orthometric_to_ellipsoidal(self) -> None:
        t0 = time.perf_counter()
        geoid = GeoidHeight(egm_type=self.egm_type)
        anomaly = (
            np.array(
                [
                    geoid.get(float(la), float(lo))
                    for la, lo in zip(self.ds.lat.values, self.ds.lon.values)
                ]
            )
            * units.meter
        )
        anom_da = xr.DataArray(
            anomaly, dims=("site_index",), coords={"site_index": self.ds.site_index}
        )
        self.ds["h"] = self.ds["h"] + anom_da
        logger.info(
            f"6/12: Orthometric→Ellipsoidal done in {time.perf_counter() - t0:.2f}s"
        )

    # ------------------------- 6 计算 e --------------------------- #
    def compute_e(self) -> None:
        t0 = time.perf_counter()
        if self.compute_e_mode == "mode1":
            self.ds["e"] = (self.ds.q * self.ds.p) / 0.622
        else:
            self.ds["e"] = (self.ds.q * self.ds.p) / (0.622 + 0.378 * self.ds.q)
        logger.info(
            f"7/12: Computing water-vapor pressure done in {time.perf_counter() - t0:.2f}s"
        )

    # ------------------- 7 ellipsoidal 重采样 -------------------- #
    def resample_to_ellipsoidal(self) -> None:
        t0 = time.perf_counter()
        if self.vertical_dimension == "pressure_level":
            logger.info("8/12: Skipping resampling (pressure levels)")
            return

        ds, loc = self.ds, self.location
        # 目标高度网格：从最低气压层高度到最低站点高度，每 50 m 一格
        h_max = ds.sel(pressure_level=(ds.pressure_level[ds.pressure_level>0]).min()).h.min()
        target_h = np.arange(loc.alt.min(), float(h_max), 50) * units.meter

        # 新建输出 Dataset，保留 number、site_index、time、h 四个坐标
        ds_new = xr.Dataset(
            coords={
                "number": ds.number,
                "site_index": ds.site_index,
                "time": ds.time,
                "h": target_h,
            }
        )

        # 单剖面插值函数：返回 (n_idx, site_idx, time0, p_name, 插值结果)
        def _interp(n_idx, site_idx, time0, p_name, interpolator):
            s_zen = ds.sel(number=n_idx, site_index=site_idx, time=time0)
            x = np.squeeze(np.flip(s_zen.h.values))
            y = np.squeeze(np.flip(s_zen[p_name].values))
            return n_idx, site_idx, time0, p_name, interpolator(x, y).interpolate(target_h)

        # 并行计算所有层次剖面的重采样
        results = ParallelPbar("Resample to Ellipsoidal")(n_jobs=self.n_jobs, batch_size=self.batch_size)(
            delayed(_interp)(n, s, t, p, interp)
            for n in ds.number.values
            for s in ds.site_index.values
            for t in ds.time.values
            for p, interp in [
                ("e", LogLinearInterpolator),
                ("p", LogLinearInterpolator),
                ("t", LinearInterpolator),
            ]
        )

        # 按参数聚合结果
        res_dict: Dict[str, List[Tuple]] = {}
        for n_idx, s_idx, t_idx, p_name, arr in results:
            res_dict.setdefault(p_name, []).append((n_idx, s_idx, t_idx, arr))

        # 将插值结果写入 buf 并转为 DataArray
        for p_name in ["e", "p", "t"]:
            buf = np.full(
                (len(ds.number), len(ds.site_index), len(ds.time), len(target_h)),
                np.nan,
            )
            n_map = {n: i for i, n in enumerate(ds.number.values)}
            s_map = {s: i for i, s in enumerate(ds.site_index.values)}
            t_map = {t: i for i, t in enumerate(ds.time.values)}

            for n_idx, s_idx, t_idx, interp_vals in res_dict[p_name]:
                buf[n_map[n_idx], s_map[s_idx], t_map[t_idx], :] = interp_vals

            da = xr.DataArray(
                buf,
                dims=("number", "site_index", "time", "h"),
                coords={
                    "number": ds.number,
                    "site_index": ds.site_index,
                    "time": ds.time,
                    "h": target_h,
                },
            ) * ds[p_name].metpy.units
            ds_new[p_name] = da

        # 复制站点信息
        ds_new["lon"], ds_new["lat"], ds_new["site"] = ds.lon, ds.lat, ds.site
        # 按高度降序排列，保证后续逻辑一致
        self.ds = ds_new.sortby("h", ascending=False)
        logger.info(
            f"8/12: Resampling to ellipsoidal done in {time.perf_counter() - t0:.2f}s"
        )


    # ------------- 8 折射率计算 ---------------- #
    def compute_refractive_index(self) -> None:
        t0 = time.perf_counter()
        t, e, p = self.ds.t, self.ds.e, self.ds.p
        p_d = p - e
        k1 = 77.689 * units.kelvin / units.hPa
        k2 = 71.2952 * units.kelvin / units.hPa
        k3 = 375463 * units.kelvin**2 / units.hPa
        r_d = 287.06 * units.joule / (units.kilogram * units.kelvin)
        r_v = 461.525 * units.joule / (units.kilogram * units.kelvin)
        k2_ = k2 - k1 * r_d / r_v

        z_d_inv = 1 + p_d * (
            57.90e-8 / units.hPa
            - 9.4581e-4 * units.kelvin / units.hPa * t**-1
            + 0.25844 * units.kelvin**2 / units.hPa * t**-2
        )
        z_v_inv = 1 + e * (1 + 3.7e-4 / units.hPa * e) * (
            -2.37321e-3 / units.hPa
            + 2.23366 * units.kelvin / units.hPa * t**-1
            - 710.792 * units.kelvin**2 / units.hPa * t**-2
            + 7.75141e4 * units.kelvin**3 / units.hPa * t**-3
        )

        n_d = z_d_inv * k1 * p_d / t
        n_v = z_v_inv * (k2 * e / t + k3 * e / t**2)
        rho_d = p_d / (r_d * t)
        rho_v = e / (r_v * t)
        rho_m = rho_d + rho_v
        n_h = k1 * r_d * rho_m
        n_w = (k2_ * e / t + k3 * e / t**2) * z_v_inv

        if self.refractive_index == "mode2":
            self.ds["n"] = n_w + n_h
            self.ds["n_w"] = n_w
            self.ds["n_h"] = n_h
        else:
            self.ds["n"] = n_d + n_v
            self.ds["n_w"] = n_w
            self.ds["n_h"] = n_d + n_v - n_w
        self.ds["n_d"], self.ds["n_v"] = n_d, n_v
        self.ds["z_d_inv"], self.ds["z_v_inv"] = z_d_inv, z_v_inv
        logger.info(
            f"9/12: Computing refractive index done in {time.perf_counter() - t0:.2f}s"
        )

    # ---------------- 9 顶层延迟 ---------------- #
    def compute_top_level_delay(self) -> None:
        t0 = time.perf_counter()
        ds = self.ds
        top = (
            ds.sel(pressure_level=ds.pressure_level[ds.pressure_level>0].min())
            if self.vertical_dimension == "pressure_level"
            else ds.sel(h=ds.h.max())
        )
        top = top.transpose("number", "time", "site_index")
        top["zwd"] = units.meter * zwd_saastamoinen(
            e=top.e.metpy.dequantify(), t=top.t.metpy.dequantify()
        )
        top["zhd"] = units.meter * zhd_saastamoinen(
            p=top.p.metpy.dequantify(),
            lat=top.lat.metpy.dequantify(),
            alt=top.h.metpy.dequantify(),
        )
        top["ztd"] = top.zwd + top.zhd
        self.top_level = top
        logger.info(
            f"10/12: Computing top-level delays done in {time.perf_counter() - t0:.2f}s"
        )

    # ---------------- 10 Simpson 积分 ---------------- #
    def simpson_numerical_integration(self) -> None:
        t0 = time.perf_counter()
        ds = (
            self.ds.sortby("pressure_level")
            if self.vertical_dimension == "pressure_level"
            else self.ds
        )
        x = -ds.transpose("number", "time", "site_index", self.vertical_dimension).h


        import numpy as np

        from scipy.integrate import cumulative_simpson

        def cumulative_simpson_unsorted(y, x, axis=-1,initial=0.0):
            """
            在无序、非等距采样点 x 上，计算 y(x) 沿指定轴从 x_max 向 x 的累积 Simpson 积分。
            在 x_max 处积分值为 0，之后均为正值，结果按原始点顺序返回。

            参数
            ----
            y : array_like 或 xr.DataArray
                待积分的多维数据。
            x : array_like 或 xr.DataArray
                对应坐标，shape 可与 y 完全一致或可广播到 y。
            axis : int, optional
                沿此轴做积分（默认最后一维）。

            返回
            -----
            res : ndarray 或 xr.DataArray
                与 y 相同形状、同类型的积分结果。
            """
            # 支持 xarray
            is_xr = isinstance(y, xr.DataArray)
            if is_xr:
                orig_dims = y.dims
                orig_coords = y.coords
                y_arr = y.values
                x_arr = x.values
            else:
                y_arr = np.asarray(y)
                x_arr = np.asarray(x)

            # 1) 升序排序索引
            order = np.argsort(x_arr, axis=axis)  # 排序得到沿 axis 升序的位置

            # 2) 排序 x 和 y
            x_sorted = np.take_along_axis(x_arr, order, axis=axis)
            y_sorted = np.take_along_axis(y_arr, order, axis=axis)

            # 3) 计算从 x_min 到各点的累积 Simpson（初始值为 0）
            asc_int = cumulative_simpson(y_sorted, x=x_sorted,
                                         axis=axis, initial=initial)

            # 4) 取出总积分（在 x_max 处的值），并展开回排序数组的形状以便广播
            total = np.take(asc_int, indices=-1, axis=axis)  # 形状中去掉该轴
            total = np.expand_dims(total, axis=axis)  # 在该轴上恢复维度

            # 5) 转换成“从 x_max 向下积分”：在排序后数组上，value = total - asc_int
            rev_int_sorted = total - asc_int

            # 6) 求逆排序索引并还原到原始顺序
            inv_order = np.argsort(order, axis=axis)
            res_arr = np.take_along_axis(rev_int_sorted, inv_order, axis=axis)

            # 7) 如果原来是 xarray，就包回 DataArray，保留 dims/coords
            if is_xr:
                return xr.DataArray(data=res_arr,
                                    dims=orig_dims,
                                    coords=orig_coords,
                                    name=y.name)
            else:
                return res_arr

        def _integ(y):
            y = y.transpose("number", "time", "site_index", self.vertical_dimension)
            return cumulative_simpson(y=y, x=x, axis=-1, initial=0)

        for zxd, n in [("ztd", "n")]:#, ("zwd", "n_w"), ("zhd", "n_h")]:
            val = _integ(ds[n] * 1.0e-6) * units.meters
            ds[f"{zxd}_simpson"] = xr.DataArray(
                data=val,
                dims=["number", "time", "site_index", self.vertical_dimension],
            )
            ds[f"{zxd}_simpson"] += self.top_level[zxd]
        self.ds = ds
        logger.info(
            f"11/12: Simpson integration done in {time.perf_counter() - t0:.2f}s"
        )

    # -------- 11 垂直插值到站点海拔（自适应并行/向量化） -------- #
    def vertical_interpolate_to_site(self) -> xr.Dataset:
        t0 = time.perf_counter()
        ds = self.ds
        import numpy as np
        from scipy.interpolate import CubicSpline

        def _safe_log_cubic_interp(x: np.ndarray,
                                  y: np.ndarray,
                                  xnew: np.ndarray) -> np.ndarray:
            """
            对任意无序、含 NaN 的 (x, y) 数据做对数三次样条插值，
            并返回 xnew 处的估计值。
            """
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            # 1. 掩掉 y 或 x 中的非有限值
            mask = np.isfinite(x) & np.isfinite(y)
            x_filt = x[mask]
            y_filt = y[mask]
            if x_filt.size < 2:
                raise ValueError(f"有效点不足：仅 {x_filt.size} 个，无法拟合")
            # 2. 按 x 排序
            idx = np.argsort(x_filt)
            x_s = x_filt[idx]
            y_s = y_filt[idx]
            # 3. 去重，保证严格递增
            #    对重复的 x 取 y 的均值
            uniq_x, inv, counts = np.unique(x_s, return_inverse=True, return_counts=True)
            if np.any(counts > 1):
                # 平均重复 x 对应的 y
                y_u = np.zeros_like(uniq_x)
                for i, xx in enumerate(uniq_x):
                    y_u[i] = y_s[inv == i].mean()
            else:
                y_u = y_s
            # 4. 构造对数 CubicSpline 并插值
            cs = CubicSpline(
                uniq_x,
                np.log(np.maximum(y_u, 1e-12)),
                extrapolate=True
            )
            return np.exp(cs(xnew))

        def log_cubic_interp(x: np.ndarray, y: np.ndarray, xnew: np.ndarray) -> np.ndarray:
            if x[0] > x[-1]:
                x, y = x[::-1], y[::-1]
            mask = np.isfinite(y)
            x_filt, y_filt = x[mask], y[mask]  # ← 修正：y_filt = y[mask]
            if x_filt.size < 2:
                raise ValueError(f"有效点不足：仅 {x_filt.size} 个，无法拟合")
            cs = CubicSpline(
                x_filt,
                np.log(np.maximum(y_filt, 1e-12)),
                extrapolate=True
            )
            return np.exp(cs(xnew))
        # —— 修复：当 vertical_dimension=='h' 时，广播 h 到 ztd_simpson 的形状 ——
        if self.vertical_dimension == "h":
            h_da, ztd_da = xr.broadcast(
                ds.h.metpy.dequantify(),
                ds.ztd_simpson.metpy.dequantify(),
            )
        else:
            h_da = ds.h.metpy.dequantify()
            ztd_da = ds.ztd_simpson.metpy.dequantify()

        h_all = (
            h_da
            .transpose("number", "time", "site_index", self.vertical_dimension)
            .values
        )
        ztd_all = (
            ztd_da
            .transpose("number", "time", "site_index", self.vertical_dimension)
            .values
        )
        alt = ds.alt.values
        num_dim, time_dim, site_dim, _ = ztd_all.shape
        total_tasks = num_dim * time_dim * site_dim

        # ---------- 小任务：直接向量化 apply_ufunc ---------- #

        if total_tasks <= self.batch_size:
            logger.info("12/12: Vertical interpolation (vectorized)…")
            alt_da = (
                xr.DataArray(
                    alt, dims="site_index", coords={"site_index": ds.site_index}
                )
                .expand_dims(number=ds.number, time=ds.time)
                .transpose("number", "site_index", "time")
            )


            result = xr.apply_ufunc(
                _safe_log_cubic_interp,
                ds.h.metpy.dequantify(),
                ds.ztd_simpson.metpy.dequantify(),
                alt_da,
                input_core_dims=[[self.vertical_dimension],
                                 [self.vertical_dimension], []],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            da = (
                    result
                    .expand_dims(h=[0])
                    .transpose("number", "site_index", "time", "h")
                    * ds.ztd_simpson.metpy.units
                    * 1000
            )
            ds_site = xr.Dataset({"ztd_simpson": da})

        # ---------- 大任务：joblib 并行 CubicSpline ---------- #
        else:
            logger.info("12/12: Vertical interpolation (CubicSpline, joblib parallel)…")
            out = np.empty((num_dim, time_dim, site_dim), dtype=float)

            def _interp_one(n, t, s):
                return _safe_log_cubic_interp(
                    h_all[n, t, s],
                    ztd_all[n, t, s],
                    alt[s]
                )

            flat = Parallel(
                n_jobs=self.n_jobs,
                batch_size=self.batch_size,
            )(
                delayed(_interp_one)(n, t, s)
                for n in range(num_dim)
                for t in range(time_dim)
                for s in range(site_dim)
            )
            out[:] = np.array(flat).reshape(num_dim, time_dim, site_dim)
            da = xr.DataArray(
                out * ds.ztd_simpson.metpy.units * 1000,
                dims=("number", "time", "site_index"),
                coords={
                    "number": ds.number,
                    "time": ds.time,
                    "site_index": ds.site_index,
                },
            ).transpose("number", "site_index", "time")
            ds_site = xr.Dataset({"ztd_simpson": da})

        ds_site["site"] = ds.site
        logger.info(
            f"12/12: Vertical interpolation done in {time.perf_counter() - t0:.2f}s"
        )
        return ds_site

    # ------------------------------ run --------------------------- #
    def run(self) -> xr.DataFrame:
        logger.info(
            f"Start ZTD computation (vertical_dimension={self.vertical_dimension})"
        )
        self.read_met_file()
        self.format_dataset()

        self.horizental_interpolate()
        self.quantify_met_parameters()
        self.geopotential_to_orthometric()
        self.orthometric_to_ellipsoidal()
        self.compute_e()
        if self.vertical_dimension != "pressure_level":
            self.resample_to_ellipsoidal()
        self.compute_refractive_index()
        self.compute_top_level_delay()
        self.simpson_numerical_integration()
        ds_site = self.vertical_interpolate_to_site()
        logger.info("ZTD computation finished")
        df=ds_site.to_dataframe().reset_index()[["time","site","number","ztd_simpson"]]

        if len(df.number.drop_duplicates())==1:
            df=df.drop(columns=["number"])
        return df
