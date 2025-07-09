"""
Fast ZTD generator based on NWM / ERA-like 3-D meteorological files.
保留高精度 CubicSpline 与双精度计算，同时对水平与垂直插值做并行 / 向量化加速。
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from metpy.units import units
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from tqdm_joblib import ParallelPbar

from nwm.Interprepter import LogLinearInterpolator, LinearInterpolator
from nwm.geoid import GeoidHeight
from nwm.height_convert import geopotential_to_geometric
from nwm.ztd_met import zhd_saastamoinen, zwd_saastamoinen

# -------------------------- 可选 xESMF -------------------------- #
try:
    import xesmf as xe

    _HAS_XESMF = True
except Exception:
    _HAS_XESMF = False

# ----------------------------- 类实现 ---------------------------- #


class ZTDNWMGenerator:
    def __init__(
        self,
        nwm_path: str | Path,
        location: xr.Dataset | None = None,
        egm_type: str = "egm96-5",
        vertical_level: str = "pressure_level",
        gravity_variation: str = "latitude",
        refractive_index: str = "mode2",
        compute_e_mode: str = "mode2",
        p_interp_step: int | None = None,
        swap_interp_step: int | None = None,
        n_jobs: int = -3,
        batch_size: int = 50000,
        load_in_memory: bool = True,
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
        self.load_in_memory = load_in_memory

        self.ds: xr.Dataset | None = None
        self.ds_site: xr.Dataset | None = None
        self.top_level: xr.Dataset | None = None

        # 用于重用 xESMF 权重
        self._regridder = None

    # -------------------------- 1 读文件 -------------------------- #
    def read_met_file(self) -> None:
        """把 NWM/ERA-like NetCDF 读成 xarray.Dataset，完全保留 double 精度。"""
        if self.load_in_memory:
            mem_url = "memory://temp.nc"
            import fsspec

            with (
                self.nwm_path.open("rb") as fsrc,
                fsspec.open(mem_url, "wb") as fdst,
            ):
                shutil.copyfileobj(fsrc, fdst, length=16 << 20)

            with fsspec.open(mem_url, "rb") as f:
                ds = xr.open_dataset(f)
                ds.load()
            self.ds = ds
        else:
            self.ds = xr.open_dataset(self.nwm_path)

        rename_map = {
            "level": "pressure_level",
            "isobaricInhPa": "pressure_level",
            "valid_time": "time",
        }
        existing = {k: v for k, v in rename_map.items() if k in self.ds}
        if existing:
            logger.info(f"Renaming dimensions: {existing}")
            self.ds = self.ds.rename(existing)

        if "time" not in self.ds.dims:
            self.ds = self.ds.expand_dims("time")
        if "number" not in self.ds.dims:
            self.ds = self.ds.expand_dims("number")

    # ---------------------- 2 水平插值 --------------------------- #
    def horizental_interpolate(self) -> None:
        t0 = time.perf_counter()
        if self.location is None:
            self.ds = self.ds.stack(site_index=("latitude", "longitude"))
            self.ds["site"] = self.ds.site_index
            self.ds["alt"] = 0
        else:
            ds = self.ds
            loc = self.location
            pts = np.column_stack((loc["lat"].values, loc["lon"].values))
            site_idx = loc.index

            new_vars: Dict[str, xr.DataArray] = {}
            lat_grid = ds.latitude.values
            lon_grid = ds.longitude.values

            for vn, da in ds.data_vars.items():
                dims = da.dims
                if "latitude" in dims and "longitude" in dims:
                    arr = da.values
                    lat_ax = dims.index("latitude")
                    lon_ax = dims.index("longitude")
                    other_axes = [i for i in range(arr.ndim) if i not in (lat_ax, lon_ax)]
                    arr2 = np.moveaxis(
                        arr,
                        (lat_ax, lon_ax) + tuple(other_axes),
                        (0,      1,      *(2 + np.arange(len(other_axes))))
                    )
                    interp = RegularGridInterpolator(
                        (lat_grid, lon_grid),
                        arr2,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    res = interp(pts)
                    if res.ndim > 1:
                        batch_axes = tuple(range(1, res.ndim))
                        res2 = np.moveaxis(res, batch_axes, tuple(range(res.ndim - 1)))
                    else:
                        res2 = res[:, None]
                    new_dims = tuple(d for d in dims if d not in ("latitude", "longitude")) + ("site_index",)
                    coords = {d: ds.coords[d] for d in new_dims if d != "site_index"}
                    coords["site_index"] = site_idx
                    new_vars[vn] = xr.DataArray(res2, dims=new_dims, coords=coords, name=vn)
                else:
                    new_vars[vn] = da

            ds2 = xr.Dataset(new_vars)
            ds2["lat"] = ("site_index", loc["lat"].values)
            ds2["lon"] = ("site_index", loc["lon"].values)
            ds2["alt"] = ("site_index", loc["alt"].values)
            ds2["site"] = ("site_index", loc["site"].values)
            ds2.coords["alt"] = ds2.alt
            self.ds = ds2

        elapsed = time.perf_counter() - t0
        logger.info(f"2/11: Performing horizontal interpolation done in {elapsed:.2f}s")

    # --------------------------- 3 量纲 --------------------------- #
    def quantify_met_parameters(self) -> None:
        t0 = time.perf_counter()
        self.ds["p"] = self.ds.pressure_level * units.hPa
        self.ds["z"]=self.ds["z"]*units.meters**2*units.seconds**-2
        self.ds["t"]=self.ds["t"]*units.kelvin

        elapsed = time.perf_counter() - t0
        logger.info(f"3/11: Quantifying meteorological parameters done in {elapsed:.2f}s")

    # --------- 4,5 高程转换（与原版本保持一致，不修改精度） --------- #
    def geopotential_to_orthometric(self) -> None:
        if self.gravity_variation == "ignore":
            Re = 6371.2229e3 * units.meters
            G = 9.80665 * units.meters / units.second**2
            geopotential_height = self.ds["z"] / G
            self.ds["h"] = (geopotential_height * Re) / (Re - geopotential_height)
        elif self.gravity_variation == "latitude":
            G = 9.80665 * units.meters / units.second**2
            geopotential_height = self.ds.z / G
            lat_vals = self.ds.lat
            if len(lat_vals.shape) == 1:
                lat_vals = np.expand_dims(lat_vals, axis=(0, 1))
            h = geopotential_to_geometric(latitude=lat_vals, geopotential_height=geopotential_height)
            self.ds["h"] = h
        else:
            raise ValueError("Gravity variation must be either ignore or latitude")

    def orthometric_to_ellipsoidal(self) -> None:
        geoid = GeoidHeight(egm_type=self.egm_type)
        site_indices = self.ds.site_index.values
        lat_vals = self.ds["lat"].values
        lon_vals = self.ds["lon"].values

        anomaly = np.array([geoid.get(float(la), float(lo)) for la, lo in zip(lat_vals, lon_vals)]) * units.meters
        anom_da = xr.DataArray(anomaly, dims=("site_index",), coords={"site_index": site_indices})
        self.ds["h"] = self.ds["h"] + anom_da

    # ------------------------- 6 计算 e --------------------------- #
    def compute_e(self) -> None:
        if self.compute_e_mode == "mode1":
            self.ds["e"] = (self.ds["q"] * self.ds["p"]) / 0.622
        elif self.compute_e_mode == "mode2":
            self.ds["e"] = (self.ds["q"] * self.ds["p"]) / (0.622 + 0.378 * self.ds["q"])

    # ------------------- 7 ellipsoidal 重采样 -------------------- #
    def resample_to_ellipsoidal(self) -> None:
        if self.vertical_dimension == "pressure_level":
            return

        ds = self.ds
        loc = self.location

        h_max = ds.sel(pressure_level=ds.pressure_level.min()).h.min()
        target_h = np.arange(loc.alt.min(), float(h_max), 50) * units.meters

        def interpolate_for_site_time(site_index, time, param_name, interpolator):
            site_zenith = ds.sel(site_index=site_index, time=time)
            x = np.flip(site_zenith.h.values)
            y = np.flip(site_zenith[param_name].values)
            return site_index, time, param_name, interpolator(x, y).interpolate(target_h)

        results = ParallelPbar("Resample to Ellipsoidal")(n_jobs=self.n_jobs)(
            delayed(interpolate_for_site_time)(s, t, p, interp)
            for s in ds.site_index.values
            for t in ds.time.values
            for p, interp in [
                ("e", LogLinearInterpolator),
                ("p", LogLinearInterpolator),
                ("t", LinearInterpolator),
            ]
        )

        res_dict: Dict[str, List[Tuple]] = {}
        for s_idx, t_idx, p_name, arr in results:
            res_dict.setdefault(p_name, []).append((s_idx, t_idx, arr))

        ds_swap = xr.Dataset(coords={"site_index": ds.site_index, "time": ds.time, "h": target_h})
        for p_name in ["e", "p", "t"]:
            da = xr.DataArray(
                dims=["site_index", "time", "h"],
                coords={"site_index": ds.site_index, "time": ds.time, "h": target_h},
            )
            buf = np.full((len(ds.site_index), len(ds.time), len(target_h)), np.nan)
            s_map = {s: i for i, s in enumerate(ds.site_index.values)}
            t_map = {t: i for i, t in enumerate(ds.time.values)}
            for s_idx, t_idx, interp_vals in res_dict[p_name]:
                buf[s_map[s_idx], t_map[t_idx], :] = interp_vals
            da.loc[dict(site_index=ds.site_index, time=ds.time, h=target_h)] = buf
            ds_swap[p_name] = da * ds[p_name].metpy.units

        ds_swap["lon"] = ds["lon"]
        ds_swap["lat"] = ds["lat"]
        ds_swap["site"] = ds["site"]
        ds_swap = ds_swap.sortby("h", ascending=False)
        self.ds = ds_swap

    # ------------- 8 折射率（改为 ASCII '-' 常数） ---------------- #
    def compute_refractive_index(self) -> None:
        t = self.ds.t  # temperature
        e = self.ds.e  # partial pressure of water vapor
        p = self.ds.p  # pressure of (moist) air
        p_d = p - e

        k1 = 77.689 * units.kelvin / units.hPa
        k2 = 71.2952 * units.kelvin / units.hPa
        k3 = 375463 * units.kelvin ** 2 / units.hPa
        r_d = 287.06 * units.joule / (units.kilogram * units.kelvin)
        r_v = 461.525 * units.joule / (units.kilogram * units.kelvin)
        k2_ = k2 - k1 * r_d / r_v

        z_d_inv = 1 + p_d * (
            57.90e-8 / units.hPa
            - (9.4581e-4 * units.kelvin / units.hPa) * t ** -1
            + 0.25844 * units.kelvin ** 2 / units.hPa * t ** -2
        )

        z_v_inv = 1 + e * (1 + 3.7e-4 / units.hPa * e) * (
            -2.37321e-3 / units.hPa
            + 2.23366 * units.kelvin / units.hPa * (t ** -1)
            - 710.792 * units.kelvin ** 2 / units.hPa * (t ** -2)
            + 7.75141e4 * units.kelvin ** 3 / units.hPa * (t ** -3)
        )

        n_d = z_d_inv * k1 * p_d / t
        n_v = z_v_inv * (k2 * e / t + k3 * e / (t ** 2))

        rho_d = p_d / (r_d * t)
        rho_v = e / (r_v * t)
        rho_m = rho_d + rho_v
        n_h = k1 * r_d * rho_m
        n_w = (k2_ * e / t + k3 * e / (t ** 2)) * z_v_inv

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

    # ---------------- 9 顶层延迟 ------------------- #
    def compute_top_level_delay(self) -> None:
        ds = self.ds
        if self.vertical_dimension == "pressure_level":
            top_level = ds.sel(pressure_level=ds.pressure_level.min())
        elif self.vertical_dimension == "h":
            top_level = ds.sel(h=ds.h.max())
        else:
            raise ValueError("vertical_dimension must be pressure_level or h")

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

    # ---------------- 10 Simpson 积分 ---------------- #
    def simpson_numerical_integration(self) -> None:
        ds = self.ds
        if self.vertical_dimension == "pressure_level":
            ds = ds.sortby("pressure_level")

        x = -ds.transpose("number", "time", "site_index", self.vertical_dimension).h

        def _integrate(y):
            y = y.transpose("number", "time", "site_index", self.vertical_dimension)
            return cumulative_simpson(y=y, x=x, axis=-1, initial=0)

        for zxd, n in [("ztd", "n"), ("zwd", "n_w"), ("zhd", "n_h")]:
            zxd_val = _integrate(ds[n] * 1.0e-6) * units.meters
            ds[f"{zxd}_simpson"] = xr.DataArray(
                data=zxd_val,
                dims=["number", "time", "site_index", self.vertical_dimension],
            )
            ds[f"{zxd}_simpson"] += self.top_level[zxd]

        self.ds = ds

    # ------------- 11 垂直插值到站点海拔 ---------------- #
    def vertical_interpolate_to_site(self) -> xr.Dataset:
        ds = self.ds

        # 1. 先把 h 转置成和 ztd_simpson 一样的维度顺序
        h_all = (
            ds.h.metpy.dequantify()
            .transpose("number", "time", "site_index", self.vertical_dimension)
            .values
        )  # 现在是 (num, time, site, lev)

        ztd_all = (
            ds["ztd_simpson"].metpy.dequantify()
            .transpose("number", "time", "site_index", self.vertical_dimension)
            .values
        )  # 同上

        alt_all = ds.alt.values  # (site,)

        num_dim, time_dim, site_dim, lev_dim = ztd_all.shape
        out = np.empty((num_dim, time_dim, site_dim), dtype=float)

        def _interp_one(n_idx: int, t_idx: int, s_idx: int) -> float:
            x = h_all[n_idx, t_idx, s_idx, :]       # (lev,)
            y = ztd_all[n_idx, t_idx, s_idx, :]     # (lev,)
            a = alt_all[s_idx]

            # 保证 x 升序
            if x[0] > x[-1]:
                x = x[::-1]
                y = y[::-1]

            cs = CubicSpline(x, np.log(np.maximum(y, 1e-12)))
            return float(np.exp(cs(a)))

        logger.info("Vertical interpolation (CubicSpline, joblib parallel)…")
        idxs = (
            (n, t, s)
            for n in range(num_dim)
            for t in range(time_dim)
            for s in range(site_dim)
        )

        flat = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size)(
            delayed(_interp_one)(n, t, s) for n, t, s in idxs
        )

        # 填回 out
        k = 0
        for n in range(num_dim):
            for t in range(time_dim):
                for s in range(site_dim):
                    out[n, t, s] = flat[k]
                    k += 1

        # 构造结果 Dataset（不变）
        da = xr.DataArray(
            out * ds["ztd_simpson"].metpy.units,
            dims=("number", "time", "site_index"),
            coords={
                "number":     ds.number,
                "time":       ds.time,
                "site_index": ds.site_index,
            },
            name="ztd_simpson",
        ).transpose("number", "site_index", "time")

        ds_site = xr.Dataset({"ztd_simpson": da*1000})#m->mm
        ds_site["site"] = ds.site
        self.ds_site = ds_site
        return ds_site





    # ------------------------------ run --------------------------- #
    def run(self, time_select: slice | None = None) -> xr.DataFrame:
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

        logger.info("10/11: Performing Simpson integration")
        self.simpson_numerical_integration()

        logger.info("11/11: Vertical interpolation to station altitude")
        ds_site = self.vertical_interpolate_to_site()

        logger.info("ZTD computation finished; returning DataFrame")
        return ds_site.to_dataframe().reset_index().drop(columns=["site_index"])