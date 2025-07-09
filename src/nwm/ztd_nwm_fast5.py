"""
Fast ZTD generator based on NWM / ERA-like 3-D meteorological files.
保留高精度 CubicSpline 与双精度计算，同时对水平与垂直插值做并行 / 向量化加速。
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
from scipy.interpolate import CubicSpline, RegularGridInterpolator
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
        gravity_variation: str = "latitude",
        refractive_index: str = "mode2",
        compute_e_mode: str = "mode2",
        p_interp_step: int | None = None,
        swap_interp_step: int | None = None,
        n_jobs: int = -1,
        batch_size: int = 100_000,
        load_in_memory: bool = True,
            stream_copy=True,
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
        self.stream_copy = True

        self.ds: xr.Dataset | None = None
        self.ds_site: xr.Dataset | None = None
        self.top_level: xr.Dataset | None = None

    # -------------------------- 1 读文件 -------------------------- #
    def read_met_file(self) -> None:
        t0 = time.perf_counter()
        if self.stream_copy:
            import fsspec
            mem_url = "memory://temp.nc"
            with (
                self.nwm_path.open("rb") as fsrc,
                fsspec.open(mem_url, "wb") as fdst,
            ):
                shutil.copyfileobj(fsrc, fdst, length=1024 << 20)
            with fsspec.open(mem_url, "rb") as f:
                self.ds = xr.open_dataset(f)
                self.ds.load()
        else:
            if self.load_in_memory:
                def _is_hdf5(path: Path, blocksize: int = 8) -> bool:
                    """快速判断文件是否以 HDF5 标识开头。"""
                    with path.open("rb") as f:
                        header = f.read(blocksize)
                    return header == b"\x89HDF\r\n\x1a\n"
                # 1️⃣ 本地文件：直接用最快 backend；远程路径可另行处理
                engine = "h5netcdf" if _is_hdf5(self.nwm_path) else None
                try:
                    self.ds = xr.open_dataset(
                        self.nwm_path,
                        engine=engine,  # None → xarray 自动检测
                        chunks="auto",  # 允许 dask 并行 I/O
                        mask_and_scale=True,
                        decode_times=True,
                    ).load()  # 真正读到内存
                except Exception as e:
                    # fallback 双保
                    logger.warning(f"{engine or 'default'} backend failed ({e}); retry with default engine")
                    self.ds = xr.open_dataset(self.nwm_path).load()
            else:
                # 懒加载：留给后续 dask 计算
                self.ds = xr.open_dataset(self.nwm_path)

        # === 后续维度重命名与补维保持不变 ===
        rename_map = {"level": "pressure_level",
                      "isobaricInhPa": "pressure_level",
                      "valid_time": "time"}
        exist = {k: v for k, v in rename_map.items() if k in self.ds}
        if exist:
            logger.info(f"Renaming dimensions: {exist}")
            self.ds = self.ds.rename(exist)

        if "time" not in self.ds.dims:
            self.ds = self.ds.expand_dims("time")
        if "number" not in self.ds.dims:
            self.ds = self.ds.expand_dims("number")

        logger.info(f"1/11: Reading meteorological file done in {time.perf_counter() - t0:.2f}s")

    # ---------------------- 2 水平插值 --------------------------- #
    def horizental_interpolate(self) -> None:
        t0 = time.perf_counter()
        if self.location is None:
            self.ds = self.ds.stack(site_index=("latitude", "longitude"))
            self.ds["site"] = self.ds.site_index
            self.ds["alt"] = 0
        else:
            ds, loc = self.ds, self.location
            pts = np.column_stack((loc["lat"].values, loc["lon"].values))
            lat_grid, lon_grid = ds.latitude.values, ds.longitude.values

            new_vars: Dict[str, xr.DataArray] = {}
            for vn, da in ds.data_vars.items():
                dims = da.dims
                if {"latitude", "longitude"} <= set(dims):
                    lat_ax, lon_ax = dims.index("latitude"), dims.index("longitude")
                    other_axes = [i for i in range(da.ndim) if i not in (lat_ax, lon_ax)]
                    arr2 = np.moveaxis(
                        da.values,
                        (lat_ax, lon_ax) + tuple(other_axes),
                        (0,      1)       + tuple(2 + np.arange(len(other_axes))),
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
                        res2 = np.moveaxis(res, range(1, res.ndim), range(res.ndim - 1))
                    else:
                        res2 = res[:, None]
                    new_dims = tuple(d for d in dims if d not in ("latitude", "longitude")) + ("site_index",)
                    coords = {d: ds.coords[d] for d in new_dims if d != "site_index"}
                    coords["site_index"] = loc.index
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
        logger.info(f"2/11: Horizontal interpolation done in {time.perf_counter()-t0:.2f}s")

    # --------------------------- 3 量纲 --------------------------- #
    def quantify_met_parameters(self) -> None:
        t0 = time.perf_counter()
        self.ds["p"] = self.ds.pressure_level * units.hPa
        self.ds["z"] = self.ds["z"] * units.meters**2 / units.second**2
        self.ds["t"] = self.ds["t"] * units.kelvin
        logger.info(f"3/11: Quantifying meteorological parameters done in {time.perf_counter()-t0:.2f}s")

    # --------------- 4 Geopotential → Orthometric ---------------- #
    def geopotential_to_orthometric(self) -> None:
        t0 = time.perf_counter()
        if self.gravity_variation == "ignore":
            Re = 6371.2229e3 * units.meter
            G0 = 9.80665 * units.meter / units.second**2
            geop_height = self.ds["z"] / G0
            self.ds["h"] = geop_height * Re / (Re - geop_height)
        elif self.gravity_variation == "latitude":
            G0 = 9.80665 * units.meter / units.second**2
            geop_height = self.ds.z / G0
            lat_vals = self.ds.lat
            if lat_vals.ndim == 1:
                lat_vals = np.expand_dims(lat_vals, axis=(0, 1))
            self.ds["h"] = geopotential_to_geometric(latitude=lat_vals, geopotential_height=geop_height)
        else:
            raise ValueError("gravity_variation must be ignore or latitude")
        logger.info(f"4/11: Geopotential→Orthometric done in {time.perf_counter()-t0:.2f}s")

    # ------------- 5 Orthometric → Ellipsoidal ------------------- #
    def orthometric_to_ellipsoidal(self) -> None:
        t0 = time.perf_counter()
        geoid = GeoidHeight(egm_type=self.egm_type)
        anomaly = np.array(
            [geoid.get(float(la), float(lo)) for la, lo in zip(self.ds.lat.values, self.ds.lon.values)]
        ) * units.meter
        anom_da = xr.DataArray(anomaly, dims=("site_index",), coords={"site_index": self.ds.site_index})
        self.ds["h"] = self.ds["h"] + anom_da
        logger.info(f"5/11: Orthometric→Ellipsoidal done in {time.perf_counter()-t0:.2f}s")

    # ------------------------- 6 计算 e --------------------------- #
    def compute_e(self) -> None:
        t0 = time.perf_counter()
        if self.compute_e_mode == "mode1":
            self.ds["e"] = (self.ds.q * self.ds.p) / 0.622
        else:
            self.ds["e"] = (self.ds.q * self.ds.p) / (0.622 + 0.378 * self.ds.q)
        logger.info(f"6/11: Computing water-vapor pressure done in {time.perf_counter()-t0:.2f}s")

    # ------------------- 7 ellipsoidal 重采样 -------------------- #
    def resample_to_ellipsoidal(self) -> None:
        t0 = time.perf_counter()
        if self.vertical_dimension == "pressure_level":
            logger.info("7/11: Skipping resampling (pressure levels)")
            return

        ds, loc = self.ds, self.location
        h_max = ds.sel(pressure_level=ds.pressure_level.min()).h.min()
        target_h = np.arange(loc.alt.min(), float(h_max), 50) * units.meter

        def _interp(site_idx, time0, p_name, interpolator):
            s_zen = ds.sel(site_index=site_idx, time=time0)
            x = np.flip(s_zen.h.values); y = np.flip(s_zen[p_name].values)
            return site_idx, time0, p_name, interpolator(x, y).interpolate(target_h)

        results = ParallelPbar("Resample to Ellipsoidal")(n_jobs=self.n_jobs)(
            delayed(_interp)(s, t, p, interp)
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

        ds_new = xr.Dataset(coords={"site_index": ds.site_index, "time": ds.time, "h": target_h})
        for p_name in ["e", "p", "t"]:
            buf = np.full((len(ds.site_index), len(ds.time), len(target_h)), np.nan)
            s_map = {s: i for i, s in enumerate(ds.site_index.values)}
            t_map = {t: i for i, t in enumerate(ds.time.values)}
            for s_idx, t_idx, interp_vals in res_dict[p_name]:
                buf[s_map[s_idx], t_map[t_idx], :] = interp_vals
            da = xr.DataArray(buf, dims=("site_index", "time", "h"), coords=ds_new.coords) * ds[p_name].metpy.units
            ds_new[p_name] = da
        ds_new["lon"], ds_new["lat"], ds_new["site"] = ds.lon, ds.lat, ds.site
        self.ds = ds_new.sortby("h", ascending=False)
        logger.info(f"7/11: Resampling to ellipsoidal done in {time.perf_counter()-t0:.2f}s")

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

        z_d_inv = 1 + p_d * (57.90e-8 / units.hPa - 9.4581e-4 * units.kelvin / units.hPa * t ** -1
                             + 0.25844 * units.kelvin**2 / units.hPa * t ** -2)
        z_v_inv = 1 + e * (1 + 3.7e-4 / units.hPa * e) * (
            -2.37321e-3 / units.hPa + 2.23366 * units.kelvin / units.hPa * t ** -1
            - 710.792 * units.kelvin**2 / units.hPa * t ** -2
            + 7.75141e4 * units.kelvin**3 / units.hPa * t ** -3
        )

        n_d = z_d_inv * k1 * p_d / t
        n_v = z_v_inv * (k2 * e / t + k3 * e / t ** 2)
        rho_d = p_d / (r_d * t); rho_v = e / (r_v * t); rho_m = rho_d + rho_v
        n_h = k1 * r_d * rho_m
        n_w = (k2_ * e / t + k3 * e / t ** 2) * z_v_inv

        if self.refractive_index == "mode2":
            self.ds["n"] = n_w + n_h; self.ds["n_w"] = n_w; self.ds["n_h"] = n_h
        else:
            self.ds["n"] = n_d + n_v; self.ds["n_w"] = n_w; self.ds["n_h"] = n_d + n_v - n_w
        self.ds["n_d"], self.ds["n_v"] = n_d, n_v
        self.ds["z_d_inv"], self.ds["z_v_inv"] = z_d_inv, z_v_inv
        logger.info(f"8/11: Computing refractive index done in {time.perf_counter()-t0:.2f}s")

    # ---------------- 9 顶层延迟 ---------------- #
    def compute_top_level_delay(self) -> None:
        t0 = time.perf_counter()
        ds = self.ds
        top = ds.sel(pressure_level=ds.pressure_level.min()) if self.vertical_dimension == "pressure_level" else ds.sel(h=ds.h.max())
        top = top.transpose("number", "time", "site_index")
        top["zwd"] = units.meter * zwd_saastamoinen(
            e=top.e.metpy.dequantify(), t=top.t.metpy.dequantify()
        )
        top["zhd"] = units.meter * zhd_saastamoinen(
            p=top.p.metpy.dequantify(), lat=top.lat.metpy.dequantify(), alt=top.h.metpy.dequantify()
        )
        top["ztd"] = top.zwd + top.zhd
        self.top_level = top
        logger.info(f"9/11: Computing top-level delays done in {time.perf_counter()-t0:.2f}s")

    # ---------------- 10 Simpson 积分 ---------------- #
    def simpson_numerical_integration(self) -> None:
        t0 = time.perf_counter()
        ds = self.ds.sortby("pressure_level") if self.vertical_dimension == "pressure_level" else self.ds
        x = -ds.transpose("number", "time", "site_index", self.vertical_dimension).h

        def _integ(y):
            y = y.transpose("number", "time", "site_index", self.vertical_dimension)
            return cumulative_simpson(y=y, x=x, axis=-1, initial=0)

        for zxd, n in [("ztd", "n"), ("zwd", "n_w"), ("zhd", "n_h")]:
            val = _integ(ds[n] * 1.0e-6) * units.meters
            ds[f"{zxd}_simpson"] = xr.DataArray(
                data=val,
                dims=["number", "time", "site_index", self.vertical_dimension],
            )
            ds[f"{zxd}_simpson"] += self.top_level[zxd]
        self.ds = ds
        logger.info(f"10/11: Simpson integration done in {time.perf_counter()-t0:.2f}s")

    # -------- 11 垂直插值到站点海拔（自适应并行/向量化） -------- #
    def vertical_interpolate_to_site(self) -> xr.Dataset:
        t0 = time.perf_counter()
        ds = self.ds
        h_all = ds.h.metpy.dequantify().transpose("number", "time", "site_index", self.vertical_dimension).values
        ztd_all = ds.ztd_simpson.metpy.dequantify().transpose("number", "time", "site_index", self.vertical_dimension).values
        alt = ds.alt.values
        num_dim, time_dim, site_dim, _ = ztd_all.shape
        total_tasks = num_dim * time_dim * site_dim

        # ---------- 小任务：直接向量化 apply_ufunc ---------- #
        if total_tasks <= self.batch_size:
            logger.info("11/11: Vertical interpolation (vectorized)…")
            alt_da = xr.DataArray(
                alt, dims="site_index", coords={"site_index": ds.site_index}
            ).expand_dims(number=ds.number, time=ds.time).transpose("number", "site_index", "time")

            def _log_cubic(x, y, xnew):
                if x[0] > x[-1]:
                    x, y = x[::-1], y[::-1]
                return np.exp(CubicSpline(x, np.log(np.maximum(y, 1e-12)))(xnew))

            result = xr.apply_ufunc(
                _log_cubic,
                ds.h.metpy.dequantify(),
                ds.ztd_simpson.metpy.dequantify(),
                alt_da,
                input_core_dims=[[self.vertical_dimension], [self.vertical_dimension], []],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            da = result.expand_dims(h=[0]).transpose("number", "site_index", "time", "h") \
                   * ds.ztd_simpson.metpy.units * 1000
            ds_site = xr.Dataset({"ztd_simpson": da})

        # ---------- 大任务：joblib 并行 CubicSpline ---------- #
        else:
            logger.info("11/11: Vertical interpolation (CubicSpline, joblib parallel)…")
            out = np.empty((num_dim, time_dim, site_dim), dtype=float)

            def _interp_one(n, t, s):
                x = h_all[n, t, s]; y = ztd_all[n, t, s]
                if x[0] > x[-1]:
                    x, y = x[::-1], y[::-1]
                return np.exp(CubicSpline(x, np.log(np.maximum(y, 1e-12)))(alt[s]))

            flat = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size)(
                delayed(_interp_one)(n, t, s)
                for n in range(num_dim)
                for t in range(time_dim)
                for s in range(site_dim)
            )
            out[:] = np.array(flat).reshape(num_dim, time_dim, site_dim)
            da = xr.DataArray(
                out * ds.ztd_simpson.metpy.units * 1000,
                dims=("number", "time", "site_index"),
                coords={"number": ds.number, "time": ds.time, "site_index": ds.site_index},
            ).transpose("number", "site_index", "time")
            ds_site = xr.Dataset({"ztd_simpson": da})

        ds_site["site"] = ds.site
        self.ds_site = ds_site
        logger.info(f"11/11: Vertical interpolation done in {time.perf_counter()-t0:.2f}s")
        return ds_site

    # ------------------------------ run --------------------------- #
    def run(self) -> xr.DataFrame:
        logger.info(f"Start ZTD computation (vertical_dimension={self.vertical_dimension})")
        self.read_met_file()
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
        return ds_site.to_dataframe().reset_index().drop(columns=["site_index"])
