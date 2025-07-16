import time
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
from nwm.ztd_nwm_back import ZTDNWMGenerator

# ---------------- 生成器 ---------------
location = pd.read_csv(r"data/global_ngl_location.csv")

nwm_path = r"data/failure/elda_pl_9_025_2023010212"
# nwm_path=r"data/failure/ERA5_20230101_00_1h.nc"
nwm_path=r"Z:\NWM\EDA\global_grid25\elda_pl_9_025_2023011800"
nwm_path=r"\\NAS\Users\Server\Desktop\elda_pl_9_025_2023011500.nc"
nwm_path=r"data/failure/elda_pl_9_025_2023011500.nc"
nwm_path=r"data/failure/elda_pl_9_025_2023010100"
nwm_path=r"Z:\NWM\EDA\global_grid25\elda_pl_9_025_2023010212"
# nwm_path=r"Y:\nwm\src\nwm\test.zarr"
print(Path(nwm_path).exists())

# nwm_path=r"Z:\NWM\EDA\global_grid25\elda_pl_9_025_2023011512"
location = pd.concat([location])
zg = ZTDNWMGenerator(
    nwm_path,
    location=location,
    egm_type="egm96-5",
    n_jobs=-1,
    # load_method="zarr",
)

# ----------- ⏱️ 计时开始 -----------
t0 = time.perf_counter()
df_eda = zg.run()

dt = time.perf_counter() - t0
print(
    f"zg.run() 用时: {dt:.2f} 秒 共 {len(location)}*50 pt {len(location) * 50 / dt:.1f} iter/s"
)
# ----------- ⏱️ 计时结束 -----------

# ---------------- 后续分析 ----------------
df_gnss = pd.read_parquet(r"data/ztd_ngl_2023.parquet")
df_gnss["time"] = pd.to_datetime(df_gnss["time"])


def resolve_eda_ztd(df_eda, df_gnss, location_gnss):
    df_eda = (
        df_eda.groupby(["time", "site"])["ztd_simpson"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ztd_nwm", "std": "ztd_nwm_sigma"})
        .merge(df_gnss, on=["time", "site"])
        .merge(location_gnss[["site", "lon", "lat"]])
    )

    df_eda["res"] = df_eda["ztd_nwm"] - df_eda["ztd_gnss"]
    from scipy.stats import zscore

    df_eda["z_score"] = zscore(df_eda["res"], ddof=1)
    df_eda = df_eda[np.abs(df_eda["z_score"]) < 5]
    rmse = np.sqrt(np.mean(df_eda["res"] ** 2))
    bias = np.mean(df_eda["res"])
    return df_eda, rmse, bias


df_nwm, rmse, bias = resolve_eda_ztd(df_eda, df_gnss, location)

# ---------------- 后续分析 ----------------
df_gnss = pd.read_parquet(r"data/ztd_ngl_2023.parquet")
df_gnss["time"] = pd.to_datetime(df_gnss["time"])
df_gnss=df_gnss[df_gnss['time'].isin(df_nwm.time.drop_duplicates())]
df_nwm = df_nwm.merge(df_gnss)

df_nwm['ztd_simpson'] = df_nwm['ztd_nwm']
df_nwm["res"] = df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]
df_nwm = df_nwm[np.abs(df_nwm["res"]) < np.quantile(np.abs(df_nwm["res"]),0.99)]

rmse = np.sqrt(np.mean((df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]) ** 2))
std= (df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]).std()
bias = (df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]).mean()

print(f"RMSE: {rmse:.3f}  |  Bias: {bias:.3f} | std: {std:.3f}")
assert rmse <= 8.31
assert np.abs(bias) < 4.41
assert std<7.05