import time

import pandas as pd
import numpy as np
from scipy.stats import zscore

from nwm import ZTDNWMGenerator

# ---------------- 生成器 ---------------

location = pd.read_csv(r"data/global_ngl_location.csv")

nwm_path=r"data/failure/elda_pl_9_025_2023010100"
nwm_path=r"data/failure/ERA5_20230101_00_1h.nc"
location = pd.concat([location] )
zg = ZTDNWMGenerator(
    nwm_path, location=location, egm_type="egm96-5", n_jobs=-1

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
df_gnss = pd.read_csv(r"data/ztd_gnss.csv")
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

print(f"RMSE: {rmse:.3f}  |  Bias: {bias:.3f}")
assert rmse <= 8.62
assert np.abs(bias) < 2.7
