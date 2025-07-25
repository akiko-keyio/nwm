import time
from nwm.ztd_nwm import ZTDNWMGenerator

import pandas as pd
import numpy as np
from scipy.stats import zscore

# ---------------- 生成器 ---------------
location = pd.read_csv(r"data/global_ngl_location.csv")
nwm_path = r"data/failure/ERA5_20230101_00_1h.nc"
nwm_path=r"data/era5_pl_025_2023010100.nc"
nwm_path=r"Z:\NWM\ERA5\global_pl\era5_pl_native_2023010100.nc"
location = pd.concat([location])
zg = ZTDNWMGenerator(
    nwm_path,
    vertical_level="h",
    location=location,
    egm_type="egm96-5",
    n_jobs=-1,
    # batch_size=1000000,
    # load_method='lazy'
    # horizental_interpolation_method="sphere_spline",
)

# ----------- ⏱️ 计时开始 -----------
t0 = time.perf_counter()
df_nwm = zg.run()
print(df_nwm)

dt = time.perf_counter() - t0
print(
    f"zg.run() 用时: {dt:.2f} 秒 共 {len(location)} pt {len(location) / dt:.1f} iter/s"
)
# ----------- ⏱️ 计时结束 -----------

# ---------------- 后续分析 ----------------
df_gnss = pd.read_csv(r"data/ztd_gnss.csv")
df_gnss["time"] = pd.to_datetime(df_gnss["time"])
df_nwm = df_nwm.merge(df_gnss)

df_nwm["res"] = df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]
df_nwm["z_score"] = zscore(df_nwm["res"], ddof=1)
df_nwm = df_nwm[np.abs(df_nwm["z_score"]) < 5]

rmse = np.sqrt(np.mean((df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]) ** 2))
std= (df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]).std()
bias = (df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]).mean()

print(f"RMSE: {rmse:.3f}  |  Bias: {bias:.3f} | std: {std:.3f}")
assert rmse <= 8.22
assert np.abs(bias) < 2.13
assert std<7.94