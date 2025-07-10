import time
from nwm.ztd_nwm_old import ZTDNWMGenerator

import pandas as pd
import numpy as np
from scipy.stats import zscore

# ---------------- 生成器 ---------------
location = pd.read_csv(r"data/location_gnss.csv")

location = pd.concat([location] )
zg = ZTDNWMGenerator(
    r"data/failure/elda_pl_9_025_2023010100", location=location, egm_type="egm96-5", n_jobs=-1
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
bias = (df_nwm["ztd_simpson"] - df_nwm["ztd_gnss"]).mean()

print(f"RMSE: {rmse:.3f}  |  Bias: {bias:.3f}")
assert rmse <= 8.22
assert np.abs(bias) < 2.13
