import time
from nwm.ztd_nwm_fast5 import ZTDNWMGenerator

import pandas as pd
import numpy as np
from scipy.stats import zscore

# ---------------- 生成器 ---------------
location=pd.read_csv(r"data/location_gnss.csv")
location=pd.concat([location]*1)
zg = ZTDNWMGenerator(
    r"Z:\NWM\EDA\us_grid10\elda_pl_9_01_2023010100",
    location=location,
    egm_type="egm96-5",
    n_jobs=-1,
    load_in_memory=True,
    stream_copy=True

    # Local fast4 True 13.31/13.08/14.41 False 12.83/12.99/12.76
    # Local fast5 True 11.64/10.70/11.42 False 13.95/12.64/12.80
    # Local fast4 True 20.92/19.05/18.80 False
    # Remote fast5 34.25 False 36.77
)

# ----------- ⏱️ 计时开始 -----------
t0 = time.perf_counter()
df_eda = zg.run()


dt = time.perf_counter() - t0
print(f"zg.run() 用时: {dt:.2f} 秒 共 {len(location)}*50 pt {len(location)*50/dt:.1f} iter/s")
# ----------- ⏱️ 计时结束 -----------

# ---------------- 后续分析 ----------------
df_gnss = pd.read_csv(r"data/ztd_gnss.csv")
df_gnss['time'] = pd.to_datetime(df_gnss['time'])


def resolve_eda_ztd(df_eda, df_gnss, location_gnss):
    df_eda = (df_eda.groupby(['time', 'site'])['ztd_simpson']
              .agg(['mean', 'std'])
              .reset_index()
              .rename(columns={'mean': 'ztd_nwm', 'std': 'ztd_nwm_sigma'})
              .merge(df_gnss, on=['time', 'site'])
              .merge(location_gnss[['site', 'lon', 'lat']]))

    df_eda['res'] = df_eda['ztd_nwm'] - df_eda['ztd_gnss']
    from scipy.stats import zscore

    df_eda['z_score'] = zscore(df_eda['res'], ddof=1)
    df_eda = df_eda[np.abs(df_eda['z_score']) < 5]
    rmse = np.sqrt(np.mean(df_eda['res'] ** 2))
    bias = np.mean(df_eda['res'])
    return df_eda, rmse, bias

df_nwm, rmse, bias=resolve_eda_ztd(df_eda, df_gnss, location)

print(f"RMSE: {rmse:.3f}  |  Bias: {bias:.3f}")
assert rmse <= 8.62
assert np.abs(bias) < 2.7