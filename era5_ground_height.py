from nwm import ZTDNWMGenerator

z=ZTDNWMGenerator(r"data/ground/era5_geo_native_2023010100.nc")
z.read_met_file()
z.format_dataset()

ds=z.ds

import numpy as np

res = 0.25

lon_edges = np.arange(-180 - res/2,  180 + res/2 + res, res)
lat_edges = np.arange( -90 - res/2,   90 + res/2 + res, res)

lon_centers = lon_edges[:-1] + res/2
lat_centers = lat_edges[:-1] + res/2

ds_binned = (
    ds
    .groupby_bins("lat", lat_edges, right=False)
    .mean(dim="lat")
    .groupby_bins("lon", lon_edges, right=False)
    .mean(dim="lon")
)


ds_coarse = (
    ds_binned
    .rename(lat_bins="lat", lon_bins="lon")
    .assign_coords(lat=lat_centers, lon=lon_centers)
)

z.ds=ds_coarse

z.horizental_interpolate()
z.quantify_met_parameters()

gh = z.ds.z.squeeze(("time", "number")).unstack("site_index")
gh = gh.drop_vars(["metpy_crs","time"])
gh.to_netcdf('reference/z.nc')

z.geopotential_to_orthometric()
z.orthometric_to_ellipsoidal()


gh = z.ds.h.squeeze(("time", "number")).unstack("site_index")

gh = gh.drop_vars(["metpy_crs","time"])

gh.to_netcdf('reference/h.nc')