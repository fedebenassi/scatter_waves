"""
model_preprocessing.py

Spatial interpolation choices:
- regular grid  -> bilinear (RegularGridInterpolator) when lon/lat are 1D
- unstructured  -> barycentric on GR3 triangles (matplotlib.tri TriFinder)
- fallback      -> scattered linear (LinearNDInterpolator) if regular grid is actually curvilinear

Keeps filters from conf:
- max_distance_in_time (only in getSeries; getSeriesLinear uses full overlap but still enforces max_distance_in_space)
- max_distance_in_space (enforced in BOTH getSeries and getSeriesLinear via mask_dist)
"""

import os
from glob import glob

import numpy as np
import xarray as xr
from natsort import natsorted
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
import matplotlib.tri as mtri

from utils import getConfigurationByID, daysBetweenDates


# -------------------------
# Mesh utilities (GR3 + cache)
# -------------------------

_MESH_CACHE = {}  # gr3_path -> (x, y, tri, triang, finder, kdtree)

def read_gr3_mesh(gr3_path: str):
    """
    Read SELFE/SCHISM .gr3 mesh.
    Returns:
      x (N,), y (N,), tri (M,3)  0-based node indices
    Notes:
      - Skips non-tri elements (nvert != 3). If you need mixed elements, you must triangulate polygons.
    """
    with open(gr3_path, "r") as f:
        _ = f.readline()  # title
        ne_np = f.readline().split()
        if len(ne_np) < 2:
            raise ValueError(f"Bad gr3 header in {gr3_path}")
        ne = int(ne_np[0])
        npn = int(ne_np[1])

        x = np.empty(npn, dtype=float)
        y = np.empty(npn, dtype=float)

        # nodes: id x y depth
        for i in range(npn):
            parts = f.readline().split()
            x[i] = float(parts[1])
            y[i] = float(parts[2])

        tri = np.empty((ne, 3), dtype=np.int64)
        tri_count = 0

        # elements: id nvert n1 n2 n3 ...
        for _ in range(ne):
            parts = f.readline().split()
            nvert = int(parts[1])
            if nvert != 3:
                continue
            n1 = int(parts[2]) - 1
            n2 = int(parts[3]) - 1
            n3 = int(parts[4]) - 1
            tri[tri_count, :] = (n1, n2, n3)
            tri_count += 1

        tri = tri[:tri_count, :]
        if tri.shape[0] == 0:
            raise ValueError(f"No triangles found in {gr3_path}")

    return x, y, tri


def get_gr3_structures(gr3_path: str):
    """Cached triangulation + finder + KDTree for a GR3 mesh."""
    if gr3_path not in _MESH_CACHE:
        x, y, tri = read_gr3_mesh(gr3_path)
        triang = mtri.Triangulation(x, y, triangles=tri)
        finder = triang.get_trifinder()
        kdtree = cKDTree(np.column_stack([x, y]))
        _MESH_CACHE[gr3_path] = (x, y, tri, triang, finder, kdtree)
    return _MESH_CACHE[gr3_path]


def barycentric_weights(px, py, x1, y1, x2, y2, x3, y3):
    """Vectorized barycentric weights for points P(px,py) in triangle (1,2,3)."""
    den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
    w3 = 1.0 - w1 - w2
    return w1, w2, w3


# -------------------------
# Sat/buoy observation helpers
# -------------------------

def get_obs_XYT(ds: xr.Dataset, conf_sat):
    """Return list of (lon, lat, time) aligned with obs dimension.
    
    Works for both satellite and buoy data - uses standardized variable names
    (longitude, latitude, time) created by preprocessing steps.
    """
    # Use standardized names (satellite_preprocessing and buoy_preprocessing both output these)
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    tim = ds['time'].values
    return np.column_stack([lon, lat]), tim


def datetime64_to_timestamp(dt64_arr):
    """Convert numpy datetime64 array to unix seconds float."""
    return (dt64_arr - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")


# -------------------------
# Reader (spatial interpolation only)
# -------------------------

class Reader:
    """
    Produces:
      - self.model_hs: 2D array (ntime, nobs_in) for value-based interps
        OR 2D array (ntime, nnodes) for nearest-index mode (same as original style)
      - self.mask_dist: boolean mask on obs_in (nobs_in,) enforcing max_distance_in_space
      - if nearest: self.idxs (nobs_in,) nearest node index
    """

    def __init__(self, conf, model: xr.Dataset, dataset: str, obs_points_lonlat: np.ndarray):
        self.conf = conf
        self.model = model
        self.dataset = dataset
        self.obs_points = obs_points_lonlat  # shape (nobs, 2) columns: lon, lat

        mconf = self.conf.datasets.models[dataset]
        self.mconf = mconf

        self.interp_type = getattr(mconf, "interp_type", "nearest")
        if self.interp_type == "auto":
            tp = mconf.type.lower()
            if tp in ["reg", "regular", "cf", "cf_compl", "cf_compliant"]:
                self.interp_type = "bilinear"
            elif tp in ["u", "unstr", "unstruct", "unstructured"]:
                self.interp_type = "barycentric"
            else:
                self.interp_type = "nearest"

        self.outputs = None  # "indexed" or "values"
        self.idxs = None
        self.mask_dist = None
        self.model_hs = None

        self.read_and_run()

    def read_and_run(self):
        tp = self.mconf.type.lower()

        if tp in ["reg", "regular", "cf", "cf_compl", "cf_compliant"]:
            if self.interp_type == "bilinear":
                self._run_bilinear_regular()
                self.outputs = "values"
            else:
                # Scattered view for regular/CF when using nearest/linear fallback
                self._run_scattered_cf_like()
        elif tp in ["u", "unstr", "unstruct", "unstructured"]:
            self._run_unstructured()
        else:
            raise ValueError(f"Unknown model type '{self.mconf.type}' for dataset '{self.dataset}'")

    # ---- regular/CF as scattered (nearest or LinearNDInterpolator)
    def _run_scattered_cf_like(self):
        m = self.model
        hs_da = m[self.mconf.hs]

        # Stack lat/lon dims to a node dim (this is what your original cf() did)
        stacked = hs_da.stack(node=(self.mconf.lat, self.mconf.lon)).dropna(dim="node")

        # Try to locate lon/lat coords after stacking (depends on your dataset naming)
        # If your stacked coords are exactly mconf.lon/mconf.lat, use those.
        if self.mconf.lon in stacked.coords:
            lon = stacked[self.mconf.lon].values
        else:
            # fallback: common names
            lon = stacked["longitude"].values

        if self.mconf.lat in stacked.coords:
            lat = stacked[self.mconf.lat].values
        else:
            lat = stacked["latitude"].values

        hs = stacked.values  # (time, node)

        if self.interp_type == "nearest":
            self._run_nearest(lon, lat, hs)
            self.outputs = "indexed"
        elif self.interp_type == "linear":
            self._run_linear_scattered(lon, lat, hs)
            self.outputs = "values"
        else:
            # If someone sets bilinear but grid isn't usable, we already routed earlier.
            # If someone sets barycentric on a regular grid, refuse.
            raise ValueError(f"interp_type '{self.interp_type}' incompatible with regular/cf dataset '{self.dataset}'")

    # ---- unstructured: nodes are already 1D (node)
    def _run_unstructured(self):
        m = self.model
        lon = m[self.mconf.lon].values
        lat = m[self.mconf.lat].values
        hs = m[self.mconf.hs].values  # (time, node) expected

        if self.interp_type == "nearest":
            self._run_nearest(lon, lat, hs)
            self.outputs = "indexed"
        elif self.interp_type == "linear":
            self._run_linear_scattered(lon, lat, hs)
            self.outputs = "values"
        elif self.interp_type == "barycentric":
            self._run_barycentric_unstructured(hs)
            self.outputs = "values"
        else:
            raise ValueError(f"interpolation type not known: {self.interp_type}")

    # ---- interpolation kernels
    def _run_nearest(self, lon_nodes, lat_nodes, hs_time_node):
        pts_nodes = np.column_stack([lon_nodes, lat_nodes])
        tree = cKDTree(pts_nodes)

        dist, idx = tree.query(self.obs_points, k=1)
        self.mask_dist = dist <= float(self.conf.filters.max_distance_in_space)
        self.idxs = idx
        self.model_hs = hs_time_node  # keep full; later slice (time, idxs)

    def _run_linear_scattered(self, lon_nodes, lat_nodes, hs_time_node):
        pts_nodes = np.column_stack([lon_nodes, lat_nodes])
        tree = cKDTree(pts_nodes)

        # distance mask (keep filter semantics consistent)
        dist, _ = tree.query(self.obs_points, k=1)
        self.mask_dist = dist <= float(self.conf.filters.max_distance_in_space)

        # Evaluate LinearNDInterpolator for each time step
        out = np.full((hs_time_node.shape[0], self.obs_points.shape[0]), np.nan, dtype=float)
        for it in range(hs_time_node.shape[0]):
            itp = LinearNDInterpolator(pts_nodes, hs_time_node[it, :], fill_value=np.nan)
            out[it, :] = itp(self.obs_points[:, 0], self.obs_points[:, 1])

        self.model_hs = out  # (time, obs)

    def _run_bilinear_regular(self):
        """
        Bilinear for rectilinear regular grid.
        Requirements:
          lon and lat must be 1D coordinates
          hs must be indexed by (time, lat, lon) in some order (we handle by name)
        If lon/lat are not 1D, we fall back to scattered 'linear' (and keep filters).
        """
        m = self.model
        lon = m[self.mconf.lon].values
        lat = m[self.mconf.lat].values
        hs_da = m[self.mconf.hs]

        # If not 1D -> fallback to scattered linear (better than lying about bilinear)
        if lon.ndim != 1 or lat.ndim != 1:
            print(f"[WARN] '{self.dataset}': lon/lat not 1D; bilinear impossible. Falling back to scattered linear.")
            # Flatten as scattered and run linear
            self.interp_type = "linear"
            self._run_scattered_cf_like()
            return

        # ensure hs has dims including time, lat, lon by name
        # (xarray will reorder with transpose)
        hs_da = hs_da.transpose(self.mconf.time, self.mconf.lat, self.mconf.lon)

        # Make coordinates increasing by reversing indices (safe)
        if not np.all(np.diff(lon) > 0):
            hs_da = hs_da.isel({self.mconf.lon: slice(None, None, -1)})
            lon = lon[::-1]
        if not np.all(np.diff(lat) > 0):
            hs_da = hs_da.isel({self.mconf.lat: slice(None, None, -1)})
            lat = lat[::-1]

        # Points for interpolation must be (lat, lon)
        pts_latlon = np.column_stack([self.obs_points[:, 1], self.obs_points[:, 0]])

        ntime = hs_da.sizes[self.mconf.time]
        nobs = self.obs_points.shape[0]
        
        # Compute data in memory once to avoid repeated I/O for each timestep
        print(f"      Loading {ntime} timesteps into memory for interpolation...")
        hs_data = hs_da.values  # Load all data at once (ntime, nlat, nlon)
        
        # Pre-allocate output
        out = np.full((ntime, nobs), np.nan, dtype=np.float32)
        
        # Interpolate all timesteps (vectorized when possible)
        print(f"      Interpolating {nobs} points across {ntime} timesteps...")
        for it in range(ntime):
            itp = RegularGridInterpolator(
                (lat, lon),
                hs_data[it],
                bounds_error=False,
                fill_value=np.nan,
                method='linear'  # Explicitly set method
            )
            out[it, :] = itp(pts_latlon)

        # distance filter: compute nearest gridpoint distance on full mesh (lon/lat)
        Lon, Lat = np.meshgrid(lon, lat)  # shape (nlat, nlon)
        tree = cKDTree(np.column_stack([Lon.ravel(), Lat.ravel()]))
        dist, _ = tree.query(self.obs_points, k=1)
        self.mask_dist = dist <= float(self.conf.filters.max_distance_in_space)

        self.model_hs = out  # (time, obs)

    def _run_barycentric_unstructured(self, hs_time_node):
        """
        Barycentric interpolation for unstructured meshes using GR3 triangles.
        hs_time_node must be (time, node) aligned with mesh node order.
        """
        gr3_path = getattr(self.mconf, "mesh_gr3", None)
        if not gr3_path:
            raise ValueError(f"'{self.dataset}': barycentric requested but mesh_gr3 missing in config")

        x, y, tri, triang, finder, kdtree = get_gr3_structures(gr3_path)

        # which triangle contains each point
        t_index = finder(self.obs_points[:, 0], self.obs_points[:, 1])  # (lon,lat) == (x,y)
        inside = t_index >= 0

        # distance filter (nearest node) AND inside-triangle
        dist, _ = kdtree.query(self.obs_points, k=1)
        self.mask_dist = (dist <= float(self.conf.filters.max_distance_in_space)) & inside

        out = np.full((hs_time_node.shape[0], self.obs_points.shape[0]), np.nan, dtype=float)

        valid_idx = np.where(self.mask_dist)[0]
        if valid_idx.size == 0:
            self.model_hs = out
            return

        t = t_index[valid_idx]
        n1 = tri[t, 0]
        n2 = tri[t, 1]
        n3 = tri[t, 2]

        px = self.obs_points[valid_idx, 0]
        py = self.obs_points[valid_idx, 1]

        w1, w2, w3 = barycentric_weights(
            px, py,
            x[n1], y[n1],
            x[n2], y[n2],
            x[n3], y[n3],
        )

        out[:, valid_idx] = (
            hs_time_node[:, n1] * w1[None, :] +
            hs_time_node[:, n2] * w2[None, :] +
            hs_time_node[:, n3] * w3[None, :]
        )

        self.model_hs = out  # (time, obs)


# -------------------------
# Time matching series extraction
# -------------------------

def getSeries(model: xr.Dataset, sat: xr.Dataset, conf, conf_sat, dataset: str, satname: str):
    """
    Nearest-in-time sampling (keeps max_distance_in_time filter).
    Works with:
      - nearest: (time,node)+idx
      - bilinear/linear/barycentric: (time,obs) values
    """
    # limited overlap in time
    first_time = max(sat.time.min(), model.time.min())
    last_time  = min(sat.time.max(), model.time.max())
    print("Time range:", first_time.values, last_time.values)

    sat = sat.isel(obs=(sat.time >= first_time) & (sat.time <= last_time))
    model = model.isel(time=(model.time >= first_time) & (model.time <= last_time))

    if sat.sizes.get("obs", 0) == 0 or model.sizes.get("time", 0) == 0:
        print("No overlap in time.")
        return None

    # nearest model time index for each obs
    model_times = model.time.values
    sat_times = sat.time.values
    time_idxs = np.array([np.argmin(np.abs(model_times - t)) for t in sat_times])

    # time filter (hours)
    max_dt_h = float(conf.filters.max_distance_in_time)
    time_filt = np.array([
        np.abs(((model.time[time_idxs[i]] - sat.time[i]).values / np.timedelta64(1, "h"))) <= max_dt_h
        for i in range(sat.sizes["obs"])
    ], dtype=bool)

    if not np.any(time_filt):
        print("No obs passing max_distance_in_time.")
        return None

    obs_points, _ = get_obs_XYT(sat, conf_sat)
    obs_points_tf = obs_points[time_filt, :]

    data = Reader(conf, model, dataset, obs_points_tf)

    if data.outputs == "indexed":
        # model_hs is (time,node); idxs is (obs_tf,)
        model_hs_1d = data.model_hs[time_idxs[time_filt][data.mask_dist], data.idxs[data.mask_dist]]
    else:
        # model_hs is (time, obs_tf)
        model_hs_1d = data.model_hs[time_idxs[time_filt], :][data.mask_dist]

    if model_hs_1d.size == 0:
        print("No obs passing max_distance_in_space / inside-mesh.")
        return None

    # Slice satellite dataset consistently: first time_filt, then spatial mask
    sat_out = sat.isel(obs=time_filt).isel(obs=data.mask_dist).sel(model=[dataset])

    # Write model_hs
    sat_out["model_hs"].values = model_hs_1d[:, None]  # (obs, 1)
    sat_out["hs"].attrs["satellite_file"] = satname

    return sat_out


def getSeriesLinear(model: xr.Dataset, sat: xr.Dataset, conf, conf_sat, dataset: str, satname: str):
    """
    Space first, then linear time interpolation at each obs.
    IMPORTANT: Enforces max_distance_in_space (mask_dist). Does NOT enforce max_distance_in_time
              because it interpolates over the model time axis; if you want a time-window guard,
              add a check on nearest time distance too.
    """
    # overlap in time (full model range)
    first_time = model.time.min()
    last_time  = model.time.max()
    sat = sat.sel(obs=(sat.time >= first_time) & (sat.time <= last_time))

    if sat.sizes.get("obs", 0) == 0:
        print("No sat obs in model time range.")
        return None

    obs_points, _ = get_obs_XYT(sat, conf_sat)

    data = Reader(conf, model, dataset, obs_points)
    # data.model_hs is (time, obs) for bilinear/linear/barycentric; for nearest it's (time,node)
    # For linear-time interpolation we NEED values per-obs; if nearest, convert to per-obs first.
    if data.outputs == "indexed":
        # Build per-obs time series from nearest node indices
        hs_per_obs = data.model_hs[:, data.idxs]  # (time, obs)
    else:
        hs_per_obs = data.model_hs  # (time, obs)

    # Apply spatial filter BEFORE time interpolation to keep semantics consistent
    mask = data.mask_dist
    if not np.any(mask):
        print("No obs passing max_distance_in_space / inside-mesh.")
        return None

    sat_out = sat.isel(obs=mask).sel(model=[dataset])
    hs_per_obs = hs_per_obs[:, mask]

    sat_t = datetime64_to_timestamp(sat_out.time.values)
    mod_t = datetime64_to_timestamp(model.time.values)

    model_hs = np.full(sat_out.sizes["obs"], np.nan, dtype=float)
    for iobs in range(sat_out.sizes["obs"]):
        f = interp1d(mod_t, hs_per_obs[:, iobs], bounds_error=False, fill_value=np.nan)
        model_hs[iobs] = float(f(sat_t[iobs]))

    sat_out["model_hs"].values = model_hs[:, None]
    sat_out["hs"].attrs["satellite_file"] = satname

    return sat_out


# -------------------------
# Preprocess + submit
# -------------------------

def preprocesser(ds: xr.Dataset, varnames):
    """Small helper: keep relevant vars."""
    keep = [varnames.hs, varnames.time, "node", varnames.lon, varnames.lat]
    if "tri" in ds.variables:
        keep.append("tri")
    keep = [k for k in keep if k in ds.variables or k in ds.coords]
    return ds[keep]


def submit(conf_path: str, start_date: str, end_date: str, obs_type: str = 'sat'):
    """
    Match model outputs to observations (satellite or buoy).
    
    Parameters
    ----------
    conf_path : str
        Path to configuration file
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
    obs_type : str, optional
        Type of observations: 'sat' for satellite (default) or 'buoy' for buoy data
        
    Returns
    -------
    xr.Dataset
        Merged dataset with model predictions matched to observations
    """
    conf_model = getConfigurationByID(conf_path, "model_preproc")
    conf_sat   = getConfigurationByID(conf_path, "sat_preproc")
    date = f"{start_date}_{end_date}"

    outdir = conf_model.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)

    # Select observation dataset based on obs_type
    if obs_type == 'buoy':
        obs_path = (conf_model.datasets.buoy.path).format(
            out_dir=outdir,
            date=date,
        )
        output_suffix = 'buoy_series_with_models'
    else:
        obs_path = (conf_model.datasets.sat.path).format(
            out_dir=outdir,
            date=date,
            sigma=conf_sat.processing.filters.zscore.sigma,
        )
        output_suffix = 'series_with_models'
    
    print(f"\n{'='*60}")
    print(f"LOADING {obs_type.upper()} OBSERVATIONS")
    print(f"{'='*60}")
    print(f"File: {obs_path}")
    
    sat = xr.open_dataset(obs_path)
    print(f"  Number of observations: {sat.sizes.get('obs', 0)}")
    print(f"  Variables: {list(sat.data_vars.keys())}")

    outname_all = os.path.join(outdir, f"{date}_{output_suffix}.nc")
    
    # Check if output exists and is newer than input (avoid unnecessary reprocessing)
    if os.path.exists(outname_all):
        obs_mtime = os.path.getmtime(obs_path)
        out_mtime = os.path.getmtime(outname_all)
        if out_mtime > obs_mtime:
            print(f"\n✓ Output already exists and is up-to-date: {outname_all}")
            return xr.open_dataset(outname_all)
        else:
            print(f"\n⚠ Output exists but observations are newer - reprocessing")

    print(f"\n{'='*60}")
    print(f"PROCESSING {len(conf_model.datasets.models)} MODEL(S)")
    print(f"{'='*60}\n")

    buffer_all = []

    for dataset_idx, dataset in enumerate(conf_model.datasets.models, 1):
        print(f"\n{'='*60}")
        print(f"MODEL {dataset_idx}/{len(conf_model.datasets.models)}: {dataset.upper()}")
        print(f"{'='*60}")
        
        # Check if per-model file already exists
        outname_ds = os.path.join(outdir, f"{date}_{dataset}_{output_suffix}.nc")
        if os.path.exists(outname_ds):
            print(f"  ✓ Using existing dataset file: {os.path.basename(outname_ds)}")
            ds = xr.open_dataset(outname_ds)
            ds_loaded = ds.load()
            ds.close()
            buffer_all.append(ds_loaded)
            continue
        
        print(f"  Processing days for {dataset}...")
        buffer_days = []
        days_list = daysBetweenDates(start_date, end_date)
        print(f"  Date range: {start_date} to {end_date} ({len(days_list)} days)\n")
        
        for day_idx, day in enumerate(days_list, 1):
            print(f"  [{day_idx}/{len(days_list)}] Processing day {day}")
            
            # Check for cached daily file
            outname_day = os.path.join(outdir, f"{day}_{dataset}_{obs_type}.nc")
            if os.path.exists(outname_day):
                print(f"    ✓ Using cached day file: {os.path.basename(outname_day)}")
                ds = xr.open_dataset(outname_day)
                ds_loaded = ds.load()
                ds.close()
                buffer_days.append(ds_loaded)
                continue

            filledPath = (conf_model.datasets.models[dataset].path).format(experiment=dataset, day=day)
            print(f"    Searching: {filledPath}")
            files = natsorted(glob(filledPath))
            if len(files) == 0:
                print(f"    ⚠ No files found for {dataset} on {day}")
                continue
            
            print(f"    Found {len(files)} file(s)")

            try:
                # Load model for that day with optimizations
                print(f"    Loading model data...")
                
                # Get observation bounding box for spatial subsetting (add buffer)
                obs_lons = sat['longitude'].values
                obs_lats = sat['latitude'].values
                lon_min, lon_max = obs_lons.min() - 0.5, obs_lons.max() + 0.5
                lat_min, lat_max = obs_lats.min() - 0.5, obs_lats.max() + 0.5
                print(f"    Obs bbox: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
                
                # Load with chunking for lazy evaluation
                model = xr.open_mfdataset(files, combine="by_coords", chunks={'time': 24})
                varnames = conf_model.datasets.models[dataset]
                model = preprocesser(model, varnames)
                
                # Spatial subset if using regular grid
                model_type = conf_model.datasets.models[dataset].type.lower()
                if model_type in ['reg', 'regular', 'cf', 'cf_compl', 'cf_compliant']:
                    lon_var = varnames.lon
                    lat_var = varnames.lat
                    if lon_var in model.coords and lat_var in model.coords:
                        # Check if coords are 1D
                        if model[lon_var].ndim == 1 and model[lat_var].ndim == 1:
                            lon_vals = model[lon_var].values
                            lat_vals = model[lat_var].values
                            lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
                            lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
                            if lon_mask.any() and lat_mask.any():
                                model = model.sel({lon_var: lon_vals[lon_mask], lat_var: lat_vals[lat_mask]})
                                print(f"    Spatial subset: {model.sizes.get(lon_var, 0)}x{model.sizes.get(lat_var, 0)} grid points")
                        else:
                            print(f"    Skipping spatial subset (2D coordinates)")
                else:
                    print(f"    Skipping spatial subset (unstructured grid)")

                itp = getattr(conf_model.datasets.models[dataset], "interp_type", "nearest")
                print(f"    Interpolation method: {itp}")
                
                import time
                t0 = time.time()
                print(f"    Interpolating to observation points...")
                
                if itp in ["linear", "bilinear", "barycentric", "auto"]:
                    daily_ds = getSeriesLinear(model, sat, conf_model, conf_sat, dataset, obs_path)
                else:
                    daily_ds = getSeries(model, sat, conf_model, conf_sat, dataset, obs_path)
                
                elapsed = time.time() - t0
                print(f"    Interpolation completed in {elapsed:.1f}s")

                if daily_ds is not None:
                    # Save daily cache file
                    daily_ds.to_netcdf(outname_day)
                    print(f"    ✓ Cached: {os.path.basename(outname_day)} ({len(daily_ds.obs)} obs)")
                    buffer_days.append(daily_ds)
                else:
                    print(f"    ⚠ No data returned for {day}")

            except Exception as e:
                print(f"    ✗ ERROR: {dataset} {day} failed: {e}")
                continue

        if len(buffer_days) == 0:
            print(f"\n  ⚠ No output created for dataset {dataset} (no valid days)\n")
            continue

        print(f"\n  Concatenating {len(buffer_days)} day(s) for {dataset}...")
        ds_model_out = xr.concat(buffer_days, dim="obs")
        print(f"  Saving per-model file: {os.path.basename(outname_ds)}")
        ds_model_out.to_netcdf(outname_ds)
        print(f"  ✓ {dataset}: {len(ds_model_out.obs)} observations\n")
        buffer_all.append(ds_model_out)

    if len(buffer_all) == 0:
        raise RuntimeError("No datasets produced any output.")

    print(f"\n{'='*60}")
    print(f"CREATING MERGED OUTPUT FILE")
    print(f"{'='*60}")
    print(f"Merging {len(buffer_all)} model dataset(s)...")
    
    ds_out = xr.merge(buffer_all)
    ds_out["model"] = [ds for ds in conf_model.datasets.models]
    ds_out.to_netcdf(outname_all)
    
    print(f"\n✓ Saved merged output: {outname_all}")
    print(f"  Total observations: {len(ds_out.obs)}")
    print(f"  Models: {list(ds_out.model.values)}")
    print(f"  Variables: {list(ds_out.data_vars.keys())}")
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}\n")
    
    return ds_out
