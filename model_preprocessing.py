
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
import time
from glob import glob

import numpy as np
import xarray as xr
from natsort import natsorted
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
import matplotlib.tri as mtri

from utils import getConfigurationByID, daysBetweenDates
from io_utils import open_dataset_flexible, open_mfdataset_flexible, save_dataset_flexible, detect_format


# -------------------------
# Mesh utilities (GR3 + cache)
# -------------------------

_MESH_CACHE = {}  # gr3_path -> (x, y, tri, triang, finder, kdtree)
_REGULAR_MODEL_TYPES = {"reg", "regular", "cf", "cf_compl", "cf_compliant"}
_UNSTRUCTURED_MODEL_TYPES = {"u", "unstr", "unstruct", "unstructured"}


def _as_mapping(obj):
    """Best-effort conversion of config object/dict into a plain mapping."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())
        except Exception:
            pass
    if hasattr(obj, "keys"):
        out = {}
        for key in obj.keys():
            try:
                out[key] = obj[key]
            except Exception:
                out[key] = getattr(obj, key)
        return out
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}

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


def _model_type(mconf) -> str:
    return str(mconf.type).lower()


def _is_regular_model(mconf) -> bool:
    return _model_type(mconf) in _REGULAR_MODEL_TYPES


def _is_unstructured_model(mconf) -> bool:
    return _model_type(mconf) in _UNSTRUCTURED_MODEL_TYPES


def _resolve_model_variable(mconf, variable: str) -> str:
    mapping = _as_mapping(getattr(mconf, "variables", None))
    return mapping.get(variable, variable)


def _ensure_observation_variable(ds: xr.Dataset, variable: str, mconf):
    """Ensure canonical observation variable exists in dataset, using mapped alias if needed."""
    if variable in ds.data_vars:
        return variable

    mapped = _resolve_model_variable(mconf, variable)
    if mapped != variable and mapped in ds.data_vars:
        ds[variable] = ds[mapped]
        return variable

    return None


def _observation_mask(sat: xr.Dataset, first_time, last_time):
    return (sat.time >= first_time) & (sat.time <= last_time)


def _load_dataset(path: str) -> xr.Dataset:
    return open_dataset_flexible(path)


def _merge_variable_dataset(base: xr.Dataset | None, incoming: xr.Dataset) -> xr.Dataset:
    if base is None:
        return incoming
    for name in incoming.data_vars:
        if name not in base.data_vars:
            base[name] = incoming[name]
    return base


def _model_output_path(outdir: str, date: str, dataset: str, obs_type: str) -> str:
    if obs_type == "sat":
        return os.path.join(outdir, f"{date}_{dataset}.nc")
    return os.path.join(outdir, f"{date}_{dataset}_{obs_type}.nc")


def _daily_cache_paths(outdir: str, day: str, dataset: str, obs_type: str, variables: list[str]):
    if obs_type == "sat":
        return [os.path.join(outdir, f"{day}_{dataset}_{obs_type}.nc")]
    return [os.path.join(outdir, f"{day}_{dataset}_{obs_type}_{var}.nc") for var in variables]


def _variables_to_process(conf_path: str, obs_type: str) -> list[str]:
    variables = ['hs']
    if obs_type != 'buoy':
        return variables
    conf_buoy = getConfigurationByID(conf_path, "buoy_preproc")
    conf_vars = getattr(conf_buoy, 'variables', None)
    variables.extend([var for var in _as_mapping(conf_vars).keys() if var != 'hs'])
    if 'mwd' in variables:
        variables.remove('mwd')
        variables.extend(['mwd_sin', 'mwd_cos'])
    return variables


def _obs_input_file(outdir: str, date: str, obs_type: str, variable: str, conf_model, conf_sat):
    if obs_type == 'sat':
        return (conf_model.datasets.sat.path).format(
            out_dir=outdir,
            date=date,
            sigma=conf_sat.processing.filters.zscore.sigma,
        )
    buoy_path_template = (conf_model.datasets.buoy.path).format(out_dir=outdir, date=date)
    if buoy_path_template.endswith('.nc'):
        buoy_path_template = buoy_path_template[:-3]
    return f"{buoy_path_template}_{variable}.nc"


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

    def __init__(self, conf, model: xr.Dataset, dataset: str, obs_points_lonlat: np.ndarray, variable: str = 'hs'):
        self.conf = conf
        self.model = model
        self.dataset = dataset
        self.obs_points = obs_points_lonlat  # shape (nobs, 2) columns: lon, lat
        self.variable = variable  # which variable to process

        mconf = self.conf.datasets.models[dataset]
        self.mconf = mconf
        self.model_type = _model_type(mconf)

        self.interp_type = getattr(mconf, "interp_type", "nearest")
        if self.interp_type == "auto":
            if _is_regular_model(mconf):
                self.interp_type = "bilinear"
            elif _is_unstructured_model(mconf):
                self.interp_type = "barycentric"
            else:
                self.interp_type = "nearest"

        self.outputs = None  # "indexed" or "values"
        self.idxs = None
        self.mask_dist = None
        self.model_hs = None  # stores interpolated values for the variable

        self.read_and_run()

    def read_and_run(self):
        if _is_regular_model(self.mconf):
            if self.interp_type == "bilinear":
                self._run_bilinear_regular()
                self.outputs = "values"
            else:
                self._run_scattered_cf_like()
        elif _is_unstructured_model(self.mconf):
            self._run_unstructured()
        else:
            raise ValueError(f"Unknown model type '{self.mconf.type}' for dataset '{self.dataset}'")

    def _get_model_variable_name(self):
        """Get actual model variable name for observation variable using config mapping."""
        return _resolve_model_variable(self.mconf, self.variable)

    # ---- regular/CF as scattered (nearest or LinearNDInterpolator)
    def _run_scattered_cf_like(self):
        m = self.model
        model_var = self._get_model_variable_name()
        if model_var not in m.data_vars:
            raise ValueError(f"Model variable '{model_var}' (mapped from obs '{self.variable}') not found in dataset '{self.dataset}'. Available: {list(m.data_vars.keys())}")
        
        hs_da = m[model_var]

        stacked = hs_da.stack(node=(self.mconf.lat, self.mconf.lon)).dropna(dim="node")

        lon = stacked[self.mconf.lon].values if self.mconf.lon in stacked.coords else stacked["longitude"].values
        lat = stacked[self.mconf.lat].values if self.mconf.lat in stacked.coords else stacked["latitude"].values

        hs = stacked.values

        if self.interp_type == "nearest":
            self._run_nearest(lon, lat, hs)
            self.outputs = "indexed"
        elif self.interp_type == "linear":
            self._run_linear_scattered(lon, lat, hs)
            self.outputs = "values"
        else:
            raise ValueError(f"interp_type '{self.interp_type}' incompatible with regular/cf dataset '{self.dataset}'")

    # ---- unstructured: nodes are already 1D (node)
    def _run_unstructured(self):
        m = self.model
        model_var = self._get_model_variable_name()
        if model_var not in m.data_vars:
            raise ValueError(f"Model variable '{model_var}' (mapped from obs '{self.variable}') not found in dataset '{self.dataset}'. Available: {list(m.data_vars.keys())}")

        lon = m[self.mconf.lon].values
        lat = m[self.mconf.lat].values
        hs = m[model_var].values

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

        dist, _ = tree.query(self.obs_points, k=1)
        self.mask_dist = dist <= float(self.conf.filters.max_distance_in_space)

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
        model_var = self._get_model_variable_name()
        if model_var not in m.data_vars:
            raise ValueError(f"Model variable '{model_var}' (mapped from obs '{self.variable}') not found in dataset '{self.dataset}'. Available: {list(m.data_vars.keys())}")

        lon = m[self.mconf.lon].values
        lat = m[self.mconf.lat].values
        hs_da = m[model_var]

        if lon.ndim != 1 or lat.ndim != 1:
            print(f"[WARN] '{self.dataset}': lon/lat not 1D; bilinear impossible. Falling back to scattered linear.")
            self.interp_type = "linear"
            self._run_scattered_cf_like()
            return

        hs_da = hs_da.transpose(self.mconf.time, self.mconf.lat, self.mconf.lon)

        if not np.all(np.diff(lon) > 0):
            hs_da = hs_da.isel({self.mconf.lon: slice(None, None, -1)})
            lon = lon[::-1]
        if not np.all(np.diff(lat) > 0):
            hs_da = hs_da.isel({self.mconf.lat: slice(None, None, -1)})
            lat = lat[::-1]

        pts_latlon = np.column_stack([self.obs_points[:, 1], self.obs_points[:, 0]])

        ntime = hs_da.sizes[self.mconf.time]
        nobs = self.obs_points.shape[0]
        print(f"      Loading {ntime} timesteps into memory for interpolation...")
        hs_data = hs_da.values

        out = np.full((ntime, nobs), np.nan, dtype=np.float32)
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

        t_index = finder(self.obs_points[:, 0], self.obs_points[:, 1])  # (lon,lat) == (x,y)
        inside = t_index >= 0

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

def getSeries(model: xr.Dataset, sat: xr.Dataset, conf, conf_sat, dataset: str, satname: str, variable: str = 'hs'):
    """
    Nearest-in-time sampling (keeps max_distance_in_time filter).
    Works with:
      - nearest: (time,node)+idx
      - bilinear/linear/barycentric: (time,obs) values
    
    Parameters
    ----------
    variable : str
        Variable name to process (e.g., 'hs', 'mwp')
    """
    # limited overlap in time
    first_time = max(sat.time.min(), model.time.min())
    last_time  = min(sat.time.max(), model.time.max())
    print(f"    Time range: {first_time.values} to {last_time.values} (variable: {variable})")

    sat_sub = sat.isel(obs=(sat.time >= first_time) & (sat.time <= last_time))
    model_sub = model.isel(time=(model.time >= first_time) & (model.time <= last_time))

    if sat_sub.sizes.get("obs", 0) == 0 or model_sub.sizes.get("time", 0) == 0:
        print(f"    No overlap in time for {variable}.")
        return None

    # nearest model time index for each obs
    model_times = model_sub.time.values
    sat_times = sat_sub.time.values
    time_idxs = np.array([np.argmin(np.abs(model_times - t)) for t in sat_times])

    # time filter (hours)
    max_dt_h = float(conf.filters.max_distance_in_time)
    time_filt = np.array([
        np.abs(((model_sub.time[time_idxs[i]] - sat_sub.time[i]).values / np.timedelta64(1, "h"))) <= max_dt_h
        for i in range(sat_sub.sizes["obs"])
    ], dtype=bool)

    if not np.any(time_filt):
        print(f"    No obs passing max_distance_in_time for {variable}.")
        return None

    obs_points, _ = get_obs_XYT(sat_sub, conf_sat)
    obs_points_tf = obs_points[time_filt, :]

    try:
        data = Reader(conf, model_sub, dataset, obs_points_tf, variable)
    except ValueError as e:
        print(f"    ✗ Skipping variable '{variable}': {e}")
        return None

    if data.outputs == "indexed":
        # model_hs is (time,node); idxs is (obs_tf,)
        model_vals = data.model_hs[time_idxs[time_filt][data.mask_dist], data.idxs[data.mask_dist]]
    else:
        # model_hs is (time, obs_tf)
        model_vals = data.model_hs[time_idxs[time_filt], :][data.mask_dist]

    if model_vals.size == 0:
        print(f"    No obs passing max_distance_in_space / inside-mesh for {variable}.")
        return None

    # Slice satellite dataset consistently: first time_filt, then spatial mask
    sat_out = sat_sub.isel(obs=time_filt).isel(obs=data.mask_dist)

    # Keep canonical obs variable name even when source provides alias (e.g., mwp_vgta -> mwp)
    _ensure_observation_variable(sat_out, variable, conf.datasets.models[dataset])

    # Write model predictions
    sat_out[f"model_{variable}"] = ('obs', model_vals)
    
    # Set attributes on observation variable if it exists
    if variable in sat_out.data_vars:
        sat_out[f"{variable}"].attrs["satellite_file"] = satname

    return sat_out


def getSeriesLinear(model: xr.Dataset, sat: xr.Dataset, conf, conf_sat, dataset: str, satname: str, variable: str = 'hs'):
    """
    Space first, then linear time interpolation at each obs.
    IMPORTANT: Enforces max_distance_in_space (mask_dist). Does NOT enforce max_distance_in_time
              because it interpolates over the model time axis; if you want a time-window guard,
              add a check on nearest time distance too.
    
    Parameters
    ----------
    variable : str
        Variable name to process (e.g., 'hs', 'mwp')
    """
    # overlap in time (full model range)
    first_time = model.time.min()
    last_time  = model.time.max()
    sat_sub = sat.sel(obs=(sat.time >= first_time) & (sat.time <= last_time))

    if sat_sub.sizes.get("obs", 0) == 0:
        print(f"    No sat obs in model time range for {variable}.")
        return None

    obs_points, _ = get_obs_XYT(sat_sub, conf_sat)

    try:
        data = Reader(conf, model, dataset, obs_points, variable)
    except ValueError as e:
        print(f"    ✗ Skipping variable '{variable}': {e}")
        return None
    # data.model_hs is (time, obs) for bilinear/linear/barycentric; for nearest it's (time,node)
    # For linear-time interpolation we NEED values per-obs; if nearest, convert to per-obs first.
    if data.outputs == "indexed":
        # Build per-obs time series from nearest node indices
        vals_per_obs = data.model_hs[:, data.idxs]  # (time, obs)
    else:
        vals_per_obs = data.model_hs  # (time, obs)

    # Apply spatial filter BEFORE time interpolation to keep semantics consistent
    mask = data.mask_dist
    if not np.any(mask):
        print(f"    No obs passing max_distance_in_space / inside-mesh for {variable}.")
        return None

    sat_out = sat_sub.isel(obs=mask)
    vals_per_obs = vals_per_obs[:, mask]

    # Keep canonical obs variable name even when source provides alias (e.g., mwp_vgta -> mwp)
    _ensure_observation_variable(sat_out, variable, conf.datasets.models[dataset])

    sat_t = datetime64_to_timestamp(sat_out.time.values)
    mod_t = datetime64_to_timestamp(model.time.values)

    model_vals = np.full(sat_out.sizes["obs"], np.nan, dtype=float)
    for iobs in range(sat_out.sizes["obs"]):
        f = interp1d(mod_t, vals_per_obs[:, iobs], bounds_error=False, fill_value=np.nan)
        model_vals[iobs] = float(f(sat_t[iobs]))

    # Write model predictions
    sat_out[f"model_{variable}"] = ('obs', model_vals)
    
    # Set attributes on observation variable if it exists
    if variable in sat_out.data_vars:
        sat_out[f"{variable}"].attrs["satellite_file"] = satname

    return sat_out


# -------------------------
# Preprocess + submit
# -------------------------

def preprocesser(ds: xr.Dataset, varnames):
    """Small helper: keep model dataset as-is. Reader class will extract needed variables."""
    # Don't filter - just return the full dataset
    # The Reader class knows how to extract the right variables and coordinates
    return ds


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
        obs_path_template = (conf_model.datasets.buoy.path).format(
            out_dir=outdir,
            date=date,
        )
        # Remove .nc extension for per-variable file construction
        if obs_path_template.endswith('.nc'):
            obs_path_template = obs_path_template[:-3]
        obs_path = obs_path_template  # Use template as base path for reference
        output_suffix = 'buoy_series_with_models'
    else:
        obs_path = (conf_model.datasets.sat.path).format(
            out_dir=outdir,
            date=date,
            sigma=conf_sat.processing.filters.zscore.sigma,
        )
        output_suffix = 'sat_series_with_models'
    
    print(f"\n{'='*60}")
    print(f"LOADING {obs_type.upper()} OBSERVATIONS")
    print(f"{'='*60}")
    
    variables_to_process = _variables_to_process(conf_path, obs_type)
    
    # Load observation file(s) - keep per-variable independence for buoy data
    sat_by_var = {}  # variable -> dataset
    if obs_type == 'buoy':
        # Load each per-variable file independently (do NOT merge to preserve independent obs counts)
        obs_files = [f"{obs_path_template}_{var}.nc" for var in variables_to_process]
        print(f"Files: {[os.path.basename(f) for f in obs_files]}")
        
        for var in variables_to_process:
            var_file = f"{obs_path_template}_{var}.nc"
            if os.path.exists(var_file):
                sat_by_var[var] = open_dataset_flexible(var_file)
                print(f"  Loaded {var}: {sat_by_var[var].sizes.get('obs', 0)} observations")
        
        if not sat_by_var:
            raise RuntimeError(f"No buoy observation files found: {obs_files}")
        
        # Use first variable for reference (for spatial/temporal union bounds)
        first_var = variables_to_process[0]
        sat = sat_by_var.get(first_var)
    else:
        obs_path = (conf_model.datasets.sat.path).format(
            out_dir=outdir,
            date=date,
            sigma=conf_sat.processing.filters.zscore.sigma,
        )
        print(f"File: {obs_path}")
        sat = open_dataset_flexible(obs_path)
        sat_by_var[variables_to_process[0]] = sat  # Use single sat file for all variables
    
    print(f"  Number of observations (reference): {sat.sizes.get('obs', 0)}")
    print(f"  Variables: {list(sat.data_vars.keys())}")
    
    # Check if all variable-specific output files exist and are up-to-date
    all_outputs_exist = True
    for variable in variables_to_process:
        outname_var = os.path.join(outdir, f"{date}_{output_suffix}.nc") if obs_type == 'sat' else os.path.join(outdir, f"{date}_{output_suffix}_{variable}.nc")
        if not os.path.exists(outname_var):
            all_outputs_exist = False
            break
        obs_input = _obs_input_file(outdir, date, obs_type, variable, conf_model, conf_sat)
        if not os.path.exists(obs_input):
            all_outputs_exist = False
            break
        obs_mtime = os.path.getmtime(obs_input)
        out_mtime = os.path.getmtime(outname_var)
        if out_mtime <= obs_mtime:
            all_outputs_exist = False
            break
    
    if all_outputs_exist:
        print(f"\n✓ All output files exist and are up-to-date")
        # Load the first variable file for return value
        first_var = variables_to_process[0]
        if obs_type == 'sat':
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}.nc")
        else:
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}_{first_var}.nc")
        return open_dataset_flexible(outname_var)
    else:
        print(f"\n⚠ Some output files missing or outdated - reprocessing")

    print(f"\n{'='*60}")
    print(f"PROCESSING {len(conf_model.datasets.models)} MODEL(S)")
    print(f"{'='*60}\n")

    # Keep track of per-model, per-variable merged outputs
    model_names = list(conf_model.datasets.models.keys())
    per_model_var_paths = {var: {} for var in variables_to_process}
    for dataset_idx, dataset in enumerate(conf_model.datasets.models, 1):
        print(f"\n{'='*60}")
        print(f"MODEL {dataset_idx}/{len(conf_model.datasets.models)}: {dataset.upper()}")
        print(f"{'='*60}")
        print(f"  Processing days for {dataset}...")
        days_list = daysBetweenDates(start_date, end_date)
        print(f"  Date range: {start_date} to {end_date} ({len(days_list)} days)\n")
        # 1) Build missing per-day variable files by loading each day model only once
        for day_idx, day in enumerate(days_list, 1):
            missing_vars = []
            for var in variables_to_process:
                outname_day_var = os.path.join(outdir, f"{day}_{dataset}_{obs_type}_{var}.nc")
                if not os.path.exists(outname_day_var):
                    missing_vars.append(var)

            if len(missing_vars) == 0:
                print(f"  [{day_idx}/{len(days_list)}] Day {day}: all variable day-files cached")
                continue

            print(f"  [{day_idx}/{len(days_list)}] Processing day {day} (missing vars: {missing_vars})")
            filledPath = (conf_model.datasets.models[dataset].path).format(experiment=dataset, day=day)
            print(f"    Searching: {filledPath}")
            files = natsorted(glob(filledPath))
            if len(files) == 0:
                print(f"    ⚠ No files found for {dataset} on {day}")
                continue

            try:
                print(f"    Loading model data once for day {day}...")
                obs_lons = sat['longitude'].values
                obs_lats = sat['latitude'].values
                lon_min, lon_max = obs_lons.min() - 0.5, obs_lons.max() + 0.5
                lat_min, lat_max = obs_lats.min() - 0.5, obs_lats.max() + 0.5
                print(f"    Obs bbox: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

                if len(files) == 1 and detect_format(files[0]) == 'zarr':
                    print(f"    Reading Zarr format: {files[0]}")
                    model = xr.open_zarr(files[0], chunks={'time': 24})
                elif any(detect_format(f) == 'zarr' for f in files):
                    print(f"    Reading {len(files)} Zarr stores...")
                    datasets = [xr.open_zarr(f, chunks={'time': 24}) for f in files]
                    model = xr.concat(datasets, dim='time')
                else:
                    model = xr.open_mfdataset(files, combine="by_coords", chunks={'time': 24})

                varnames = conf_model.datasets.models[dataset]
                model = preprocesser(model, varnames)
                model_type = conf_model.datasets.models[dataset].type.lower()
                if model_type in ['reg', 'regular', 'cf', 'cf_compl', 'cf_compliant']:
                    lon_var = varnames.lon
                    lat_var = varnames.lat
                    if lon_var in model.coords and lat_var in model.coords:
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

                for var in missing_vars:
                    print(f"      Processing variable: {var}")
                    # For buoy data, use variable-specific observations; for satellite, use single dataset
                    sat_var = sat_by_var.get(var) if obs_type == 'buoy' else sat
                    if itp in ["linear", "bilinear", "barycentric", "auto"]:
                        daily_ds_var = getSeriesLinear(model, sat_var, conf_model, conf_sat, dataset, obs_path, var)
                    else:
                        daily_ds_var = getSeries(model, sat_var, conf_model, conf_sat, dataset, obs_path, var)

                    if daily_ds_var is None:
                        print(f"      ⚠ No data for {var}")
                        continue

                    var_name = f"model_{var}"
                    obs_var_name = var if var in daily_ds_var.data_vars else None
                    if obs_var_name is None:
                        mapped_var = _resolve_model_variable(conf_model.datasets.models[dataset], var)
                        if mapped_var in daily_ds_var.data_vars:
                            obs_var_name = mapped_var

                    data_vars = {var_name: daily_ds_var[var_name]}
                    if obs_var_name is not None:
                        data_vars[var] = daily_ds_var[obs_var_name]

                    var_ds = xr.Dataset(
                        data_vars=data_vars,
                        coords={'obs': daily_ds_var.coords['obs'], 'model': [dataset]},
                    )
                    for coord_var in ['time', 'longitude', 'latitude', 'station', 'satellite']:
                        if coord_var in daily_ds_var.data_vars:
                            var_ds[coord_var] = daily_ds_var[coord_var]

                    outname_day_var = os.path.join(outdir, f"{day}_{dataset}_{obs_type}_{var}.nc")
                    save_dataset_flexible(var_ds, outname_day_var)

            except Exception as e:
                print(f"    ✗ ERROR: {dataset} {day} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 2) Merge per-day files into per-model, per-variable files
        for var in variables_to_process:
            outname_ds = os.path.join(outdir, f"{date}_{dataset}_{output_suffix}_{var}.nc")
            if os.path.exists(outname_ds):
                print(f"  ✓ Using cached model-variable file: {dataset} - {var}")
                per_model_var_paths[var][dataset] = outname_ds
                continue

            per_day_var_files = []
            for day in days_list:
                outname_day_var = os.path.join(outdir, f"{day}_{dataset}_{obs_type}_{var}.nc")
                if os.path.exists(outname_day_var):
                    per_day_var_files.append(outname_day_var)

            if len(per_day_var_files) == 0:
                print(f"\n  ⚠ No output created for dataset {dataset}, variable {var} (no valid days)\n")
                continue

            print(f"\n  Concatenating {len(per_day_var_files)} day(s) for {dataset}, variable {var}...")
            ds_list = []
            for fpath in per_day_var_files:
                ds_part = open_dataset_flexible(fpath)
                ds_list.append(ds_part.load())
                if hasattr(ds_part, "close"):
                    ds_part.close()
            ds_model_out = xr.concat(ds_list, dim='obs')
            print(f"  Saving per-model, per-variable file: {os.path.basename(outname_ds)}")
            save_dataset_flexible(ds_model_out, outname_ds)
            print(f"  ✓ {dataset}, {var}: {len(ds_model_out.obs)} observations (indices from ALLSAT)\n")
            per_model_var_paths[var][dataset] = outname_ds

    # For buoy data, each variable has independent obs; for satellite, all variables share obs
    obs_ref_by_var = {}
    for variable in variables_to_process:
        if obs_type == 'buoy':
            obs_ref_by_var[variable] = sat_by_var[variable]['obs'].values
        else:
            obs_ref_by_var[variable] = sat['obs'].values
    
    # Use first variable's obs as reference for output structure (for metadata)
    first_var = variables_to_process[0]
    obs_ref = obs_ref_by_var[first_var]
    n_valid = len(obs_ref)

    # Collect model predictions aligned by obs index for each variable independently
    model_vals_by_var = {}  # variable -> (nobs_var, n_models) array
    for variable in variables_to_process:
        obs_ref_var = obs_ref_by_var[variable]
        n_obs_var = len(obs_ref_var)
        
        print(f"    Aligning {variable} predictions across {len(model_names)} model(s)...")
        print(f"      Variable obs count: {n_obs_var}")
        
        model_vals = np.full((n_obs_var, len(model_names)), np.nan, dtype=np.float32)  # (obs, model)
        ref_obs_to_idx = {obs: i for i, obs in enumerate(obs_ref_var)}
        
        for model_idx, model_name in enumerate(model_names):
            path_model_var = per_model_var_paths.get(variable, {}).get(model_name)
            if not path_model_var or not os.path.exists(path_model_var):
                continue
            ds = open_dataset_flexible(path_model_var)
            var_name = f"model_{variable}"
            if var_name not in ds.data_vars:
                if hasattr(ds, "close"):
                    ds.close()
                continue
            vals = ds[var_name].values
            if vals.ndim == 2:
                vals = vals[:, 0]
            ds_obs = ds.obs.values

            # Deterministic alignment on obs index:
            # - sort by obs
            # - de-duplicate obs indices (keep first occurrence)
            if ds_obs.size > 0:
                order = np.argsort(ds_obs)
                ds_obs_sorted = ds_obs[order]
                vals_sorted = vals[order]
                unique_obs, first_idx = np.unique(ds_obs_sorted, return_index=True)
                vals_unique = vals_sorted[first_idx]

                matched_count = 0
                for local_idx, obs_val in enumerate(unique_obs):
                    ref_idx = ref_obs_to_idx.get(obs_val)
                    if ref_idx is not None:
                        model_vals[ref_idx, model_idx] = vals_unique[local_idx]
                        matched_count += 1
                print(f"      {model_name}: matched {matched_count}/{n_obs_var} obs for {variable}")

            if hasattr(ds, "close"):
                ds.close()
        model_vals_by_var[variable] = model_vals

    print(f"\n  Creating per-variable merged datasets...\n")
    for var_idx, variable in enumerate(variables_to_process, 1):
        if obs_type == 'sat':
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}.nc")
        else:
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}_{variable}.nc")
        
        print(f"  [{var_idx}/{len(variables_to_process)}] Creating {os.path.basename(outname_var)} ({variable})")
        
        # Load metadata from variable-specific observation file
        ds_base = sat_by_var.get(variable) if obs_type == 'buoy' else sat
        if ds_base is None:
            print(f"    ⚠ Skipping {variable}: no observation data")
            continue
        
        ds_base = ds_base.load()
        obs_ref_var = obs_ref_by_var[variable]
        n_obs_var = len(obs_ref_var)
        
        data_vars = {}
        coords_data = {'obs': obs_ref_var, 'model': model_names}
        
        # Use metadata from variable's observation file
        if 'time' in ds_base.data_vars:
            data_vars['time'] = (('obs',), ds_base['time'].values)
        if 'longitude' in ds_base.data_vars:
            data_vars['longitude'] = (('obs',), ds_base['longitude'].values)
        if 'latitude' in ds_base.data_vars:
            data_vars['latitude'] = (('obs',), ds_base['latitude'].values)
        
        # Add observation variable if it exists
        if variable in ds_base.data_vars:
            data_vars[variable] = (('obs',), ds_base[variable].values)
        
        # Add model predictions for this variable
        model_vals_var = model_vals_by_var.get(variable, np.full((n_obs_var, len(model_names)), np.nan, dtype=np.float32))
        data_vars[f"model_{variable}"] = (('obs', 'model'), model_vals_var)
        
        # Add station or satellite identifier if present
        if 'station' in ds_base.data_vars:
            data_vars['station'] = (('obs',), ds_base['station'].values)
        elif 'satellite' in ds_base.data_vars:
            data_vars['satellite'] = (('obs',), ds_base['satellite'].values)
        
        ds_out = xr.Dataset(data_vars=data_vars, coords=coords_data)
        save_dataset_flexible(ds_out, outname_var)
        print(f"      ✓ Saved: {n_obs_var} observations")
    
    print(f"\n✓ Saved per-variable outputs:")
    for variable in variables_to_process:
        if obs_type == 'sat':
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}.nc")
        else:
            outname_var = os.path.join(outdir, f"{date}_{output_suffix}_{variable}.nc")
        if os.path.exists(outname_var):
            ds_check = open_dataset_flexible(outname_var)
            n_obs_check = len(ds_check.obs)
            if hasattr(ds_check, "close"):
                ds_check.close()
            print(f"  ✓ {variable}: {n_obs_check} observations")
    print(f"  Models: {list(conf_model.datasets.models.keys())}")
    print(f"  Variables: {variables_to_process}")
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}\n")
    first_var = variables_to_process[0]
    if obs_type == 'sat':
        outname_var = os.path.join(outdir, f"{date}_{output_suffix}.nc")
    else:
        outname_var = os.path.join(outdir, f"{date}_{output_suffix}_{first_var}.nc")
    return open_dataset_flexible(outname_var)
