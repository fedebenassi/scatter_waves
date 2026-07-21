import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from types import SimpleNamespace
from glob import glob
from natsort import natsorted

from utils import getConfigurationByID, daysBetweenDates
try:
    from io_utils import open_dataset_flexible, save_dataset_flexible
except ImportError:
    from .io_utils import open_dataset_flexible, save_dataset_flexible


def _find_wave_height_column(df, preferred_col=None):
    """
    Find wave height column in dataframe, trying common alternatives.
    
    CMEMS buoys may use different variable names: VHM0, VAVH, VGHS, SWH, Hm0, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    preferred_col : str, optional
        Preferred column name to try first
        
    Returns
    -------
    str
        Name of the wave height column found
        
    Raises
    ------
    ValueError
        If no wave height column is found
    """
    # Common wave height variable names in order of preference
    wave_height_cols = ['VHM0', 'VGHS', 'SWH', 'Hm0', 'hs', 'swh', 'wave_height']
    
    # Try preferred column first if specified
    if preferred_col:
        if preferred_col in df.columns:
            return preferred_col
        # Try case-insensitive match
        for col in df.columns:
            if col.upper() == preferred_col.upper():
                return col
    
    # Try common alternatives
    for wave_col in wave_height_cols:
        if wave_col in df.columns:
            return wave_col
        # Try case-insensitive match
        for col in df.columns:
            if col.upper() == wave_col.upper():
                return col
    
    # Last resort: look for columns containing 'hm0' or 'vhm' or 'swh'
    for col in df.columns:
        col_upper = col.upper()
        if any(pattern in col_upper for pattern in ['HM0', 'VHM', 'SWH', 'WAVE']):
            return col
    
    raise ValueError(f"Could not find wave height column. Available columns: {list(df.columns)}")


def _safe_station_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s#[:120]


def _variables_to_process(conf_vars):
    """Return ordered list of variables to process (always including hs)."""
    variables = ['hs']
    if conf_vars is not None:
        configured = [var for var in conf_vars.keys() if var != 'hs']
        variables.extend(configured)

        # Auto-add directional components for direction-like variables.
        # Example: mwd -> mwd_sin, mwd_cos
        for var in configured:
            var_conf = conf_vars[var]
            units = str(getattr(var_conf, 'units', '')).lower()
            long_name = str(getattr(var_conf, 'long_name', '')).lower()
            col_name = str(getattr(var_conf, 'column', '')).lower()
            is_direction = (
                'degree' in units
                or 'direction' in long_name
                or 'dir' in var.lower()
                or 'dir' in col_name
            )
            if is_direction:
                variables.append(f"{var}_sin")
                variables.append(f"{var}_cos")
    return variables


def _input_sources(conf_buoy):
    """Normalize input sources configuration to a list."""
    sources = getattr(conf_buoy.input, 'sources', None)
    if sources is None:
        return [getattr(conf_buoy.input, 'source', 'csv')]
    if isinstance(sources, str):
        return [sources]
    return list(sources)


def _final_var_file(outdir, date, variable):
    return os.path.join(outdir, f"{date}_buoy_series_{variable}.nc")


def _read_source_dataframe(source, conf_buoy, start_date, end_date):
    """Read a single source and return dataframe + standardized column mapping."""
    if source == 'copernicusmarine':
        df = read_from_copernicusmarine(conf_buoy, start_date, end_date)
        cols = SimpleNamespace(
            time='time', lon='longitude', lat='latitude', hs='VHM0', station='platform_id', qc='QC'
        )
        return df, cols

    if source == 'ispra_folders':
        df = read_from_ispra_folders(conf_buoy, start_date, end_date)
        cols = SimpleNamespace(
            time='time', lon='longitude', lat='latitude', hs='VHM0', station='platform_id', qc='QC'
        )
        return df, cols

    if source == 'csv':
        print(f"Reading buoy data from CSV: {conf_buoy.input.path}")
        df = pd.read_csv(conf_buoy.input.path)
        return df, conf_buoy.columns

    raise ValueError(f"Unknown source '{source}'")


def _collect_existing_station_files(outdir, source, date):
    """Collect already-processed per-station variable files for a source/date."""
    existing_source_dirs = [
        d for d in os.listdir(outdir)
        if os.path.isdir(os.path.join(outdir, d)) and d.startswith(f"{source}_")
    ]

    if not existing_source_dirs:
        return []

    source_station_files = []
    for station_dir_name in existing_source_dirs:
        station_dir = os.path.join(outdir, station_dir_name)
        var_files = glob(os.path.join(station_dir, f"{station_dir_name}_{date}_*.nc"))
        source_station_files.extend(var_files)

    return source_station_files


def build_obs_dataset(df, model_names, conf_cols, conf_vars=None):
    """Build xarray dataset from DataFrame with multiple variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with buoy observations
    model_names : list
        List of model names
    conf_cols : object
        Configuration object with column mappings
    conf_vars : object, optional
        Configuration object with additional variables to include
        
    Returns
    -------
    xr.Dataset
        Dataset with observations and model prediction placeholders
    """
    df = df.sort_values(conf_cols.time, kind="mergesort").reset_index(drop=True)
    nobs = len(df)

    data_vars = dict(
        time=("obs", df[conf_cols.time].to_numpy(dtype="datetime64[ns]")),
        longitude=("obs", df[conf_cols.lon].to_numpy(dtype=np.float64)),
        latitude=("obs", df[conf_cols.lat].to_numpy(dtype=np.float64)),
        hs=("obs", df[conf_cols.hs].to_numpy(dtype=np.float32)),
        station=("obs", df[conf_cols.station].astype(str).to_numpy(dtype=object)),
    )
    
    # Add additional variables if configured
    if conf_vars is not None:
        for var_name, var_conf in conf_vars.items():
            if var_name == 'hs':  # Already added above
                continue
            col_name = var_conf.column
            if col_name in df.columns:
                data_vars[var_name] = ("obs", df[col_name].to_numpy(dtype=np.float32))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            obs=np.arange(nobs),
            model=np.array(model_names, dtype=object),
        ),
    )

    # Add attributes
    ds["hs"].attrs["units"] = "m"
    if conf_vars is not None:
        for var_name, var_conf in conf_vars.items():
            if var_name in ds.data_vars:
                if hasattr(var_conf, 'units'):
                    ds[var_name].attrs["units"] = var_conf.units
                if hasattr(var_conf, 'long_name'):
                    ds[var_name].attrs["long_name"] = var_conf.long_name
    
    return ds


def read_from_copernicusmarine(conf_buoy, start_date, end_date):
    """
    Read buoy data from Copernicus Marine Service API with optional caching.
    
    Parameters
    ----------
    conf_buoy : object
        Configuration object for buoy preprocessing
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
        
    Returns
    -------
    pd.DataFrame
        Buoy data as pandas DataFrame
    """
    try:
        import copernicusmarine
    except ImportError:
        raise ImportError("copernicusmarine package not installed. Install with: pip install copernicusmarine")
    
    cm_conf = conf_buoy.input.copernicusmarine
    
    # Convert date format YYYYMMDD -> YYYY-MM-DDTHH:MM:SS
    start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}T00:00:00"
    end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}T23:59:59"
    
    # Get bounding box from filters
    lon_min, lon_max = conf_buoy.filters.valid_lon_range
    lat_min, lat_max = conf_buoy.filters.valid_lat_range
    
    # Check for cache file (optional)
    cache_enabled = getattr(cm_conf, 'cache_enabled', False)
    cache_dir = getattr(cm_conf, 'cache_dir', None)
    
    cache_file = None
    if cache_enabled and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"copernicusmarine_{cm_conf.dataset_id}_{start_date}_{end_date}.csv"
        )
        
        if os.path.exists(cache_file):
            print(f"Loading cached Copernicus Marine data...")
            print(f"  Cache file: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['time'])
            print(f"  Loaded {len(df)} observations from cache")
            return df
    
    print("Fetching data from Copernicus Marine Service...")
    print(f"  Dataset: {cm_conf.dataset_id}")
    print(f"  Time range: {start_dt} to {end_dt}")
    print(f"  Region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
    
    df = copernicusmarine.read_dataframe(
        dataset_id=cm_conf.dataset_id,
        dataset_part=cm_conf.dataset_part,
        variables=cm_conf.variables,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_dt,
        end_datetime=end_dt,
        minimum_depth=cm_conf.minimum_depth,
        maximum_depth=cm_conf.maximum_depth,
    )
    
    print(f"Retrieved {len(df)} observations from Copernicus Marine Service")
    
    # Check if data is in long format (variable names in 'variable' column)
    if 'variable' in df.columns and 'value' in df.columns:
        print("  Detected long format data, pivoting to wide format...")
        
        # Pivot: each variable becomes a column
        df_wide = df.pivot_table(
            index=['platform_id', 'time', 'longitude', 'latitude'],
            columns='variable',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Also preserve QC columns if they exist
        if 'value_qc' in df.columns:
            df_qc = df.pivot_table(
                index=['platform_id', 'time', 'longitude', 'latitude'],
                columns='variable',
                values='value_qc',
                aggfunc='first'
            ).reset_index()
            
            # Add QC columns with _QC suffix
            for col in df_qc.columns:
                if col not in ['platform_id', 'time', 'longitude', 'latitude']:
                    df_wide[f'{col}_QC'] = df_qc[col]
        
        df = df_wide
        print(f"  Pivoted to wide format with columns: {list(df.columns)}")
    
    # Save to cache if enabled
    if cache_file:
        df.to_csv(cache_file, index=False)
        print(f"  Cached to: {cache_file}")
    
    return df


def read_from_ispra_folders(conf_buoy, start_date, end_date):
    """
    Read buoy data from ISPRA folder structure (station folders with monthly CSV files).
    
    Parameters
    ----------
    conf_buoy : object
        Configuration object for buoy preprocessing
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
        
    Returns
    -------
    pd.DataFrame
        Buoy data as pandas DataFrame with standardized columns
    """
    ispra_conf = conf_buoy.input.ispra_folders
    base_path = ispra_conf.base_path
    
    # Convert date format
    start_dt = pd.to_datetime(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
    end_dt = pd.to_datetime(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}T23:59:59")
    
    print("Reading data from ISPRA folder structure...")
    print(f"  Base path: {base_path}")
    print(f"  Time range: {start_dt} to {end_dt}")
    
    all_data = []
    station_positions = ispra_conf.station_positions
    
    # --- Process Nausicaa single file if configured ---
    if hasattr(ispra_conf, 'nausicaa') and ispra_conf.nausicaa.enabled:
        nausicaa_file = ispra_conf.nausicaa.file_path
        if os.path.exists(nausicaa_file):
            print(f"\n  Reading Nausicaa data: {nausicaa_file}")
            lat, lon = ispra_conf.nausicaa.position
            
            try:
                df_nausicaa = pd.read_csv(nausicaa_file, delimiter=ispra_conf.delimiter)
                
                # Parse time from separate columns (aaaa, mm, dd, HH, MM)
                df_nausicaa['time'] = pd.to_datetime(
                    df_nausicaa['aaaa'].astype(str) + '-' + 
                    df_nausicaa['mm'].astype(str).str.zfill(2) + '-' + 
                    df_nausicaa['dd'].astype(str).str.zfill(2) + ' ' + 
                    df_nausicaa['HH'].astype(str).str.zfill(2) + ':' + 
                    df_nausicaa['MM'].astype(str).str.zfill(2) + ':00',
                    errors='coerce'
                )
                
                # Standardize columns
                df_nausicaa['longitude'] = lon
                df_nausicaa['latitude'] = lat
                df_nausicaa['platform_id'] = 'nausicaa'
                df_nausicaa['VHM0'] = pd.to_numeric(df_nausicaa['SWH'], errors='coerce')
                
                # Add other variables
                if 'Dir' in df_nausicaa.columns:
                    df_nausicaa['VMDR'] = pd.to_numeric(df_nausicaa['Dir'], errors='coerce')              
                if 'Tm' in df_nausicaa.columns:
                    df_nausicaa['VTM10'] = pd.to_numeric(df_nausicaa['Tm'], errors='coerce')
                if 'Tp' in df_nausicaa.columns:
                    df_nausicaa['VTZA'] = pd.to_numeric(df_nausicaa['Tp'], errors='coerce')
                

                # Keep only needed columns
                keep_cols = ['time', 'longitude', 'latitude', 'platform_id', 'VHM0']
                if 'VMDR' in df_nausicaa.columns:
                    keep_cols.append('VMDR')
                if 'VTM10' in df_nausicaa.columns:
                    keep_cols.append('VTM10')
                if 'VTZA' in df_nausicaa.columns:
                    keep_cols.append('VTZA')
                
                df_nausicaa = df_nausicaa[keep_cols]
                
                # Filter by date range
                df_nausicaa = df_nausicaa[(df_nausicaa['time'] >= start_dt) & (df_nausicaa['time'] <= end_dt)]
                
                if len(df_nausicaa) > 0:
                    all_data.append(df_nausicaa)
                    print(f"    Kept {len(df_nausicaa)} Nausicaa observations in date range")
                
            except Exception as e:
                print(f"    Warning: Error reading Nausicaa file: {e}")
        else:
            print(f"    Warning: Nausicaa file not found: {nausicaa_file}")
    
    # --- Iterate through station folders ---
    for station_name in os.listdir(base_path):
        station_path = os.path.join(base_path, station_name)
        if not os.path.isdir(station_path):
            continue
        
        # Get station position
        if station_name not in station_positions:
            print(f"  Warning: No position for station {station_name}, skipping")
            continue
        
        lat, lon = station_positions[station_name]
        
        # Find all CSV files for this station
        file_pattern = ispra_conf.file_pattern.format(station=station_name)
        csv_files = sorted(glob(os.path.join(station_path, file_pattern)))
        
        if len(csv_files) == 0:
            continue
        
        print(f"  Reading station {station_name}: {len(csv_files)} files")
        
        station_data = []
        for csv_file in csv_files:
            try:
                df_file = pd.read_csv(csv_file, delimiter=ispra_conf.delimiter)
                
                # Handle different time formats
                if ispra_conf.columns.time in df_file.columns:
                    # Standard format with UTC column
                    df_file['time'] = pd.to_datetime(df_file[ispra_conf.columns.time], 
                                                      format=ispra_conf.time_format, 
                                                      errors='coerce')
                elif all(col in df_file.columns for col in ['aaaa', 'mm', 'dd', 'HH', 'MM']):
                    # Nausicaa format with separate date/time columns
                    df_file['time'] = pd.to_datetime(
                        df_file['aaaa'].astype(str) + '-' + 
                        df_file['mm'].astype(str).str.zfill(2) + '-' + 
                        df_file['dd'].astype(str).str.zfill(2) + ' ' + 
                        df_file['HH'].astype(str).str.zfill(2) + ':' + 
                        df_file['MM'].astype(str).str.zfill(2) + ':00',
                        errors='coerce'
                    )
                else:
                    print(f"    Warning: Cannot parse time from {csv_file}")
                    continue
                
                # Rename columns to standard format
                df_file['longitude'] = lon
                df_file['latitude'] = lat
                df_file['platform_id'] = station_name
                
                # Map wave height column
                if ispra_conf.columns.hs in df_file.columns:
                    df_file['VHM0'] = pd.to_numeric(df_file[ispra_conf.columns.hs], errors='coerce')
                elif 'SWH' in df_file.columns:
                    df_file['VHM0'] = pd.to_numeric(df_file['SWH'], errors='coerce')
                
                # Add standard derived columns
                if 'Dir' in df_file.columns:
                    df_file['VMDR'] = pd.to_numeric(df_file['Dir'], errors='coerce')
                if 'Tm' in df_file.columns:
                    df_file['VTM10'] = pd.to_numeric(df_file['Tm'], errors='coerce')
                if 'Tp' in df_file.columns:
                    df_file['VTZA'] = pd.to_numeric(df_file['Tp'], errors='coerce')


                # Keep only needed columns
                keep_cols = ['time', 'longitude', 'latitude', 'platform_id', 'VHM0']
                if 'VMDR' in df_file.columns:
                    keep_cols.append('VMDR')
                if 'VTM10' in df_file.columns:
                    keep_cols.append('VTM10')
                if 'VTZA' in df_file.columns:
                    keep_cols.append('VTZA')
                
                df_file = df_file[keep_cols]
                station_data.append(df_file)
                
            except Exception as e:
                print(f"    Error reading {csv_file}: {e}")
                continue
        
        if len(station_data) > 0:
            station_df = pd.concat(station_data, ignore_index=True)
            # Filter by date range
            station_df = station_df[(station_df['time'] >= start_dt) & (station_df['time'] <= end_dt)]
            if len(station_df) > 0:
                all_data.append(station_df)
                print(f"    Kept {len(station_df)} observations in date range")
    
    if len(all_data) == 0:
        raise RuntimeError("No ISPRA data found in specified date range")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(df)} observations from {df['platform_id'].nunique()} ISPRA stations (including Nausicaa if enabled)")
    return df


def submit(conf_path, start_date, end_date):
    """
    Process buoy data into satellite-compatible NetCDF format.
    
    Supports reading from multiple sources (CSV, Copernicus Marine, ISPRA folders) in one run.
    Creates per-station files and a merged series file compatible with model_preprocessing.py.
    Output format matches satellite preprocessing with 'obs' dimension and
    standardized variable names (longitude, latitude, time, hs, model_hs).
    
    Parameters
    ----------
    conf_path : str
        Path to configuration file
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
        
    Returns
    -------
    xr.Dataset
        Merged buoy dataset compatible with model_preprocessing
    """
    conf_buoy = getConfigurationByID(conf_path, "buoy_preproc")
    conf_model = getConfigurationByID(conf_path, "model_preproc")
    conf_vars = getattr(conf_buoy, 'variables', None)
    
    date = f"{start_date}_{end_date}"
    outdir = conf_model.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)

    model_names = list(conf_model.datasets.models.keys())
    if not model_names:
        raise RuntimeError("No models found in model_preproc config.")

    sources = _input_sources(conf_buoy)
    variables_to_process = _variables_to_process(conf_vars)


    # Determine which variables actually need processing (missing merged file)
    final_var_files = {var: _final_var_file(outdir, date, var) for var in variables_to_process}
    missing_vars = [var for var, path in final_var_files.items() if not os.path.exists(path)]

    if not missing_vars:
        print("\nAll final merged variable files already exist. Skipping re-creation.")
        for variable, path in final_var_files.items():
            print(f"  ✓ {variable}: {path}")
        return open_dataset_flexible(final_var_files[variables_to_process[0]])

    print(f"\nVariables to process (missing merged file): {missing_vars}")
    
    # Check if combined series file already exists and is recent
    # outname_series = os.path.join(outdir, f"{date}_buoy_series.nc")
    # if os.path.exists(outname_series):
    #     # Check if it's newer than the script modification time (heuristic)
    #     file_age_hours = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(outname_series), unit='s')).total_seconds() / 3600
    #     if file_age_hours < 1:  # Less than 1 hour old
    #         print(f"\nCombined series file is recent (< 1 hour old), using existing: {outname_series}")
    #         # Load into memory and close file handle
    #         ds = open_dataset_flexible(outname_series)
    #         ds_copy = ds.load()
    #         ds.close()
    #         return ds_copy
    
    print(f"\n{'='*60}")
    print(f"Processing {len(sources)} data source(s): {', '.join(sources)}")
    print(f"{'='*60}\n")
    
    all_station_files = []  # Track all station files from all sources

    # Loop over sources
    for source_idx, source in enumerate(sources):
        source = source.lower()
        print(f"\n{'='*60}")
        print(f"SOURCE {source_idx + 1}/{len(sources)}: {source.upper()}")
        print(f"{'='*60}\n")

        # Only collect station files for variables that are missing
        source_station_files = []
        existing_source_dirs = [
            d for d in os.listdir(outdir)
            if os.path.isdir(os.path.join(outdir, d)) and d.startswith(f"{source}_")
        ]
        for station_dir_name in existing_source_dirs:
            station_dir = os.path.join(outdir, station_dir_name)
            for var in missing_vars:
                var_files = glob(os.path.join(station_dir, f"{station_dir_name}_{date}_{var}.nc"))
                source_station_files.extend(var_files)

        # If all missing_vars are present for this source, skip reading source
        found_vars = set()
        for f in source_station_files:
            for var in missing_vars:
                if f.endswith(f"_{var}.nc"):
                    found_vars.add(var)
        if found_vars == set(missing_vars):
            print(f"Using existing station files for all missing variables, skipping source read")
            all_station_files.extend(source_station_files)
            continue

        try:
            df, cols = _read_source_dataframe(source, conf_buoy, start_date, end_date)
        except ValueError as exc:
            print(f"WARNING: {exc}, skipping")
            continue

        # Process this source, but only for missing_vars
        station_files = process_single_source(
            df, cols, conf_buoy, model_names, outdir, date, source, start_date, end_date,
            variables_override=missing_vars
        )
        all_station_files.extend(station_files)
    
    # --- Create combined series file from all sources ---
    if len(all_station_files) == 0:
        raise RuntimeError("No station files were created from any source.")
    

    # Create separate files for each variable (only for missing_vars)
    print(f"\n{'='*60}")
    print(f"Creating separate series files for each variable...")
    print(f"{'='*60}")
    print(f"Variables: {variables_to_process}")
    print(f"Merging {len(all_station_files)} station files from {len(sources)} source(s)...\n")

    output_format = getattr(conf_buoy, 'output_format', 'netcdf') if hasattr(conf_buoy, 'output_format') else 'netcdf'

    for var_idx, variable in enumerate(variables_to_process, 1):
        outname_var = _final_var_file(outdir, date, variable)

        # Only process if missing
        if os.path.exists(outname_var):
            print(f"  [{var_idx}/{len(variables_to_process)}] Using existing: {os.path.basename(outname_var)}")
            continue

        # Fallback: if processing hs and old all-variables file exists, use it
        if variable == 'hs':
            outname_fallback = os.path.join(outdir, f"{date}_buoy_series.nc")
            if os.path.exists(outname_fallback):
                print(f"  [{var_idx}/{len(variables_to_process)}] Found legacy all-variables file, extracting hs...")
                ds_legacy = open_dataset_flexible(outname_fallback)
                if 'hs' in ds_legacy.data_vars and 'model_hs' in ds_legacy.data_vars:
                    # Extract just hs and model_hs from legacy file
                    ds_hs_only = ds_legacy[['time', 'longitude', 'latitude', 'station', 'hs', 'model_hs']]
                    save_dataset_flexible(ds_hs_only, outname_var, format=output_format)
                    print(f"  [{var_idx}/{len(variables_to_process)}] Saved as: {os.path.basename(outname_var)}")
                    continue

        print(f"  [{var_idx}/{len(variables_to_process)}] Creating {os.path.basename(outname_var)}")

        # Filter to only station files for this variable
        variable_station_files = [f for f in all_station_files if f.endswith(f"_{variable}.nc")]

        if len(variable_station_files) == 0:
            print(f"    ✗ No station files found for variable '{variable}'")
            continue

        # Open and concatenate station files for this variable
        ds_list = []
        for f in variable_station_files:
            ds = open_dataset_flexible(f)

            # Verify all required variables exist
            required = ['time', 'longitude', 'latitude', 'station', variable]
            missing = [v for v in required if v not in ds.data_vars and v not in ds.coords]

            if missing:
                print(f"    Warning: Missing {missing} in {os.path.basename(f)}, skipping")
                continue

            # Keep only required variables and drop obs coordinate
            ds_subset = ds[required].drop_vars('obs', errors='ignore')
            ds_list.append(ds_subset)

        if len(ds_list) > 0:
            # Concatenate all datasets along obs dimension (will create a new obs dim automatically)
            ds_var = xr.concat(ds_list, dim='obs')

            # Replace obs coordinate with clean int64 array
            obs_coord = np.arange(len(ds_var.obs), dtype=np.int64)
            ds_var = ds_var.assign_coords(obs=obs_coord)

            # Fix station variable: ensure it's string dtype, not object with mixed types
            if 'station' in ds_var.data_vars:
                station_values = ds_var['station'].values
                # Convert to string array
                ds_var['station'] = ('obs', np.asarray(station_values, dtype=str))

            # Clear all encoding to avoid rint issues during save
            for var in ds_var.data_vars:
                ds_var[var].encoding = {}
            for coord in ds_var.coords:
                ds_var[coord].encoding = {}

            save_dataset_flexible(ds_var, outname_var, format=output_format)
            print(f"      ✓ Saved: {len(ds_var.obs)} observations")
        else:
            print(f"    ✗ ERROR: Could not create file for variable '{variable}'")
    
    print(f"\n{'='*60}")
    print(f"✓ Separate files created for each variable:")
    for variable in variables_to_process:
        outname_var = _final_var_file(outdir, date, variable)
        if os.path.exists(outname_var):
            print(f"  ✓ {variable}: {outname_var}")
    print(f"{'='*60}\n")
    
    # Return first variable file
    first_var = variables_to_process[0]
    outname_var = _final_var_file(outdir, date, first_var)
    return open_dataset_flexible(outname_var)


def process_single_source(df, cols, conf_buoy, model_names, outdir, date, source, start_date, end_date, variables_override=None):
    """
    Process a single data source and create per-station files.
    
    Creates separate files for each variable to avoid reprocessing when adding new variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data from source
    cols : object
        Column mapping configuration
    conf_buoy : object
        Buoy preprocessing configuration
    model_names : list
        List of model names
    outdir : str
        Output directory
    date : str
        Date string for output filenames
    source : str
        Source identifier (csv, copernicusmarine, ispra_folders)
    start_date : str
        Start date in YYYYMMDD format
    end_date : str
        End date in YYYYMMDD format
        
    Returns
    -------
    list
        List of paths to created station files (returns all variable files for each station)
    """
    conf_vars = getattr(conf_buoy, 'variables', None)
    variables_to_process = _variables_to_process(conf_vars)
    if variables_override is not None:
        # Only process a subset of variables (e.g., missing merged files)
        variables_to_process = [v for v in variables_to_process if v in variables_override]

    # --- Check if data is in long format and pivot to wide format ---
    if 'variable' in df.columns and 'value' in df.columns:
        print(f"  Detected long format data, pivoting to wide format...")
        
        # Identify index columns (all except 'variable', 'value', 'value_qc')
        index_cols = [col for col in df.columns if col not in ['variable', 'value', 'value_qc']]
        
        # Pivot: each variable becomes a column
        df_wide = df.pivot_table(
            index=index_cols,
            columns='variable',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Also preserve QC columns if they exist
        if 'value_qc' in df.columns:
            df_qc = df.pivot_table(
                index=index_cols,
                columns='variable',
                values='value_qc',
                aggfunc='first'
            ).reset_index()
            
            # Add QC columns with _QC suffix
            for col in df_qc.columns:
                if col not in index_cols:
                    df_wide[f'{col}_QC'] = df_qc[col]
        
        df = df_wide
        print(f"  Pivoted to wide format with columns: {list(df.columns)}")

    # --- time parsing (if not already datetime) ---
    if not pd.api.types.is_datetime64_any_dtype(df[cols.time]):
        if hasattr(conf_buoy.time, 'format') and conf_buoy.time.format:
            df[cols.time] = pd.to_datetime(
                df[cols.time],
                format=conf_buoy.time.format,
                errors="coerce",
                utc=conf_buoy.time.utc if hasattr(conf_buoy.time, 'utc') else False,
            )
        else:
            df[cols.time] = pd.to_datetime(
                df[cols.time],
                errors="coerce",
                utc=conf_buoy.time.utc if hasattr(conf_buoy.time, 'utc') else False,
            )
    
    # --- timezone handling: convert to UTC naive ---
    # Check if datetime is timezone-aware
    if df[cols.time].dt.tz is not None:
        print(f"Detected timezone-aware datetime: {df[cols.time].dt.tz}")
        # Convert to UTC if not already
        if str(df[cols.time].dt.tz) != 'UTC':
            print(f"Converting from {df[cols.time].dt.tz} to UTC")
            df[cols.time] = df[cols.time].dt.tz_convert('UTC')
        # Remove timezone info (make naive)
        df[cols.time] = df[cols.time].dt.tz_localize(None)
        print("Converted to UTC naive datetime")
    else:
        print("Datetime is already timezone-naive (assuming UTC)")

    # --- Find wave height column (handle different variable names) ---
    hs_available = True
    try:
        preferred_hs = getattr(cols, 'hs', None)
        actual_hs_col = _find_wave_height_column(df, preferred_col=preferred_hs)
        if preferred_hs and actual_hs_col != preferred_hs:
            print(f"  Using '{actual_hs_col}' for wave height (configured: '{preferred_hs}')")
        cols.hs = actual_hs_col  # Store for later use (create if needed)
    except ValueError as e:
        hs_available = False
        cols.hs = None
        print(f"  WARNING: {e}")
        print("  hs column not found: processing non-hs variables only for this source")
        variables_to_process = [v for v in variables_to_process if v != 'hs']

    # --- numeric coercion ---
    df[cols.lon] = pd.to_numeric(df[cols.lon], errors="coerce")
    df[cols.lat] = pd.to_numeric(df[cols.lat], errors="coerce")
    if hs_available and cols.hs in df.columns:
        df[cols.hs] = pd.to_numeric(df[cols.hs], errors="coerce")
    
    # Coerce additional variables if configured
    if conf_vars is not None:
        for var_name, var_conf in conf_vars.items():
            if var_name == 'hs':
                continue  # Already coerced above
            col_name = var_conf.column
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # --- basic cleaning ---
    required_cols = [cols.station, cols.time, cols.lon, cols.lat]
    if conf_buoy.filters.drop_na:
        df = df.dropna(subset=required_cols)

    # --- geographic filters ---
    lon_min, lon_max = conf_buoy.filters.valid_lon_range
    lat_min, lat_max = conf_buoy.filters.valid_lat_range
    df = df[(df[cols.lon] >= lon_min) & (df[cols.lon] <= lon_max)]
    df = df[(df[cols.lat] >= lat_min) & (df[cols.lat] <= lat_max)]
    
    # --- Exclude specific regions (e.g., Gulf of Biscay) ---
    if hasattr(conf_buoy.filters, 'exclude_regions') and conf_buoy.filters.exclude_regions:
        for region in conf_buoy.filters.exclude_regions:
            region_name = region.get('name', 'unnamed')
            ex_lon_min, ex_lon_max = region['lon_range']
            ex_lat_min, ex_lat_max = region['lat_range']
            
            # Create mask for points INSIDE the exclusion region
            in_exclusion = (
                (df[cols.lon] >= ex_lon_min) & (df[cols.lon] <= ex_lon_max) &
                (df[cols.lat] >= ex_lat_min) & (df[cols.lat] <= ex_lat_max)
            )
            
            excluded_count = in_exclusion.sum()
            if excluded_count > 0:
                print(f"  Excluding {excluded_count} observations from '{region_name}' "
                      f"(lon: [{ex_lon_min}, {ex_lon_max}], lat: [{ex_lat_min}, {ex_lat_max}])")
                df = df[~in_exclusion]

    # --- QC filter (variable-specific if variables are configured) ---
    if conf_buoy.qc.enabled:
        allowed = set(conf_buoy.qc.allowed)
        
        if conf_vars is not None:
            # Apply QC per variable
            for var_name, var_conf in conf_vars.items():
                qc_col = var_conf.qc_column if hasattr(var_conf, 'qc_column') else None
                if qc_col and qc_col in df.columns:
                    df[qc_col] = pd.to_numeric(df[qc_col], errors="coerce")
                    # Set variable to NaN where QC fails (don't drop entire row)
                    bad_qc = ~df[qc_col].isin(allowed)
                    if bad_qc.any():
                        df.loc[bad_qc, var_conf.column] = np.nan
                        print(f"  Set {bad_qc.sum()} {var_name} values to NaN due to QC flags")
        elif hasattr(cols, 'qc') and cols.qc in df.columns:
            # Legacy: single QC column for all variables
            df[cols.qc] = pd.to_numeric(df[cols.qc], errors="coerce")
            df = df[df[cols.qc].isin(allowed)]

    # --- Time range filter ---
    start_dt = pd.to_datetime(f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}")
    end_dt = pd.to_datetime(f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}T23:59:59")
    df = df[(df[cols.time] >= start_dt) & (df[cols.time] <= end_dt)]

    if len(df) == 0:
        print(f"WARNING: No valid observations for source '{source}' after filtering.")
        return []
    
    print(f"Filtered data: {len(df)} observations from {df[cols.station].nunique()} stations")
    
    # Get minimum valid observations threshold (applies per station for entire period)
    min_valid_obs = getattr(conf_buoy.filters, 'min_valid_obs', 1)
    print(f"Minimum valid observations per station: {min_valid_obs}")
    
    # --- Loop over stations and create per-station time series files ---
    processed_stations = []
    skipped_stations = []
    
    for station_id, station_df in df.groupby(cols.station):
        station_name = _safe_station_name(station_id)
        nobs = len(station_df)
        
        # Create per-station directory with source prefix
        station_dir_name = f"{source}_{station_name}"
        station_dir = os.path.join(outdir, station_dir_name)
        os.makedirs(station_dir, exist_ok=True)
        
        # Sort by time
        station_df = station_df.sort_values(cols.time).reset_index(drop=True)
        
        # Create separate files for each variable
        for var_idx, variable in enumerate(variables_to_process):
            outname_station_var = os.path.join(station_dir, f"{station_dir_name}_{date}_{variable}.nc")
            
            # Check if file already exists
            if os.path.exists(outname_station_var):
                processed_stations.append(outname_station_var)
                continue
            
            # Build variable values on station rows; then keep only valid rows for this variable
            var_values = None
            if variable == 'hs':
                if hs_available and cols.hs in station_df.columns:
                    var_values = pd.to_numeric(station_df[cols.hs], errors='coerce').to_numpy(dtype=np.float32)
            elif conf_vars is not None and variable in conf_vars:
                col_name = conf_vars[variable].column
                if col_name in station_df.columns:
                    var_values = pd.to_numeric(station_df[col_name], errors='coerce').to_numpy(dtype=np.float32)
            elif variable.endswith('_sin') or variable.endswith('_cos'):
                base_var, component = variable.rsplit('_', 1)
                if conf_vars is not None and base_var in conf_vars:
                    base_col = conf_vars[base_var].column
                    if base_col in station_df.columns:
                        ang_deg = pd.to_numeric(station_df[base_col], errors='coerce').to_numpy(dtype=np.float32)
                        ang_rad = np.deg2rad(ang_deg.astype(np.float64))
                        trig_values = np.sin(ang_rad) if component == 'sin' else np.cos(ang_rad)
                        var_values = trig_values.astype(np.float32)

            # Skip if variable unavailable for this station
            if var_values is None:
                continue

            valid_mask = np.isfinite(var_values)
            nobs_var = int(valid_mask.sum())
            if nobs_var < min_valid_obs:
                continue

            station_var_df = station_df.loc[valid_mask].reset_index(drop=True)
            var_values_valid = var_values[valid_mask]

            data_vars = {
                'time': ("obs", station_var_df[cols.time].to_numpy(dtype="datetime64[ns]")),
                'longitude': ("obs", station_var_df[cols.lon].to_numpy(dtype=np.float64)),
                'latitude': ("obs", station_var_df[cols.lat].to_numpy(dtype=np.float64)),
                'station': ("obs", np.full(nobs_var, station_id, dtype=object)),
                variable: ("obs", var_values_valid.astype(np.float32)),
            }
            
            ds_station = xr.Dataset(
                data_vars=data_vars,
                coords=dict(
                    obs=np.arange(nobs_var),
                ),
            )
            
            # Add attributes
            if variable == 'hs' and 'hs' in ds_station.data_vars:
                ds_station["hs"].attrs["units"] = "m"
            elif variable in conf_vars and variable in ds_station.data_vars:
                var_conf = conf_vars[variable]
                if hasattr(var_conf, 'units'):
                    ds_station[variable].attrs["units"] = var_conf.units
                if hasattr(var_conf, 'long_name'):
                    ds_station[variable].attrs["long_name"] = var_conf.long_name
            elif (variable.endswith('_sin') or variable.endswith('_cos')) and variable in ds_station.data_vars:
                base_var, component = variable.rsplit('_', 1)
                ds_station[variable].attrs["units"] = "1"
                ds_station[variable].attrs["long_name"] = f"{base_var} {component} component"
            
            ds_station.attrs["station_id"] = station_id
            ds_station.attrs["n_observations"] = nobs_var
            ds_station.attrs["data_source"] = source
            
            ds_station["time"].values = ds_station["time"].dt.round(freq="H")
            ds_station.to_netcdf(outname_station_var)
            processed_stations.append(outname_station_var)
    
    # Report summary for this source
    print(f"\n=== Source '{source}' Processing Summary ===")
    print(f"Processed {len(processed_stations)} stations")
    if len(skipped_stations) > 0:
        print(f"Skipped {len(skipped_stations)} stations with < {min_valid_obs} observations:")
        for station_id, nobs in skipped_stations[:10]:  # Show first 10
            print(f"  - {station_id}: {nobs} obs")
        if len(skipped_stations) > 10:
            print(f"  ... and {len(skipped_stations) - 10} more")
    
    return processed_stations
