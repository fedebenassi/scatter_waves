import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from natsort import natsorted

from utils import getConfigurationByID, daysBetweenDates


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
    wave_height_cols = ['VHM0', 'VAVH', 'VGHS', 'SWH', 'Hm0', 'hs', 'swh', 'wave_height']
    
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
    return s[:120]


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
        lon=("obs", df[conf_cols.lon].to_numpy(dtype=np.float64)),
        lat=("obs", df[conf_cols.lat].to_numpy(dtype=np.float64)),
        hs=("obs", df[conf_cols.hs].to_numpy(dtype=np.float32)),
        station=("obs", df[conf_cols.station].astype(str).to_numpy(dtype=object)),
        model_hs=(("obs", "model"),
                  np.full((nobs, len(model_names)), np.nan, dtype=np.float32)),
    )
    
    # Add additional variables if configured
    if conf_vars is not None:
        for var_name, var_conf in conf_vars.items():
            if var_name == 'hs':  # Already added above
                continue
            col_name = var_conf.column
            if col_name in df.columns:
                data_vars[var_name] = ("obs", df[col_name].to_numpy(dtype=np.float32))
                # Add placeholder for model predictions
                data_vars[f"model_{var_name}"] = (("obs", "model"),
                                                     np.full((nobs, len(model_names)), np.nan, dtype=np.float32))

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
    Read buoy data from Copernicus Marine Service API.
    
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
                
                # Add other variables if present
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
    
    date = f"{start_date}_{end_date}"
    outdir = conf_model.out_dir.out_dir
    os.makedirs(outdir, exist_ok=True)

    model_names = list(conf_model.datasets.models.keys())
    if not model_names:
        raise RuntimeError("No models found in model_preproc config.")

    # Get list of sources to process (support both 'sources' list and legacy 'source' string)
    sources = getattr(conf_buoy.input, 'sources', None)
    if sources is None:
        # Fallback to single source for backward compatibility
        sources = [getattr(conf_buoy.input, 'source', 'csv')]
    elif isinstance(sources, str):
        sources = [sources]
    
    # Check if combined series file already exists and is recent
    outname_series = os.path.join(outdir, f"{date}_buoy_series.nc")
    if os.path.exists(outname_series):
        # Check if it's newer than the script modification time (heuristic)
        file_age_hours = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(outname_series), unit='s')).total_seconds() / 3600
        if file_age_hours < 1:  # Less than 1 hour old
            print(f"\nCombined series file is recent (< 1 hour old), using existing: {outname_series}")
            # Load into memory and close file handle
            ds = xr.open_dataset(outname_series)
            ds_copy = ds.load()
            ds.close()
            return ds_copy
    
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
        
        # Quick check: see if this source has already been processed
        # Look for any existing station directories with this source prefix
        existing_source_dirs = [d for d in os.listdir(outdir) 
                               if os.path.isdir(os.path.join(outdir, d)) and d.startswith(f"{source}_")]
        
        if len(existing_source_dirs) > 0:
            print(f"Found {len(existing_source_dirs)} existing station directories for source '{source}'")
            # Collect existing station files
            for station_dir_name in existing_source_dirs:
                station_file = os.path.join(outdir, station_dir_name, f"{station_dir_name}_{date}.nc")
                if os.path.exists(station_file):
                    all_station_files.append(station_file)
            
            if len(all_station_files) > 0:
                print(f"Using {len(all_station_files)} existing station files, skipping data download")
                continue
        
        # Read data from source (only if needed)
        if source == 'copernicusmarine':
            df = read_from_copernicusmarine(conf_buoy, start_date, end_date)
            cols = conf_buoy.columns
        elif source == 'ispra_folders':
            df = read_from_ispra_folders(conf_buoy, start_date, end_date)
            # ISPRA data already has standardized column names
            # Create a temporary config object for column mapping
            class ColsISPRA:
                time = 'time'
                lon = 'longitude'
                lat = 'latitude'
                hs = 'VHM0'
                station = 'platform_id'
                qc = 'QC'
            cols = ColsISPRA()
        elif source == 'csv':
            print(f"Reading buoy data from CSV: {conf_buoy.input.path}")
            df = pd.read_csv(conf_buoy.input.path)
            cols = conf_buoy.columns
        else:
            print(f"WARNING: Unknown source '{source}', skipping")
            continue
        
        # Process this source
        station_files = process_single_source(
            df, cols, conf_buoy, model_names, outdir, date, source, start_date, end_date
        )
        all_station_files.extend(station_files)
    
    # --- Create combined series file from all sources ---
    if len(all_station_files) == 0:
        raise RuntimeError("No station files were created from any source.")
    
    outname_series = os.path.join(outdir, f"{date}_buoy_series.nc")
    
    if not os.path.exists(outname_series):
        print(f"\n{'='*60}")
        print(f"Creating combined series file from all sources...")
        print(f"{'='*60}")
        print(f"Merging {len(all_station_files)} station files from {len(sources)} source(s)...")
        
        # Open and concatenate all station files
        ds_merged = xr.open_dataset(all_station_files[0])
        obs = np.arange(len(ds_merged.obs.values))
        
        for f in all_station_files[1:]:
            ds = xr.open_dataset(f)
            obs = np.arange(len(ds.obs.values)) + (obs[-1] + 1)
            ds = ds.assign_coords(obs=obs)
            ds_merged = xr.concat([ds_merged, ds], "obs")
        
        # Reset obs coordinate
        ds_merged = ds_merged.assign_coords(obs=np.arange(len(ds_merged.obs)))
        ds_merged.to_netcdf(outname_series)
        print(f"\nSaved combined buoy series file: {outname_series}")
        print(f"Total observations: {len(ds_merged.obs)} from {len(all_station_files)} stations")
    else:
        # Load into memory and close file handle
        ds = xr.open_dataset(outname_series)
        ds_merged = ds.load()
        ds.close()
        print(f"Using existing combined series file: {outname_series}")

    return ds_merged


def process_single_source(df, cols, conf_buoy, model_names, outdir, date, source, start_date, end_date):
    """
    Process a single data source and create per-station files.
    
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
        List of paths to created station files
    """
    conf_vars = getattr(conf_buoy, 'variables', None)

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
    try:
        actual_hs_col = _find_wave_height_column(df, preferred_col=cols.hs if hasattr(cols, 'hs') else None)
        if actual_hs_col != cols.hs:
            print(f"  Using '{actual_hs_col}' for wave height (configured: '{cols.hs}')")
            # Update cols reference
            cols.hs = actual_hs_col
    except ValueError as e:
        print(f"  ERROR: {e}")
        return []

    # --- numeric coercion ---
    df[cols.lon] = pd.to_numeric(df[cols.lon], errors="coerce")
    df[cols.lat] = pd.to_numeric(df[cols.lat], errors="coerce")
    df[cols.hs] = pd.to_numeric(df[cols.hs], errors="coerce")
    
    # Coerce additional variables if configured
    if conf_vars is not None:
        for var_name, var_conf in conf_vars.items():
            col_name = var_conf.column
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # --- basic cleaning ---
    required_cols = [cols.station, cols.time, cols.lon, cols.lat, cols.hs]
    if conf_buoy.filters.drop_na:
        df = df.dropna(subset=required_cols)

    # --- geographic filters ---
    lon_min, lon_max = conf_buoy.filters.valid_lon_range
    lat_min, lat_max = conf_buoy.filters.valid_lat_range
    df = df[(df[cols.lon] >= lon_min) & (df[cols.lon] <= lon_max)]
    df = df[(df[cols.lat] >= lat_min) & (df[cols.lat] <= lat_max)]

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
        
        # Check minimum valid observations threshold
        if nobs < min_valid_obs:
            skipped_stations.append((station_id, nobs))
            continue
        
        # Create per-station directory with source prefix
        station_dir_name = f"{source}_{station_name}"
        station_dir = os.path.join(outdir, station_dir_name)
        outname_station = os.path.join(station_dir, f"{station_dir_name}_{date}.nc")
        
        # Check if file already exists before processing
        if os.path.exists(outname_station):
            processed_stations.append(outname_station)
            continue
        
        # Only process if file doesn't exist
        os.makedirs(station_dir, exist_ok=True)
        
        # Sort by time
        station_df = station_df.sort_values(cols.time).reset_index(drop=True)
        
        if True:  # Keep indentation level for the dataset creation code below
            # Create dataset structure with additional variables
            data_vars = dict(
                time=("obs", station_df[cols.time].to_numpy(dtype="datetime64[ns]")),
                longitude=("obs", station_df[cols.lon].to_numpy(dtype=np.float64)),
                latitude=("obs", station_df[cols.lat].to_numpy(dtype=np.float64)),
                hs=("obs", station_df[cols.hs].to_numpy(dtype=np.float32)),
                station=("obs", np.full(nobs, station_id, dtype=object)),
                model_hs=(("obs", "model"),
                          np.full((nobs, len(model_names)), np.nan, dtype=np.float32)),
            )
            
            # Add additional variables if configured
            if conf_vars is not None:
                for var_name, var_conf in conf_vars.items():
                    if var_name == 'hs':  # Already added
                        continue
                    col_name = var_conf.column
                    if col_name in station_df.columns:
                        data_vars[var_name] = ("obs", station_df[col_name].to_numpy(dtype=np.float32))
                        data_vars[f"model_{var_name}"] = (("obs", "model"),
                                                            np.full((nobs, len(model_names)), np.nan, dtype=np.float32))
            
            ds_station = xr.Dataset(
                data_vars=data_vars,
                coords=dict(
                    obs=np.arange(nobs),
                    model=np.array(model_names, dtype=object),
                ),
            )
            
            # Add attributes
            ds_station["hs"].attrs["units"] = "m"
            ds_station.attrs["station_id"] = station_id
            ds_station.attrs["n_observations"] = nobs
            ds_station.attrs["data_source"] = source
            
            if conf_vars is not None:
                for var_name, var_conf in conf_vars.items():
                    if var_name in ds_station.data_vars:
                        if hasattr(var_conf, 'units'):
                            ds_station[var_name].attrs["units"] = var_conf.units
                        if hasattr(var_conf, 'long_name'):
                            ds_station[var_name].attrs["long_name"] = var_conf.long_name
            
            ds_station["time"].values = ds_station["time"].dt.round(freq="H")
            ds_station.to_netcdf(outname_station)
        
        processed_stations.append(outname_station)
    
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
