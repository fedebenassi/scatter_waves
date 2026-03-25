"""
I/O utilities for handling both NetCDF and Zarr format files.

This module provides flexible dataset opening functions that automatically
detect and handle different file formats (NetCDF, Zarr).
"""

import os
import xarray as xr
from typing import Union, List


def detect_format(filepath: str) -> str:
    """
    Detect file format based on path/extension.
    
    Parameters
    ----------
    filepath : str
        Path to the file or directory
        
    Returns
    -------
    str
        'zarr' or 'netcdf'
    """
    if filepath.endswith('.zarr') or (os.path.isdir(filepath) and '.zarr' in filepath):
        return 'zarr'
    else:
        return 'netcdf'


def open_dataset_flexible(filepath: str, **kwargs) -> xr.Dataset:
    """
    Open a dataset, automatically detecting format (NetCDF or Zarr).
    
    Parameters
    ----------
    filepath : str
        Path to NetCDF file or Zarr store
    **kwargs
        Additional arguments passed to xr.open_dataset or xr.open_zarr
        
    Returns
    -------
    xr.Dataset
        Opened dataset
        
    Examples
    --------
    >>> ds = open_dataset_flexible('model_output.nc')
    >>> ds = open_dataset_flexible('model_output.zarr')
    """
    fmt = detect_format(filepath)
    
    if fmt == 'zarr':
        return xr.open_zarr(filepath, **kwargs)
    else:
        return xr.open_dataset(filepath, **kwargs)


def open_mfdataset_flexible(paths: Union[str, List[str]], **kwargs) -> xr.Dataset:
    """
    Open multiple datasets, automatically detecting format.
    
    Parameters
    ----------
    paths : str or list of str
        Paths to NetCDF files or Zarr stores (can be glob pattern or list)
    **kwargs
        Additional arguments passed to xr.open_mfdataset or concatenation
        
    Returns
    -------
    xr.Dataset
        Combined dataset
        
    Examples
    --------
    >>> ds = open_mfdataset_flexible('model_*.nc')
    >>> ds = open_mfdataset_flexible(['day1.zarr', 'day2.zarr'])
    """
    # Handle single path string (potentially with wildcards)
    if isinstance(paths, str):
        from glob import glob
        paths = sorted(glob(paths))
    
    if not paths:
        raise ValueError("No files found matching the pattern")
    
    # Check format of first file
    fmt = detect_format(paths[0])
    
    if fmt == 'zarr':
        # Open and concatenate multiple Zarr stores
        datasets = [xr.open_zarr(p, **kwargs) for p in paths]
        # Concatenate along time dimension
        concat_dim = kwargs.pop('concat_dim', 'time')
        return xr.concat(datasets, dim=concat_dim)
    else:
        # Use xarray's built-in multi-file opener for NetCDF
        return xr.open_mfdataset(paths, **kwargs)


def save_dataset_flexible(ds: xr.Dataset, filepath: str, format: str = 'netcdf', **kwargs):
    """
    Save a dataset in the specified format.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save
    filepath : str
        Output path
    format : str, optional
        Output format: 'netcdf' (default) or 'zarr'
    **kwargs
        Additional arguments passed to to_netcdf or to_zarr
        
    Examples
    --------
    >>> save_dataset_flexible(ds, 'output.nc')
    >>> save_dataset_flexible(ds, 'output.zarr', format='zarr')
    """
    if format == 'zarr':
        ds.to_zarr(filepath, **kwargs)
    else:
        ds.to_netcdf(filepath, **kwargs)
