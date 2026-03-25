"""
Statistical Metrics Module for Wave Model Validation

This module provides comprehensive statistical metrics for validating wave model outputs
against observational data (e.g., satellite measurements). It includes both standard
error metrics and advanced quantile-based statistics.

Author: Wave Model Validation Team
Date: 2024
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional


def BIAS(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate the mean bias (systematic error).
    
    BIAS = mean(model - observations)
    
    Positive values indicate model overestimation, negative values indicate underestimation.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        Mean bias rounded to 4 decimal places
        
    Examples
    --------
    >>> model = np.array([1.5, 2.0, 2.5])
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> BIAS(model, obs)
    0.0
    """
    return np.round((np.nanmean(data - obs)).data if hasattr(data - obs, 'data') 
                    else np.nanmean(data - obs), 4)


def RMSE(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    RMSE = sqrt(mean((model - observations)²))
    
    RMSE measures the standard deviation of the prediction errors.
    Lower values indicate better model performance.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        RMSE value rounded to 3 decimal places
        
    Examples
    --------
    >>> model = np.array([1.0, 2.0, 3.0])
    >>> obs = np.array([1.1, 1.9, 3.2])
    >>> RMSE(model, obs)
    0.141
    """
    return np.round(np.sqrt(np.nanmean((data - obs)**2)), 3)


def MAE(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = mean(|model - observations|)
    
    MAE is less sensitive to outliers than RMSE.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        MAE value rounded to 3 decimal places
    """
    return np.round(np.nanmean(np.abs(data - obs)), 3)


def ScatterIndex(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Scatter Index (normalized RMSE).
    
    SI = sqrt(sum(((model - mean(model)) - (obs - mean(obs)))²) / sum(obs²))
    
    The scatter index is a normalized measure of the variability of errors.
    Values close to 0 indicate better agreement.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        Scatter index rounded to 3 decimal places
        
    Notes
    -----
    The scatter index removes the bias effect and focuses on the scatter
    of the data points around the mean.
    """
    num = np.sum(((data - np.nanmean(data)) - (obs - np.nanmean(obs)))**2)
    denom = np.sum(obs**2)
    return np.round(np.sqrt((num / denom)), 3)


def correlation(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    R = cov(model, obs) / (std(model) * std(obs))
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        Correlation coefficient (-1 to 1) rounded to 4 decimal places
    """
    valid_mask = ~(np.isnan(data) | np.isnan(obs))
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.round(np.corrcoef(data[valid_mask], obs[valid_mask])[0, 1], 4)


def MADp(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Mean Absolute Difference of Percentiles.
    
    Computes the mean absolute difference between the percentile distributions
    of model data and observations.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        MADp metric value rounded to 4 decimal places
    """
    percentiles = np.arange(0, 1.01, 0.01)
    data_prc = np.nanquantile(data, percentiles)
    obs_prc = np.nanquantile(obs, percentiles)
    return np.round(np.nanmean(np.abs(data_prc - obs_prc)), 4)


def MADc(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Combined Mean Absolute Difference.
    
    Combines MADp (distribution difference) with mean absolute error.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        MADc metric value rounded to 4 decimal places
    """
    return np.round(MADp(data, obs) + np.nanmean(np.abs(data - obs)), 4)


def skill_score(data: np.ndarray, obs: np.ndarray, 
                reference: Optional[np.ndarray] = None) -> float:
    """
    Calculate Murphy's Skill Score.
    
    SS = 1 - (MSE_model / MSE_reference)
    
    If no reference is provided, uses the observation mean as reference (climatology).
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
    reference : np.ndarray, optional
        Reference forecast (default: observation mean)
        
    Returns
    -------
    float
        Skill score. Values > 0 indicate better than reference,
        < 0 indicate worse than reference
        
    Notes
    -----
    A perfect model has SS = 1, while SS = 0 means the model is no better
    than the reference (climatology).
    """
    mse_model = np.nanmean((data - obs)**2)
    if reference is None:
        reference = np.nanmean(obs)
    mse_reference = np.nanmean((reference - obs)**2)
    
    if mse_reference == 0:
        return np.nan
    
    return np.round(1 - (mse_model / mse_reference), 4)


def quantile_bias(data: np.ndarray, obs: np.ndarray, quantile: float = 0.9) -> float:
    """
    Calculate bias for a specific quantile.
    
    Useful for evaluating model performance at extreme values.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
    quantile : float, optional
        Quantile to evaluate (default: 0.9 for 90th percentile)
        
    Returns
    -------
    float
        Quantile bias rounded to 4 decimal places
        
    Examples
    --------
    >>> model = np.random.rand(1000) * 5
    >>> obs = np.random.rand(1000) * 5
    >>> quantile_bias(model, obs, quantile=0.95)  # 95th percentile bias
    """
    valid_mask = ~(np.isnan(data) | np.isnan(obs))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    data_q = np.nanquantile(data[valid_mask], quantile)
    obs_q = np.nanquantile(obs[valid_mask], quantile)
    
    return np.round(data_q - obs_q, 4)


def quantile_skill_score(data: np.ndarray, obs: np.ndarray, 
                         quantiles: list = [0.5, 0.75, 0.9, 0.95]) -> dict:
    """
    Calculate skill scores for multiple quantiles.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
    quantiles : list, optional
        List of quantiles to evaluate
        
    Returns
    -------
    dict
        Dictionary with quantile values as keys and metrics as values
        
    Examples
    --------
    >>> model = np.random.rand(1000) * 5
    >>> obs = np.random.rand(1000) * 5
    >>> qs = quantile_skill_score(model, obs)
    >>> print(qs[0.9])  # Metrics for 90th percentile
    """
    valid_mask = ~(np.isnan(data) | np.isnan(obs))
    data_valid = data[valid_mask]
    obs_valid = obs[valid_mask]
    
    results = {}
    for q in quantiles:
        # Select data above this quantile
        threshold = np.nanquantile(obs_valid, q)
        high_mask = obs_valid >= threshold
        
        if np.sum(high_mask) > 1:
            data_high = data_valid[high_mask]
            obs_high = obs_valid[high_mask]
            
            results[q] = {
                'bias': BIAS(data_high, obs_high),
                'rmse': RMSE(data_high, obs_high),
                'mae': MAE(data_high, obs_high),
                'correlation': correlation(data_high, obs_high),
                'count': np.sum(high_mask)
            }
        else:
            results[q] = {
                'bias': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'correlation': np.nan,
                'count': 0
            }
    
    return results


def symmetric_slope(data: np.ndarray, obs: np.ndarray) -> float:
    """
    Calculate symmetric slope (mean of forward and backward regression slopes).
    
    The symmetric slope is less sensitive to outliers than standard regression.
    
    Parameters
    ----------
    data : np.ndarray
        Model predictions
    obs : np.ndarray
        Observations (reference data)
        
    Returns
    -------
    float
        Symmetric slope value
        
    Notes
    -----
    A perfect model has a symmetric slope of 1.0
    """
    valid_mask = ~(np.isnan(data) | np.isnan(obs))
    if np.sum(valid_mask) < 2:
        return np.nan
    
    data_valid = data[valid_mask]
    obs_valid = obs[valid_mask]
    
    # Forward regression: model = a * obs + b
    slope_forward = np.nansum((obs_valid - np.nanmean(obs_valid)) * 
                              (data_valid - np.nanmean(data_valid))) / \
                    np.nansum((obs_valid - np.nanmean(obs_valid))**2)
    
    # Backward regression: obs = a * model + b, then invert
    slope_backward_inv = np.nansum((data_valid - np.nanmean(data_valid)) * 
                                   (obs_valid - np.nanmean(obs_valid))) / \
                         np.nansum((data_valid - np.nanmean(data_valid))**2)
    slope_backward = 1 / slope_backward_inv if slope_backward_inv != 0 else np.nan
    
    return np.round((slope_forward + slope_backward) / 2, 4)


def metrics(data: xr.Dataset, percentile_thresholds: Optional[list] = None) -> xr.Dataset:
    """
    Calculate comprehensive validation metrics for xarray Dataset.
    
    This function computes multiple statistical metrics for model-observation comparison,
    including standard errors, normalized metrics, and optionally percentile-based statistics.
    
    Parameters
    ----------
    data : xr.Dataset
        Dataset containing 'model_hs' and 'hs' (observations) variables
        with an 'obs' dimension
    percentile_thresholds : list, optional
        List of percentiles (0-100) for threshold-based validation.
        Example: [75, 90, 95] for 75th, 90th, and 95th percentiles
        
    Returns
    -------
    xr.Dataset
        Dataset containing computed metrics:
        - bias: Mean bias
        - nbias: Normalized bias
        - rmse: Root mean square error
        - nrmse: Normalized RMSE
        - mae: Mean absolute error
        - nmae: Normalized MAE
        - madp: Mean Absolute Difference of Percentiles
        - madc: Combined Mean Absolute Difference
        - correlation: Pearson correlation coefficient
        - skill_score: Murphy's skill score
        - scatter_index: Scatter index
        - symmetric_slope: Symmetric regression slope
        - nobs: Number of valid observations
        - model_hs: Mean model value
        - sat_hs: Mean observation value
        - model_std: Model standard deviation
        - sat_std: Observation standard deviation
        
        If percentile_thresholds provided, also includes:
        - bias_pXX: Bias for values above XXth percentile
        - rmse_pXX: RMSE for values above XXth percentile
        - mae_pXX: MAE for values above XXth percentile
        - nobs_pXX: Count for values above XXth percentile
        
    Examples
    --------
    >>> ds = xr.Dataset({
    ...     'model_hs': (['obs'], np.random.rand(100)),
    ...     'hs': (['obs'], np.random.rand(100))
    ... })
    >>> results = metrics(ds, percentile_thresholds=[90, 95])
    >>> print(results.bias.values)
    >>> print(results.bias_p90.values)  # Bias for 90th percentile and above
    
    Notes
    -----
    All normalized metrics are calculated with respect to the mean observation value.
    NaN values in input data are automatically excluded from calculations.
    """
    result = xr.Dataset()
    
    # Basic error metrics
    bias = data['model_hs'] - data['hs']
    result['bias'] = bias.mean(dim='obs')
    
    obs_sum = data['hs'].sum(dim='obs')
    obs_mean = data['hs'].mean(dim='obs')
    
    # Normalized metrics
    result['nbias'] = bias.sum(dim='obs') / obs_sum
    result['rmse'] = np.sqrt((bias ** 2.).mean(dim='obs'))
    result['nrmse'] = np.sqrt((bias ** 2.).sum(dim='obs') / (data['hs']**2).sum(dim='obs'))
    
    # Mean Absolute Error
    result['mae'] = np.abs(bias).mean(dim='obs')
    result['nmae'] = result['mae'] / obs_mean
    
    # Count of observations
    result['nobs'] = bias.count(dim='obs')
    
    # Mean values
    result['model_hs'] = data['model_hs'].mean(dim='obs')
    result['sat_hs'] = data['hs'].mean(dim='obs')
    
    # Standard deviations
    result['model_std'] = data['model_hs'].std(dim='obs')
    result['sat_std'] = data['hs'].std(dim='obs')
    
    # MADp and MADc metrics (requires numpy arrays for percentile calculations)
    # Check if we have multiple models (extra dimensions beyond 'obs')
    model_vals = data['model_hs'].values
    obs_vals = data['hs'].values
    
    # Determine if we have multiple models by checking array dimensions
    if model_vals.ndim > 1 and obs_vals.ndim == 1:
        # Multi-model case
        n_models = model_vals.shape[0] if model_vals.shape[0] != len(obs_vals) else model_vals.shape[1]
        
        madp_vals = np.full(n_models, np.nan)
        madc_vals = np.full(n_models, np.nan)
        
        for i in range(n_models):
            model_i = model_vals[i, :] if model_vals.shape[0] == n_models else model_vals[:, i]
            valid_mask = ~(np.isnan(model_i) | np.isnan(obs_vals))
            
            if np.sum(valid_mask) > 1:
                madp_vals[i] = MADp(model_i[valid_mask], obs_vals[valid_mask])
                madc_vals[i] = MADc(model_i[valid_mask], obs_vals[valid_mask])
        
        if 'model' in data.dims:
            result['madp'] = (['model'], madp_vals)
            result['madc'] = (['model'], madc_vals)
        else:
            result['madp'] = madp_vals
            result['madc'] = madc_vals
    else:
        # Single model case
        valid_mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
        if np.sum(valid_mask) > 1:
            result['madp'] = MADp(model_vals[valid_mask], obs_vals[valid_mask])
            result['madc'] = MADc(model_vals[valid_mask], obs_vals[valid_mask])
        else:
            result['madp'] = np.nan
            result['madc'] = np.nan
    
    # Correlation and other metrics (requires special handling for xarray)
    # We already extracted model_vals and obs_vals above
    
    # Determine if we have multiple models by checking array dimensions
    if model_vals.ndim > 1 and obs_vals.ndim == 1:
        # Multi-model case: model_vals has shape (n_models, n_obs) or (n_obs, n_models)
        # We need to process each model separately
        n_models = model_vals.shape[0] if model_vals.shape[0] != len(obs_vals) else model_vals.shape[1]
        
        # Initialize result arrays
        corr = np.full(n_models, np.nan)
        skill = np.full(n_models, np.nan)
        si = np.full(n_models, np.nan)
        sym_slope = np.full(n_models, np.nan)
        
        # Process each model
        for i in range(n_models):
            model_i = model_vals[i, :] if model_vals.shape[0] == n_models else model_vals[:, i]
            valid_mask = ~(np.isnan(model_i) | np.isnan(obs_vals))
            
            if np.sum(valid_mask) > 1:
                corr[i] = correlation(model_i[valid_mask], obs_vals[valid_mask])
                skill[i] = skill_score(model_i[valid_mask], obs_vals[valid_mask])
                si[i] = ScatterIndex(model_i[valid_mask], obs_vals[valid_mask])
                sym_slope[i] = symmetric_slope(model_i[valid_mask], obs_vals[valid_mask])
        
        # For multi-model case, need to specify dimension for proper xarray handling
        # Create a 'model' dimension if it doesn't exist
        if 'model' in data.dims:
            result['correlation'] = (['model'], corr)
            result['skill_score'] = (['model'], skill)
            result['scatter_index'] = (['model'], si)
            result['symmetric_slope'] = (['model'], sym_slope)
        else:
            # If no model dimension, just store as arrays (will be indexed by position)
            result['correlation'] = corr
            result['skill_score'] = skill
            result['scatter_index'] = si
            result['symmetric_slope'] = sym_slope
    else:
        # Single model case: both are 1D arrays
        valid_mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
        if np.sum(valid_mask) > 1:
            result['correlation'] = correlation(model_vals[valid_mask], obs_vals[valid_mask])
            result['skill_score'] = skill_score(model_vals[valid_mask], obs_vals[valid_mask])
            result['scatter_index'] = ScatterIndex(model_vals[valid_mask], obs_vals[valid_mask])
            result['symmetric_slope'] = symmetric_slope(model_vals[valid_mask], obs_vals[valid_mask])
        else:
            result['correlation'] = np.nan
            result['skill_score'] = np.nan
            result['scatter_index'] = np.nan
            result['symmetric_slope'] = np.nan
    
    # Percentile-based metrics (for extreme value validation)
    if percentile_thresholds is not None:
        # Use the original values for percentile calculations
        model_vals_orig = data['model_hs'].values
        obs_vals_orig = data['hs'].values
        
        for percentile in percentile_thresholds:
            # Calculate threshold value from observations
            if obs_vals_orig.ndim == 1:
                threshold = np.nanpercentile(obs_vals_orig, percentile)
                high_mask = obs_vals_orig >= threshold
            else:
                # For multi-dimensional case, use flattened obs
                threshold = np.nanpercentile(obs_vals_orig[~np.isnan(obs_vals_orig)], percentile)
                high_mask = obs_vals_orig >= threshold
            
            if np.sum(high_mask) > 1:
                obs_high = obs_vals_orig[high_mask]
                
                if model_vals_orig.ndim == 1:
                    # Single model case
                    model_high = model_vals_orig[high_mask]
                    
                    result[f'bias_p{percentile}'] = BIAS(model_high, obs_high)
                    result[f'rmse_p{percentile}'] = RMSE(model_high, obs_high)
                    result[f'mae_p{percentile}'] = MAE(model_high, obs_high)
                    result[f'correlation_p{percentile}'] = correlation(model_high, obs_high)
                    result[f'nobs_p{percentile}'] = np.sum(high_mask)
                    
                    obs_high_mean = np.nanmean(obs_high)
                    result[f'nbias_p{percentile}'] = (np.nanmean(model_high) - obs_high_mean) / obs_high_mean
                    result[f'nrmse_p{percentile}'] = RMSE(model_high, obs_high) / obs_high_mean
                else:
                    # Multi-model case: compute metrics for each model
                    # Determine which dimension has the models vs observations
                    if model_vals_orig.shape[0] == len(obs_vals_orig):
                        # Shape is (n_obs, n_models)
                        n_models = model_vals_orig.shape[1]
                        models_axis = 1
                    else:
                        # Shape is (n_models, n_obs)
                        n_models = model_vals_orig.shape[0]
                        models_axis = 0
                    
                    bias_p = np.full(n_models, np.nan)
                    rmse_p = np.full(n_models, np.nan)
                    mae_p = np.full(n_models, np.nan)
                    corr_p = np.full(n_models, np.nan)
                    nbias_p = np.full(n_models, np.nan)
                    nrmse_p = np.full(n_models, np.nan)
                    
                    for i in range(n_models):
                        # Extract model i data based on axis
                        if models_axis == 0:
                            model_i_full = model_vals_orig[i, :]
                        else:
                            model_i_full = model_vals_orig[:, i]
                        
                        model_high_i = model_i_full[high_mask]
                        valid_high = ~(np.isnan(model_high_i) | np.isnan(obs_high))
                        
                        if np.sum(valid_high) > 1:
                            bias_p[i] = BIAS(model_high_i[valid_high], obs_high[valid_high])
                            rmse_p[i] = RMSE(model_high_i[valid_high], obs_high[valid_high])
                            mae_p[i] = MAE(model_high_i[valid_high], obs_high[valid_high])
                            corr_p[i] = correlation(model_high_i[valid_high], obs_high[valid_high])
                            
                            obs_high_mean = np.nanmean(obs_high[valid_high])
                            nbias_p[i] = (np.nanmean(model_high_i[valid_high]) - obs_high_mean) / obs_high_mean
                            nrmse_p[i] = rmse_p[i] / obs_high_mean
                    
                    # For multi-model, specify dimension if it exists
                    if 'model' in data.dims:
                        result[f'bias_p{percentile}'] = (['model'], bias_p)
                        result[f'rmse_p{percentile}'] = (['model'], rmse_p)
                        result[f'mae_p{percentile}'] = (['model'], mae_p)
                        result[f'correlation_p{percentile}'] = (['model'], corr_p)
                        result[f'nobs_p{percentile}'] = np.sum(high_mask)
                        result[f'nbias_p{percentile}'] = (['model'], nbias_p)
                        result[f'nrmse_p{percentile}'] = (['model'], nrmse_p)
                    else:
                        result[f'bias_p{percentile}'] = bias_p
                        result[f'rmse_p{percentile}'] = rmse_p
                        result[f'mae_p{percentile}'] = mae_p
                        result[f'correlation_p{percentile}'] = corr_p
                        result[f'nobs_p{percentile}'] = np.sum(high_mask)
                        result[f'nbias_p{percentile}'] = nbias_p
                        result[f'nrmse_p{percentile}'] = nrmse_p
            else:
                # Not enough data for this percentile
                result[f'bias_p{percentile}'] = np.nan
                result[f'rmse_p{percentile}'] = np.nan
                result[f'mae_p{percentile}'] = np.nan
                result[f'correlation_p{percentile}'] = np.nan
                result[f'nobs_p{percentile}'] = 0
                result[f'nbias_p{percentile}'] = np.nan
                result[f'nrmse_p{percentile}'] = np.nan
    
    return result
