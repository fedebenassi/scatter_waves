import munch
from datetime import date, timedelta
import yaml
import os

def getConfigurationByID(path,confId):
    globalConf = yaml.load(open(path),Loader=yaml.Loader)
    return munch.Munch.fromDict(globalConf[confId])

def daysBetweenDates(start_date,end_date):
    start_date=str(start_date)
    end_date = str(end_date)
    
    # Validate date string format
    if len(start_date) != 8:
        raise ValueError(f"Invalid start_date format: '{start_date}'. Expected YYYYMMDD (8 digits).")
    if len(end_date) != 8:
        raise ValueError(f"Invalid end_date format: '{end_date}'. Expected YYYYMMDD (8 digits).")
    
    try:
        start_date = date(int(start_date[:4]), int(start_date[4:6]), int(start_date[6:8]))
    except ValueError as e:
        raise ValueError(f"Invalid start_date: '{start_date}'. Error: {e}")
    
    try:
        end_date = date(int(end_date[:4]), int(end_date[4:6]), int(end_date[6:8]))
    except ValueError as e:
        raise ValueError(f"Invalid end_date: '{end_date}'. Error: {e}")
    
    delta = end_date - start_date  # returns timedelta
    if delta.days < 0:
        raise ValueError(f"end_date ({end_date}) must be after start_date ({start_date})")
    
    return [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]

def ticker(coord_min,coord_max):
    dif =coord_max - coord_min
    if dif >= 180:
        step = 25
    elif dif >= 50:
        step = 10
    elif dif >= 25:
        step = 5
    elif dif >= 10:
        step = 2
    else:
        step = 1
    return step
