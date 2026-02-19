from satellite_preprocessing import Sat_processer
from model_preprocessing import submit as model_preproc
from buoy_preprocessing import submit as buoy_preproc
from validation import main as scatter_plot
from validation_map import main as stats_map
from validation_timeseries import main as timeseries
from validation_tracks import main as tracks
import os
from argparse import ArgumentParser

parser = ArgumentParser(description='Wave model validation')
parser.add_argument('-c', '--configuration', default='conf.yaml', help='catalog file')
parser.add_argument('-s', '--start_date', help='start date')
parser.add_argument('-e', '--end_date',  help='end date')
parser.add_argument('--use-buoy', action='store_true', help='use buoy data instead of satellite')
args = parser.parse_args()
conf_path=args.configuration

start_date=args.start_date
end_date=args.end_date

if args.use_buoy:
    # Buoy validation workflow
    print("=" * 60)
    print("BUOY VALIDATION WORKFLOW")
    print("=" * 60)
    buoy_preproc(conf_path, start_date, end_date)
    model_preproc(conf_path, start_date, end_date, obs_type='buoy')
else:
    # Satellite validation workflow (original)
    print("=" * 60)
    print("SATELLITE VALIDATION WORKFLOW")
    print("=" * 60)
    sat=Sat_processer(conf_path,start_date,end_date).run()
    model_preproc(conf_path,start_date,end_date)

# scatter_plot(conf_path,start_date,end_date)
# stats_map(conf_path,start_date,end_date)
# timeseries(conf_path,start_date,end_date)
# tracks(conf_path,start_date,end_date)
