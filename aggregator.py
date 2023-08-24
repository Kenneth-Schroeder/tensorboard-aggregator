# MIT License
# Copyright (c) 2019 Sebastian Penhouet
# GitHub project: https://github.com/Spenhouet/tensorboard-aggregator
# ==============================================================================
"""Aggregates multiple tensorbaord runs"""

import ast
import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.core.util.event_pb2 import Event

FOLDER_NAME = 'csvs'

def get_items_if_key(scalar_accumulator, key):
    try:
        return scalar_accumulator.Items(key)
    except KeyError:
        return []
    
def extract_singles(dpath, subpath):
    accumulators = [EventAccumulator(str(dpath / dname / subpath)).Reload() for dname in os.listdir(dpath) if dname != FOLDER_NAME]

    scalar_accumulators = [acc.scalars for acc in accumulators]
    # Filter non event files
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

    # Get and validate all scalar keys
    all_keys = [key for scalar_accumulator in scalar_accumulators for key in scalar_accumulator.Keys()]
    keys = list(set(all_keys))

    all_scalar_events_per_key = [[get_items_if_key(scalar_accumulator, key) for scalar_accumulator in scalar_accumulators] for key in keys]

    # Get and validate all steps per key
    all_steps_per_key = [[[scalar_event.step for scalar_event in run_events] for run_events in all_scalar_events]
                         for all_scalar_events in all_scalar_events_per_key]


    # keys = table names / identifiers
    # steps per key = total steps recorded per key (might differ from run to run - some are even 0) [num_keys, number_of_runs]
    # wall_times_per_key = average wall times per step per key (one value per key) [num_keys, number_of_runs, num_scalars]
    # scalars_per_key_n_run_n = [num_keys, number_of_runs, num_scalars]

    steps_per_key_n_run = all_steps_per_key

    # Get and average wall times per step per key
    wall_times_per_key_n_run_n_scalar = [[[scalar_event.wall_time for scalar_event in run_events] for run_events in all_scalar_events]
                          for all_scalar_events in all_scalar_events_per_key]

    # Get values per step per key
    scalars_per_key_n_run_n = [[[scalar_event.value for scalar_event in run_events] for run_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    # run folders with one file per key
    #run_counter = 0
    #for acc in accumulators:
    #    if acc.scalars.Keys():
    #        dname = acc.path.split('/')[-1]
    #        assert(dname != ".DS_Store")
    #        if dname != FOLDER_NAME:
    #            for j, key in enumerate(keys):
    #                csv_dict = {}
    #                if len(steps_per_key_n_run[j][run_counter]) > 0:
    #                    csv_dict[key] = scalars_per_key_n_run_n[j][run_counter]
    #                    csv_dict['Steps'] = steps_per_key_n_run[j][run_counter]
    #                    csv_dict['Wall_Time'] = wall_times_per_key_n_run_n_scalar[j][run_counter]
    #                if csv_dict:
    #                    write_single_csv(dpath / FOLDER_NAME, dname, key, csv_dict)
    #            run_counter+=1

    # key folders with one file per run
    for j, key in enumerate(keys):
        run_counter = 0
        for acc in accumulators:
            if acc.scalars.Keys():
                dname = acc.path.split('/')[-1]
                assert(dname != ".DS_Store")
                if dname != FOLDER_NAME:
                    csv_dict = {}
                    if len(steps_per_key_n_run[j][run_counter]) > 0:
                        csv_dict[key] = scalars_per_key_n_run_n[j][run_counter]
                        csv_dict['Steps'] = steps_per_key_n_run[j][run_counter]
                        csv_dict['Wall_Time'] = wall_times_per_key_n_run_n_scalar[j][run_counter]
                    if csv_dict:
                        write_single_csv(dpath / FOLDER_NAME, key, dname, csv_dict)
                    run_counter+=1


def get_valid_filename(s):
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

def write_single_csv(output_path, folder_name, file_title, data_dict):
    path = output_path / get_valid_filename(folder_name)

    if not path.exists():
        os.makedirs(path)

    file_name = get_valid_filename(file_title) + '.csv'
    df = pd.DataFrame(data_dict)
    df.to_csv(path / file_name, sep=';')

def collect_csvs(dpath, output, subpaths):
    name = dpath.name

    aggregation_ops = [np.mean, np.min, np.max, np.median, np.std, np.var]

    for subpath in subpaths:
        extract_singles(dpath, subpath)

if __name__ == '__main__':
    def param_list(param):
        p_list = ast.literal_eval(param)
        if type(p_list) is not list:
            raise argparse.ArgumentTypeError("Parameter {} is not a list".format(param))
        return p_list

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="main path for tensorboard files", default=os.getcwd())
    parser.add_argument("--subpaths", type=param_list, help="subpath structures", default=['.'])
    parser.add_argument("--output", type=str, help="aggregation can be saved as tensorboard file (summary) or as table (csv)", default='summary')

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(path))

    subpaths = [path / dname / subpath for subpath in args.subpaths for dname in os.listdir(path) if dname != FOLDER_NAME]

    for subpath in subpaths:
        if not os.path.exists(subpath):
            raise argparse.ArgumentTypeError("Parameter {} is not a valid path".format(subpath))

    if args.output not in ['summary', 'csv']:
        raise argparse.ArgumentTypeError("Parameter {} is not summary or csv".format(args.output))

    collect_csvs(path, args.output, args.subpaths)
