# original script
# https://github.com/google-research/google-research/blob/master/tft/script_download_data.py

import argparse

import gc
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pyunpack
import wget
from tqdm import tqdm


class Config(object):
    datasets = ["volatility", "electricity", "traffic", "favorita"]

    def __init__(self, data_folder, data_csv_path):
        self.data_folder = data_folder
        self.data_csv_path = data_csv_path

    @classmethod
    def for_name(cls, name, output_folder):
        dataset_map = {
            "volatility": "volatility.csv",
            "electricity": "electricity.csv",
            "traffic": "traffic.csv",
            "favorita": "favorita.csv",
        }

        return Config(output_folder, os.path.join(output_folder, dataset_map[name]))


def download_from_url(url, output_path):
    wget.download(url, output_path)


def recreate_folder(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
    print("Unzipping file: {}".format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

    if not os.path.exists(output_file):
        raise ValueError(
            "Error in unzipping process! {} not found.".format(output_file)
        )


def download_and_unzip(url, zip_path, csv_path, data_folder):
    download_from_url(url, zip_path)
    unzip(zip_path, csv_path, data_folder)
    print("Done.")


def download_volatility(config):
    url = "https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip"

    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, "oxfordmanrealizedvolatilityindices.csv")
    zip_path = os.path.join(data_folder, "oxfordmanrealizedvolatilityindices.zip")

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print("Unzip complete. Adding extra inputs")

    df = pd.read_csv(csv_path, index_col=0)  # no explicit index

    # Adds additional date/day fields
    idx = [
        str(s).split("+")[0] for s in df.index
    ]  # ignore timezones, we don't need them
    dates = pd.to_datetime(idx)
    df["date"] = dates
    df["days_from_start"] = (dates - pd.datetime(2000, 1, 3)).days
    df["day_of_week"] = dates.dayofweek
    df["day_of_month"] = dates.day
    df["week_of_year"] = dates.weekofyear
    df["month"] = dates.month
    df["year"] = dates.year
    df["categorical_id"] = df["Symbol"].copy()

    # Processes log volatility
    vol = df["rv5_ss"].copy()
    vol.loc[vol == 0.0] = np.nan
    df["log_vol"] = np.log(vol)

    # Adds static information
    symbol_region_mapping = {
        ".AEX": "EMEA",
        ".AORD": "APAC",
        ".BFX": "EMEA",
        ".BSESN": "APAC",
        ".BVLG": "EMEA",
        ".BVSP": "AMER",
        ".DJI": "AMER",
        ".FCHI": "EMEA",
        ".FTMIB": "EMEA",
        ".FTSE": "EMEA",
        ".GDAXI": "EMEA",
        ".GSPTSE": "AMER",
        ".HSI": "APAC",
        ".IBEX": "EMEA",
        ".IXIC": "AMER",
        ".KS11": "APAC",
        ".KSE": "APAC",
        ".MXX": "AMER",
        ".N225": "APAC ",
        ".NSEI": "APAC",
        ".OMXC20": "EMEA",
        ".OMXHPI": "EMEA",
        ".OMXSPI": "EMEA",
        ".OSEAX": "EMEA",
        ".RUT": "EMEA",
        ".SMSI": "EMEA",
        ".SPX": "AMER",
        ".SSEC": "APAC",
        ".SSMI": "EMEA",
        ".STI": "APAC",
        ".STOXX50E": "EMEA",
    }

    df["Region"] = df["Symbol"].apply(lambda k: symbol_region_mapping[k])

    # Performs final processing
    output_df_list = []
    for grp in df.groupby("Symbol"):
        sliced = grp[1].copy()
        sliced.sort_values("days_from_start", inplace=True)
        # Impute log volatility values
        sliced["log_vol"].fillna(method="ffill", inplace=True)
        sliced.dropna()
        output_df_list.append(sliced)

    df = pd.concat(output_df_list, axis=0)

    output_file = config.data_csv_path
    print("Completed formatting, saving to {}".format(output_file))
    df.to_csv(output_file)

    print("Done.")


def download_electricity(config):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, "LD2011_2014.txt")
    zip_path = csv_path + ".zip"

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print("Aggregating to hourly data")

    df = pd.read_csv(csv_path, index_col=0, sep=";", decimal=",")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    output = df.resample("1h").mean().replace(0.0, np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in tqdm(output):
        srs = output[label]

        start_date = min(srs.fillna(method="ffill").dropna().index)
        end_date = max(srs.fillna(method="bfill").dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.0)

        tmp = pd.DataFrame({"power_usage": srs})
        date = tmp.index
        tmp["t"] = (date - earliest_time).seconds / 60 / 60 + (
            date - earliest_time
        ).days * 24
        tmp["days_from_start"] = (date - earliest_time).days
        tmp["categorical_id"] = label
        tmp["date"] = date
        tmp["id"] = label
        tmp["hour"] = date.hour
        tmp["day"] = date.day
        tmp["day_of_week"] = date.dayofweek
        tmp["month"] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join="outer").reset_index(drop=True)

    output["categorical_id"] = output["id"].copy()
    output["hours_from_start"] = output["t"]
    output["categorical_day_of_week"] = output["day_of_week"].copy()
    output["categorical_hour"] = output["hour"].copy()

    # Filter to match range used by other academic papers
    output = output[
        (output["days_from_start"] >= 1096) & (output["days_from_start"] < 1346)
    ].copy()

    output.to_csv(config.data_csv_path)

    print("Done.")


def download_traffic(config):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip"

    data_folder = config.data_folder
    csv_path = os.path.join(data_folder, "PEMS_train")
    zip_path = os.path.join(data_folder, "PEMS-SF.zip")

    download_and_unzip(url, zip_path, csv_path, data_folder)

    print("Aggregating to hourly data")

    def process_list(s, variable_type=int, delimiter=None):
        """Parses a line in the PEMS format to a list."""
        if delimiter is None:
            return [
                variable_type(i) for i in s.replace("[", "").replace("]", "").split()
            ]
        else:
            return [
                variable_type(i)
                for i in s.replace("[", "").replace("]", "").split(delimiter)
            ]

    def read_single_list(filename):
        """Returns single list from a file in the PEMS-custom format."""
        with open(os.path.join(data_folder, filename), "r") as dat:
            return process_list(dat.readlines()[0])

    def read_matrix(filename):
        """Returns a matrix from a file in the PEMS-custom format."""
        array_list = []
        with open(os.path.join(data_folder, filename), "r") as dat:
            lines = dat.readlines()
            for i, line in enumerate(lines):
                if (i + 1) % 50 == 0:
                    print(
                        "Completed {} of {} rows for {}".format(
                            i + 1, len(lines), filename
                        )
                    )

                array = [
                    process_list(row_split, variable_type=float, delimiter=None)
                    for row_split in process_list(
                        line, variable_type=str, delimiter=";"
                    )
                ]
                array_list.append(array)
        return array_list

    shuffle_order = np.array(read_single_list("randperm")) - 1  # index from 0
    train_dayofweek = read_single_list("PEMS_trainlabels")
    train_tensor = read_matrix("PEMS_train")
    test_dayofweek = read_single_list("PEMS_testlabels")
    test_tensor = read_matrix("PEMS_test")

    # Inverse permutate shuffle order
    print("Shuffling")
    inverse_mapping = {
        new_location: previous_location
        for previous_location, new_location in enumerate(shuffle_order)
    }
    reverse_shuffle_order = np.array(
        [inverse_mapping[new_location] for new_location, _ in enumerate(shuffle_order)]
    )

    # Group and reoder based on permuation matrix
    print("Reodering")
    day_of_week = np.array(train_dayofweek + test_dayofweek)
    combined_tensor = np.array(train_tensor + test_tensor)

    day_of_week = day_of_week[reverse_shuffle_order]
    combined_tensor = combined_tensor[reverse_shuffle_order]

    # Put everything back into a dataframe
    print("Parsing as dataframe")
    labels = ["traj_{}".format(i) for i in read_single_list("stations_list")]

    hourly_list = []
    for day, day_matrix in enumerate(combined_tensor):
        # Hourly data
        hourly = pd.DataFrame(day_matrix.T, columns=labels)
        hourly["hour_on_day"] = [
            int(i / 6) for i in hourly.index
        ]  # sampled at 10 min intervals
        if hourly["hour_on_day"].max() > 23 or hourly["hour_on_day"].min() < 0:
            raise ValueError(
                "Invalid hour! {}-{}".format(
                    hourly["hour_on_day"].min(), hourly["hour_on_day"].max()
                )
            )

        hourly = hourly.groupby("hour_on_day", as_index=True).mean()[labels]
        hourly["sensor_day"] = day
        hourly["time_on_day"] = hourly.index
        hourly["day_of_week"] = day_of_week[day]

        hourly_list.append(hourly)

    hourly_frame = pd.concat(hourly_list, axis=0, ignore_index=True, sort=False)

    # Flatten such that each entitiy uses one row in dataframe
    store_columns = [c for c in hourly_frame.columns if "traj" in c]
    other_columns = [c for c in hourly_frame.columns if "traj" not in c]
    flat_df = pd.DataFrame(
        columns=["values", "prev_values", "next_values"] + other_columns + ["id"]
    )

    def format_index_string(x):
        """Returns formatted string for key."""

        if x < 10:
            return "00" + str(x)
        elif x < 100:
            return "0" + str(x)
        elif x < 1000:
            return str(x)

        raise ValueError("Invalid value of x {}".format(x))

    for store in tqdm(store_columns):
        sliced = hourly_frame[[store] + other_columns].copy()
        sliced.columns = ["values"] + other_columns
        sliced["id"] = int(store.replace("traj_", ""))

        # Sort by Sensor-date-time
        key = (
            sliced["id"].apply(str)
            + sliced["sensor_day"].apply(lambda x: "_" + format_index_string(x))
            + sliced["time_on_day"].apply(lambda x: "_" + format_index_string(x))
        )
        sliced = sliced.set_index(key).sort_index()

        sliced["values"] = sliced["values"].fillna(method="ffill")
        sliced["prev_values"] = sliced["values"].shift(1)
        sliced["next_values"] = sliced["values"].shift(-1)

        flat_df = flat_df.append(sliced.dropna(), ignore_index=True, sort=False)

    # Filter to match range used by other academic papers
    index = flat_df["sensor_day"]
    flat_df = flat_df[index < 173].copy()

    # Creating columns fo categorical inputs
    flat_df["categorical_id"] = flat_df["id"].copy()
    flat_df["hours_from_start"] = flat_df["time_on_day"] + flat_df["sensor_day"] * 24.0
    flat_df["categorical_day_of_week"] = flat_df["day_of_week"].copy()
    flat_df["categorical_time_on_day"] = flat_df["time_on_day"].copy()

    flat_df.to_csv(config.data_csv_path)
    print("Done.")


def process_favorita(config):
    url = "https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data"

    data_folder = config.data_folder

    # Save manual download to root folder to avoid deleting when re-processing.
    zip_file = os.path.join(data_folder, "..", "favorita-grocery-sales-forecasting.zip")

    if not os.path.exists(zip_file):
        raise ValueError(
            "Favorita zip file not found in {}!".format(zip_file)
            + " Please manually download data from Kaggle @ {}".format(url)
        )

    # Unpack main zip file
    output_folder = os.path.join(data_folder,"favorita-grocery-sales-forecasting")
    outputs_file = os.path.join(output_folder, "train.csv.7z")
    unzip(zip_file, outputs_file, data_folder)

    # Unpack individually zipped files
    for fl in glob.glob(os.path.join(output_folder, "*.7z")):
        _, file_name = os.path.split(fl.replace(".7z", ""))
        csv_file = os.path.join(data_folder, file_name)
        unzip(fl, csv_file, data_folder)

    print("Unzipping complete, commencing data processing...")

    # Extract only a subset of data to save/process for efficiency
    start_date = pd.datetime(2015, 1, 1)
    end_date = pd.datetime(2016, 6, 1)

    print("Regenerating data...")

    # load temporal data
    temporal = pd.read_csv(os.path.join(data_folder, "train.csv"), index_col=0)

    store_info = pd.read_csv(os.path.join(data_folder, "stores.csv"), index_col=0)
    oil = pd.read_csv(os.path.join(data_folder, "oil.csv"), index_col=0).iloc[:, 0]
    holidays = pd.read_csv(os.path.join(data_folder, "holidays_events.csv"))
    items = pd.read_csv(os.path.join(data_folder, "items.csv"), index_col=0)
    transactions = pd.read_csv(os.path.join(data_folder, "transactions.csv"))

    # Take first 6 months of data
    temporal["date"] = pd.to_datetime(temporal["date"])

    # Filter dates to reduce storage space requirements
    if start_date is not None:
        temporal = temporal[(temporal["date"] >= start_date)]
    if end_date is not None:
        temporal = temporal[(temporal["date"] < end_date)]

    dates = temporal["date"].unique()

    # Add trajectory identifier
    temporal["traj_id"] = (
        temporal["store_nbr"].apply(str) + "_" + temporal["item_nbr"].apply(str)
    )
    temporal["unique_id"] = temporal["traj_id"] + "_" + temporal["date"].apply(str)

    # Remove all IDs with negative returns
    print("Removing returns data")
    min_returns = temporal["unit_sales"].groupby(temporal["traj_id"]).min()
    valid_ids = set(min_returns[min_returns >= 0].index)
    selector = temporal["traj_id"].apply(lambda traj_id: traj_id in valid_ids)
    new_temporal = temporal[selector].copy()
    del temporal
    gc.collect()
    temporal = new_temporal
    temporal["open"] = 1

    # Resampling
    print("Resampling to regular grid")
    resampled_dfs = []
    for traj_id, raw_sub_df in temporal.groupby("traj_id"):
        print("Resampling", traj_id)
        sub_df = raw_sub_df.set_index("date", drop=True).copy()
        sub_df = sub_df.resample("1d").last()
        sub_df["date"] = sub_df.index
        sub_df[["store_nbr", "item_nbr", "onpromotion"]] = sub_df[
            ["store_nbr", "item_nbr", "onpromotion"]
        ].fillna(method="ffill")
        sub_df["open"] = sub_df["open"].fillna(0)  # flag where sales data is unknown
        sub_df["log_sales"] = np.log(sub_df["unit_sales"])

        resampled_dfs.append(sub_df.reset_index(drop=True))

    new_temporal = pd.concat(resampled_dfs, axis=0)
    del temporal
    gc.collect()
    temporal = new_temporal

    print("Adding oil")
    oil.name = "oil"
    oil.index = pd.to_datetime(oil.index)
    oil = oil.reindex(dates)
    temporal = temporal.join(
        oil.loc[dates].fillna(method="ffill"), on="date", how="left"
    )
    temporal["oil"] = temporal["oil"].fillna(-1)

    print("Adding store info")
    temporal = temporal.join(store_info, on="store_nbr", how="left")

    print("Adding item info")
    temporal = temporal.join(items, on="item_nbr", how="left")

    transactions["date"] = pd.to_datetime(transactions["date"])
    temporal = temporal.merge(
        transactions,
        left_on=["date", "store_nbr"],
        right_on=["date", "store_nbr"],
        how="left",
    )
    temporal["transactions"] = temporal["transactions"].fillna(-1)

    # Additional date info
    temporal["day_of_week"] = pd.to_datetime(temporal["date"].values).dayofweek
    temporal["day_of_month"] = pd.to_datetime(temporal["date"].values).day
    temporal["month"] = pd.to_datetime(temporal["date"].values).month

    # Add holiday info
    print("Adding holidays")
    holiday_subset = holidays[holidays["transferred"].apply(lambda x: not x)].copy()
    holiday_subset.columns = [
        s if s != "type" else "holiday_type" for s in holiday_subset.columns
    ]
    holiday_subset["date"] = pd.to_datetime(holiday_subset["date"])
    local_holidays = holiday_subset[holiday_subset["locale"] == "Local"]
    regional_holidays = holiday_subset[holiday_subset["locale"] == "Regional"]
    national_holidays = holiday_subset[holiday_subset["locale"] == "National"]

    temporal["national_hol"] = temporal.merge(
        national_holidays, left_on=["date"], right_on=["date"], how="left"
    )["description"].fillna("")
    temporal["regional_hol"] = temporal.merge(
        regional_holidays,
        left_on=["state", "date"],
        right_on=["locale_name", "date"],
        how="left",
    )["description"].fillna("")
    temporal["local_hol"] = temporal.merge(
        local_holidays,
        left_on=["city", "date"],
        right_on=["locale_name", "date"],
        how="left",
    )["description"].fillna("")

    temporal.sort_values("unique_id", inplace=True)

    print("Saving processed file to {}".format(config.data_csv_path))
    temporal.to_csv(config.data_csv_path)


def download(expt_name, output_folder, force_download=True):
    print("#### Running download script ###")

    config = Config.for_name(expt_name, output_folder)

    if os.path.exists(config.data_csv_path) and not force_download:
        print("Data has been processed for {}. Skipping download...".format(expt_name))
        sys.exit(0)
    else:
        print("Resetting data folder...")
        recreate_folder(config.data_folder)

    # Default download functions
    download_functions = {
        "volatility": download_volatility,
        "electricity": download_electricity,
        "traffic": download_traffic,
        "favorita": process_favorita,
    }

    if expt_name not in download_functions:
        raise ValueError("Unrecongised experiment! name={}".format(expt_name))

    download_function = download_functions[expt_name]

    # Run data download
    print("Getting {} data...".format(expt_name))
    download_function(config)

    print("Download completed.")


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        experiment_names = Config.datasets

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)),
        )
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download",
        )
        parser.add_argument(
            "force_download",
            metavar="r",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to re-run data download",
        )

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, args.force_download == "yes", root_folder

    name, force, folder = get_args()
    download(expt_name=name, output_folder=folder, force_download=force)
