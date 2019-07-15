import pandas as pd
import numpy as np
import datetime


def generate_sample_train():
    train = pd.read_csv("data/new_train.csv",
                        index_col=0,
                        dtype={"StateHoliday": object})
    train = train.drop(columns=["Sales"])
    train.sample(frac=0.05).to_csv("data/train_sampled.csv")


generate_sample_train()


def process_data(train_data_filepath):
    print("Starting to process data...")
    store = pd.read_csv("data/store.csv")
    new_train = pd.read_csv(train_data_filepath,
                            index_col=0,
                            dtype={"StateHoliday": object})
    data = store.merge(new_train, on="Store")
    print("Data read into memory...")

    data["Date"] = pd.to_datetime(data["Date"])

    # Feature engineering
    data["HasCompetition"] = np.where(data["CompetitionOpenSinceMonth"] >= 0,
                                      1, 0)
    int_cols = [
        "CompetitionOpenSinceYear",
        "CompetitionOpenSinceMonth",
        "Promo2SinceYear",
        "Promo2SinceWeek",
    ]

    # Deal with Nans
    data["CompetitionDistance"].fillna(-1, inplace=True)
    data["PromoInterval"].fillna(-1, inplace=True)
    data[int_cols] = data[int_cols].fillna(-1)

    assert not data.isna().any().any()

    # Convert to ints once nans handled
    data[int_cols] = data[int_cols].astype(int)

    # Make time categorical
    data["Week"] = data["Date"].dt.week
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    # Use one-hot-encoding
    data = pd.get_dummies(data).sort_values(by="Date", ascending=True)
    data = data.drop(columns=['Date'])
    return data

    # Commenting out because it made scores worse
    # data["Promo2LengthInDays"] = data.apply(promo2_length, axis=1)
    # data["CompetitionLengthInDays"] = data.apply(competition_length, axis=1)


def competition_length(row):
    year = int(row.CompetitionOpenSinceYear)
    month = int(row.CompetitionOpenSinceMonth)
    if year == -1:
        return 0
    else:
        competition_start = datetime.datetime(year=year, month=month, day=1)
        duration = row.Date - competition_start
        return duration.days


def promo2_length(row):
    year = int(row.Promo2SinceYear)
    week = int(row.Promo2SinceWeek)
    if year == -1:
        return 0
    else:
        promo_start = datetime.datetime.strptime(f'{year}-W{week}' + '-1',
                                                 '%G-W%V-%u')
        duration = row.Date - promo_start
        return duration.days
