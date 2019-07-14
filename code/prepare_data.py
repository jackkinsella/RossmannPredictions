import pandas as pd
import numpy as np

store = pd.read_csv("data/store.csv")
new_train = pd.read_csv("data/new_train.csv", index_col=0,
                        dtype={"StateHoliday": object})

data = store.merge(new_train, on="Store")

data["Date"] = pd.to_datetime(data["Date"])

# Feature engineering
data["HasCompetition"] = np.where(data["CompetitionOpenSinceMonth"] >= 0, 1, 0)
int_cols = ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth", "Promo2SinceYear",
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
data = pd.get_dummies(data)

# Prepare predictors and labels
X = data.drop(["Sales", "Date"], axis=1)
# X = data.loc[:, "Customers"]
y = data["Sales"]


def show_data_structure():
    print(data.info())


if __name__ == '__main__':
    show_data_structure()
