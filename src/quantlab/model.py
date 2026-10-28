
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "volume_ratio_20d"
]

# Target return includes the intended next-open entry and exit.
TARGET = "future_open_return_5d"

# Never train on targets whose exit has not occurred.
OUTCOME_DATE = "execution_outcome_date"


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:

    required = FEATURES + [TARGET, OUTCOME_DATE]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model columns: {missing}. "
            "Calculate execution targets before training."
        )

    data = df.sort_values(["Ticker", "Date"]).copy()

    universe_average = (
        data.groupby("Date")[TARGET].transform("mean")
    )

    data["relative_target"] = (
        data[TARGET] - universe_average
    )

    data = data.dropna(
        subset=required + ["relative_target"]
    )

    return data


def create_ridge_model():

    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    )


def train_first_model(df: pd.DataFrame):

    data = prepare_model_data(df)

    # Training outcomes must be fully observed before validation starts.
    train = data[
        (data["Date"] < "2022-01-01")
        & (data[OUTCOME_DATE] < "2022-01-01")
    ].copy()

    validation = data[
        (data["Date"] >= "2022-01-01")
        & (data["Date"] < "2025-01-01")
        ].copy()

    if train.empty or validation.empty:
        raise ValueError("Insufficient model data")

    model = create_ridge_model()

    model.fit(
        train[FEATURES],
        train["relative_target"]
    )

    validation["model_prediction"] = model.predict(
        validation[FEATURES]
    )

    return model, train, validation


def walk_forward_model(df: pd.DataFrame) -> pd.DataFrame:

    data = prepare_model_data(df)

    all_predictions = []

    for year in [2022, 2023, 2024]:

        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year + 1}-01-01")

        # Rolling training windows exclude unsettled targets.
        train = data[
            (data["Date"] < start)
            & (data[OUTCOME_DATE] < start)
        ].copy()

        test = data[
            (data["Date"] >= start)
            & (data["Date"] < end)
            ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"Insufficient data for {year}"
            )

        model = create_ridge_model()

        model.fit(
            train[FEATURES],
            train["relative_target"]
        )

        test["model_prediction"] = model.predict(
            test[FEATURES]
        )

        test["evaluation_year"] = year

        print(
            f"{year}: "
            f"{len(train)} training rows, "
            f"{len(test)} testing rows"
        )

        coefficients = pd.Series(
            model.named_steps["ridge"].coef_,
            index=FEATURES
        )

        print("Learned coefficients:")
        print(coefficients.round(6))

        all_predictions.append(test)

    return pd.concat(
        all_predictions,
        ignore_index=True
    )
