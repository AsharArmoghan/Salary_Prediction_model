import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

# Categorical encoding orders shared across the default preprocessor
GENDER_ORDER = ["female", "male", "other"]
EDUCATION_ORDER = [
    "primary",
    "high school",
    "diploma",
    "bachelors",
    "masters",
    "phd",
]
CATEGORICAL_INDICES = [1, 2, 4]

# Output column order after the default preprocessor:
# [Age(scaled), Education(ord), Gender(ord), Experience(scaled), Job(ord)]
# so the native-categorical indices for HGBR are [1, 2, 4].

# setting log transform for Right skewed data
log_transform = FunctionTransformer(func=np.log1p, validate=False)


# PIPELINES
def create_age_pipeline():
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )


def create_experience_pipeline():
    return Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', log_transform),
            ('scaler', StandardScaler()),
        ]
    )


def create_gender_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[GENDER_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def create_education_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[EDUCATION_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def create_job_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


def create_preprocessor():
    """Default preprocessor (most recent, GBR-ready).

    All categorical features are ordinal-encoded so HistGradientBoostingRegressor
    can treat them as native categoricals, and numeric features keep median
    imputation + log scaling (harmless for trees, consistent with baselines).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('age', create_age_pipeline(), ["Age"]),
            ('education', create_education_pipeline(), ["Education"]),
            ('gender', create_gender_pipeline(), ["Gender"]),
            ('skewed', create_experience_pipeline(), ["Experience"]),
            ('job', create_job_pipeline(), ["Job"]),
        ]
    )
    return preprocessor


# creating Models
def linear_model_pipeline(preprocessor):
    return Pipeline(
        steps=[('preprocessor', preprocessor), ('model', LinearRegression())]
    )


def ridge_model_pipeline(preprocessor):
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge())])


def ridge_gridCv_pipeline(preprocessor):
    pipe = ridge_model_pipeline(preprocessor)
    param_grid = {
        'model__alpha': [0.1, 1, 1.8307, 10, 100],
    }
    return GridSearchCV(
        pipe, param_grid=param_grid, cv=5, scoring='r2', n_jobs=1, verbose=2
    )


def gbr_pipeline(preprocessor):
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", HistGradientBoostingRegressor(random_state=42)),
        ]
    )


def gbr_gridCv_pipeline(preprocessor):
    """GridSearchCV over HGBR hyperparameters using the given preprocessor."""
    pipe = gbr_pipeline(preprocessor)
    param_grid = {
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5, 7, None],
        "model__max_iter": [100, 200, 300],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__min_samples_leaf": [20, 50, 100],
    }
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )


def create_native_gbr_pipeline():
    """Core HGBR component with native categorical handling.

    Uses the default preprocessor so categorical indices are CATEGORICAL_INDICES.
    """
    preprocessor = create_preprocessor()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    categorical_features=CATEGORICAL_INDICES,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    learning_rate=0.1,
                    max_iter=300,
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, Y_test):
    Y_predict = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test, Y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_predict)
    return {'predictions': Y_predict, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}


# ---------------------------------------------------------------------------
# Data loading helper (reuses notebook data-prep logic)
# ---------------------------------------------------------------------------
def load_training_data():
    """Reproduce the notebook's data prep and return (X_train, X_test, Y_train, Y_test)."""
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    from sklearn.model_selection import train_test_split

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS, "mohithsairamreddy/salary-data", "Salary_Data.csv"
    )

    df.rename(
        columns={
            "Years of Experience": "Experience",
            "Education Level": "Education",
            "Job Title": "Job",
        },
        inplace=True,
    )

    for col in ["Age", "Experience", "Salary"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce", downcast="integer")
            .fillna(df[col].mean())
            .abs()
        )
    for col in ["Education", "Job", "Gender"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates().reset_index(drop=True)
    df["log_salary"] = np.log1p(df["Salary"])

    X = df[["Age", "Gender", "Education", "Experience", "Job"]]
    Y = df["log_salary"]
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def predict_salary(model, records):
    """Predict salary (USD) from a DataFrame/records using a fitted pipeline.

    Accepts raw feature rows; returns de-logged salary predictions.
    """
    import pandas as pd

    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    log_preds = model.predict(records)
    return np.expm1(log_preds)
