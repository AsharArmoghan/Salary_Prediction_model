from pyexpat import features

from matplotlib.pyplot import step
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

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
                "onehot",
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )


def create_education_pipline():
    edu_order = [
        "primary",
        "high school",
        "diploma",
        "bachelors",
        "masters",
        "phd",
    ]
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "oridnal",
                OrdinalEncoder(
                    categories=[edu_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )


# both low and high card job feature giving poor results, as job feature is not contributing much to the model, so we will drop it for now and can be used in future for further analysis
# def create_low_card_job_pipeline():
#     return Pipeline(
#         [
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             (
#                 "onehot",
#                 OneHotEncoder(
#                     drop="first",
#                     handle_unknown="infrequent_if_exist",
#                     min_frequency=0.01,
#                     sparse_output=False,
#                 ),
#             ),
#         ]
#     )


def create_high_card_job_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target", TargetEncoder(target_type="auto", smooth=5000)),
        ]
    )


# def create_job_pipeline():

#     return Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             (
#                 "onehot",
#                 OneHotEncoder(
#                     drop="first",
#                     handle_unknown="ignore",
#                     sparse_output=False,
#                 ),
#             ),
#         ]
#     )


def create_preprocessor():
    age_pipeline = create_age_pipeline()
    experience_pipeline = create_experience_pipeline()
    gender_pipeline = create_gender_pipeline()
    education_pipeline = create_education_pipline()
    job_pipeline = create_high_card_job_pipeline()
    preprocessor = ColumnTransformer(
        transformers=[
            ('age', age_pipeline, ["Age"]),
            ('education', education_pipeline, ["Education"]),
            ('gender', gender_pipeline, ["Gender"]),
            ('skewed', experience_pipeline, ["Experience"]),
            ('job', job_pipeline, ["Job"]),
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


def evaluate_model(model, X_test, Y_test):
    Y_predict = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_predict)
    mae = mean_absolute_error(Y_test, Y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_predict)
    return {'predictions': Y_predict, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
