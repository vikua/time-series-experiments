from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer

from .dataset import VarType, DatasetConfig


def std_one_hot_pipeline(config: DatasetConfig) -> Dict[str, VarType]:
    features = config.feature_types
    numeric_features = [k for k, v in features.items() if v == VarType.NUM]
    categorical_features = [k for k, v in features.items() if v == VarType.CAT]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("typeconvert", FunctionTransformer(lambda x: x.astype(str))),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="missing", add_indicator=True
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


def std_ord_pipeline(config: DatasetConfig) -> Dict[str, VarType]:
    features = config.feature_types
    numeric_features = [k for k, v in features.items() if v == VarType.NUM]
    categorical_features = [k for k, v in features.items() if v == VarType.CAT]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("typeconvert", FunctionTransformer(lambda x: x.astype(str))),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="missing", add_indicator=True
                ),
            ),
            ("encode", OrdinalEncoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])
