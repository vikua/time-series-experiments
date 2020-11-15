from typing import Dict

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from ._pipeline import Step, Pipeline, BranchProcessor
from .tasks import Wrap, OrdCat
from .dataset import VarType, DatasetConfig


def std_ord_pipeline(config: DatasetConfig) -> Dict[str, VarType]:
    features = config.feature_types
    numeric_features = [k for k, v in features.items() if v == VarType.NUM]
    categorical_features = [k for k, v in features.items() if v == VarType.CAT]

    numeric_pipeline = Pipeline(
        steps=[
            Step("impute", Wrap(SimpleImputer(strategy="mean"))),
            Step("scale", Wrap(StandardScaler())),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            Step(
                "imputer",
                Wrap(
                    SimpleImputer(
                        strategy="constant", fill_value="missing", add_indicator=True
                    )
                ),
            ),
            Step("encode", OrdCat()),
        ]
    )
    preprocessor = BranchProcessor(
        branches=[
            Step("num", numeric_pipeline, numeric_features),
            Step("cat", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(steps=[Step("preprocessor", preprocessor)])
