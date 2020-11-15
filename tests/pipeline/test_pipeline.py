import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from time_series_experiments.pipeline import Pipeline, Step, ColumnsProcessor
from time_series_experiments.pipeline.tasks import Wrap, OrdCat, DateFeatures
from time_series_experiments.pipeline.data import to_task_data, ColumnType, take_columns
from time_series_experiments.pipeline.dataset import VarType, DatasetConfig

from ..conftest import RANDOM_SEED


def create_dataset(num=1, cat=1, date=0, target=True, size=1000, seed=RANDOM_SEED):
    random_state = np.random.RandomState(seed)

    X = []
    colnames = []
    for i in range(num):
        val = random_state.random((size, 1))
        nans = random_state.choice(val.shape[0], size=int(size * 0.02))
        val[nans] = np.nan
        X.append(val)
        colnames.append("num_{}".format(i))

    for i in range(cat):
        colnames.append("cat_{}".format(i))
        card = random_state.randint(low=10, high=100)
        val = random_state.randint(low=1, high=card, size=size).astype(np.float)

        stringlify = random_state.random() > 0.5
        if stringlify:
            f = np.vectorize(lambda x: "cat_{}_{}".format(i, x))
            val = f(val)

        nans = random_state.choice(val.shape[0], size=int(size * 0.02))
        val[nans] = np.nan

        X.append(np.reshape(val, (-1, 1)))

    for i in range(date):
        colnames.append("date_{}".format(i))
        dates = pd.date_range(start="2000-01-01", freq="1H", periods=size * 100)
        dates = random_state.choice(dates, size=size)
        dates = np.reshape(dates, (-1, 1))
        X.append(dates)

    if target:
        X.append(random_state.random((size, 1)))
        colnames.append("target")

    X = np.concatenate(X, axis=1)
    return pd.DataFrame(X, columns=colnames)


def test_pipeline_numeric():
    df = create_dataset(num=10, cat=0, size=5000)

    train_df = df.iloc[:-1000]
    y_train = train_df.pop("target").values

    test_df = df.iloc[-1000:]
    y_test = test_df.pop("target").values

    num_pipeline = Pipeline(
        steps=[
            Step(name="impute", task=Wrap(SimpleImputer(strategy="mean"))),
            Step(
                name="standatize",
                task=Wrap(StandardScaler(), type_override=VarType.NUM),
            ),
        ]
    )

    train = num_pipeline.fit_transform(to_task_data(train_df, y_train))
    test = num_pipeline.transform(to_task_data(test_df, y_test))

    assert all([x == ColumnType(VarType.NUM) for x in train.column_types])
    assert all([x == ColumnType(VarType.NUM) for x in test.column_types])

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    res = imputer.fit_transform(train_df.values)
    train_expected = scaler.fit_transform(res)
    res = imputer.transform(test_df.values)
    test_expected = scaler.transform(res)

    assert np.all(np.isclose(train_expected, train.X))
    assert np.all(np.isclose(test_expected, test.X))


def test_pipeline_cat():
    df = create_dataset(num=0, cat=10, size=5000)

    train_df = df.iloc[:-1000]
    y_train = train_df.pop("target").values

    test_df = df.iloc[-1000:]
    y_test = test_df.pop("target").values

    cat_pipeline = Pipeline(
        steps=[
            Step(
                name="impute",
                task=Wrap(SimpleImputer(strategy="constant", fill_value="missing")),
            ),
            Step(name="ordcat", task=OrdCat()),
        ]
    )

    train = cat_pipeline.fit_transform(to_task_data(train_df, y_train))
    test = cat_pipeline.transform(to_task_data(test_df, y_test))

    assert all([x.var_type == VarType.CAT and x.level > 0 for x in train.column_types])
    assert all([x.var_type == VarType.CAT and x.level > 0 for x in test.column_types])
    assert all([x1 == x2 for x1, x2 in zip(train.column_types, test.column_types)])

    imputer = SimpleImputer(strategy="constant", fill_value="missing")
    ordcat = OrdinalEncoder()
    res = imputer.fit_transform(train_df.values)
    train_expected = ordcat.fit_transform(res)
    res = imputer.transform(test_df.values)
    test_expected = ordcat.transform(res)

    assert np.all(np.isclose(train_expected, train.X))
    assert np.all(np.isclose(test_expected, test.X))


def test_column_proccessing_by_name():
    df = create_dataset(num=5, cat=10, size=5000)

    num_features = [x for x in df.columns if x.startswith("num_")]
    cat_features = [x for x in df.columns if x.startswith("cat_")]

    train_df = df.iloc[:-1000]
    y_train = train_df.pop("target").values

    test_df = df.iloc[-1000:]
    y_test = test_df.pop("target").values

    num_pipeline = Pipeline(
        steps=[
            Step(name="impute", task=Wrap(SimpleImputer(strategy="mean"))),
            Step(name="standatize", task=Wrap(StandardScaler())),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            Step(
                name="impute",
                task=Wrap(SimpleImputer(strategy="constant", fill_value="missing")),
            ),
            Step(name="ordcat", task=OrdCat()),
        ]
    )
    branch_proc = ColumnsProcessor(
        branches=[
            Step("num", num_pipeline, num_features),
            Step("cat", cat_pipeline, cat_features),
        ]
    )

    _verify_pipeline(
        train_df,
        y_train,
        test_df,
        y_test,
        branch_proc,
        num_pipeline,
        cat_pipeline,
        num_features,
        cat_features,
    )


def _verify_pipeline(
    train_df,
    y_train,
    test_df,
    y_test,
    branch_proc,
    num_pipeline,
    cat_pipeline,
    num_features,
    cat_features,
    config=None,
):
    train = branch_proc.fit_transform(to_task_data(train_df, y_train, config))
    test = branch_proc.transform(to_task_data(test_df, y_test, config))

    assert all([x.level == 0 for x in train.column_types[:5]])
    assert all([x.level == 0 for x in test.column_types[:5]])
    assert all([x.level > 0 for x in train.column_types[5:]])
    assert all([x.level > 0 for x in test.column_types[5:]])

    assert all([x.startswith("num_") for x in train.column_names[:5]])
    assert all([x.startswith("num_") for x in test.column_names[:5]])
    assert all([x.startswith("cat_") for x in train.column_names[5:]])
    assert all([x.startswith("cat_") for x in test.column_names[5:]])

    num_train_exp = num_pipeline.fit_transform(
        to_task_data(train_df[num_features], y_train, config)
    )
    num_test_exp = num_pipeline.transform(
        to_task_data(test_df[num_features], y_test, config)
    )

    cat_train_exp = cat_pipeline.fit_transform(
        to_task_data(train_df[cat_features], y_train, config)
    )
    cat_test_exp = cat_pipeline.transform(
        to_task_data(test_df[cat_features], y_test, config)
    )

    assert train.column_names == (
        num_train_exp.column_names + cat_train_exp.column_names
    )
    assert test.column_names == (num_test_exp.column_names + cat_test_exp.column_names)
    assert train.column_types == (
        num_train_exp.column_types + cat_train_exp.column_types
    )
    assert test.column_types == (num_test_exp.column_types + cat_test_exp.column_types)

    X_exp_train = np.concatenate([num_train_exp.X, cat_train_exp.X], axis=1)
    X_exp_test = np.concatenate([num_test_exp.X, cat_test_exp.X], axis=1)
    assert np.all(np.isclose(train.X, X_exp_train))
    assert np.all(np.isclose(test.X, X_exp_test))


def test_column_proccessing_by_type():
    df = create_dataset(num=5, cat=10, size=5000)

    train_df = df.iloc[:-1000]
    y_train = train_df.pop("target").values

    test_df = df.iloc[-1000:]
    y_test = test_df.pop("target").values

    num_pipeline = Pipeline(
        steps=[
            Step(name="impute", task=Wrap(SimpleImputer(strategy="mean"))),
            Step(name="standatize", task=Wrap(StandardScaler())),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            Step(
                name="impute",
                task=Wrap(SimpleImputer(strategy="constant", fill_value="missing")),
            ),
            Step(name="ordcat", task=OrdCat()),
        ]
    )
    branch_proc = ColumnsProcessor(
        branches=[
            Step("num", num_pipeline, types=[VarType.NUM]),
            Step("cat", cat_pipeline, types=[VarType.CAT]),
        ]
    )

    num_features = [x for x in df.columns if x.startswith("num_")]
    cat_features = [x for x in df.columns if x.startswith("cat_")]
    feature_types = {x: VarType.NUM for x in num_features}
    feature_types.update({x: VarType.CAT for x in cat_features})
    config = DatasetConfig("", "", "", "", feature_types=feature_types)

    _verify_pipeline(
        train_df,
        y_train,
        test_df,
        y_test,
        branch_proc,
        num_pipeline,
        cat_pipeline,
        num_features,
        cat_features,
        config=config,
    )


def test_date_pipeline():
    df = create_dataset(num=0, cat=0, date=5, target=False, size=5000)
    train_df = df.iloc[:-1000]
    test_df = df.iloc[-1000:]

    date_pipeline = Pipeline(
        steps=[
            Step("date", DateFeatures()),
            Step(
                "derived_processing",
                ColumnsProcessor(
                    branches=[
                        Step(
                            "num_derived", Wrap(StandardScaler()), types=[VarType.NUM]
                        ),
                        Step("cat_derived", OrdCat(), types=[VarType.CAT]),
                    ]
                ),
            ),
        ]
    )
    train = date_pipeline.fit_transform(to_task_data(train_df))
    test = date_pipeline.transform(to_task_data(test_df))

    for data in [train, test]:
        assert data.column_types[:5] == [ColumnType(VarType.NUM)] * 5
        assert set([c.var_type for c in data.column_types[5:]]) == set([VarType.CAT])
        assert all([c.level > 0 for c in data.column_types[5:]])

    date_features = DateFeatures()
    dates_train = date_features.fit_transform(to_task_data(train_df))
    dates_test = date_features.transform(to_task_data(test_df))

    num_train = take_columns(dates_train, types=[VarType.NUM])
    cat_train = take_columns(dates_train, types=[VarType.CAT])
    scaler = StandardScaler()
    enc = OrdinalEncoder()

    num_train = scaler.fit_transform(num_train.X)
    cat_train = enc.fit_transform(cat_train.X)

    assert np.all(np.isclose(num_train, take_columns(train, types=[VarType.NUM]).X))
    assert np.all(np.isclose(cat_train, take_columns(train, types=[VarType.CAT]).X))

    num_test = take_columns(dates_test, types=[VarType.NUM])
    cat_test = take_columns(dates_test, types=[VarType.CAT])
    num_test = scaler.transform(num_test.X)
    cat_test = enc.transform(cat_test.X)

    assert np.all(np.isclose(num_test, take_columns(test, types=[VarType.NUM]).X))
    assert np.all(np.isclose(cat_test, take_columns(test, types=[VarType.CAT]).X))
