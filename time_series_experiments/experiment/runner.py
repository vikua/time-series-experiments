import attr
import pandas as pd
import tensorflow as tf

from ..pipeline import Pipeline
from ..pipeline.validation import BacktestingCrossVal
from ..pipeline.data import to_task_data
from ..pipeline.sequences import TimeSeriesSequence
from ..pipeline.dataset import DatasetConfig


@attr.s
class TrainingParams(object):
    fdw: int = attr.ib()
    fw: int = attr.ib()
    epochs: int = attr.ib()
    batch_size: int = attr.ib()


class ExperimentRunner(object):
    def __init__(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
        cross_val: BacktestingCrossVal,
        preprocessing_pipeline: Pipeline,
        model: tf.keras.Model,
        training_params: TrainingParams,
    ):
        self._df = df
        self._config = config
        self._cross_val = cross_val
        self._preprocessing_pipeline = preprocessing_pipeline
        self._model = model
        self._training_params = training_params

        self._y = df.pop(self._config.target_col).values

    def run_backtesting(self):
        results = dict()
        for bt in self._cross_val:
            X_train = pd.DataFrame(
                self._df.values[bt.train_index], columns=self._df.columns
            )
            y_train = self._y[bt.train_index]
            X_test = pd.DataFrame(
                self._df.values[bt.test_index], columns=self._df.columns
            )
            y_test = self._y[bt.test_index]

            results[bt.backtest_number] = self.run(X_train, y_train, X_test, y_test)

        return results

    def run(self, X_train, y_train, X_test, y_test):
        train = to_task_data(X_train, y_train, self._config)
        test = to_task_data(X_test, y_test, self._config)

        train = self._preprocessing_pipeline.fit_transform(train)
        test = self._preprocessing_pipeline.transform(test)

        train_seq = TimeSeriesSequence(
            train,
            fdw=self._training_params.fdw,
            fw=self._training_params.fw,
            batch_size=self._training_params.batch_size,
        )
        test_seq = TimeSeriesSequence(
            test,
            fdw=self._training_params.fdw,
            fw=self._training_params.fw,
            batch_size=self._training_params.batch_size,
        )

        history = self._model.fit(
            train_seq,
            validation_data=test_seq,
            epochs=self._training_params.epochs,
            shuffle=True,
            workers=1,
            verbose=0,
        )
        return history
