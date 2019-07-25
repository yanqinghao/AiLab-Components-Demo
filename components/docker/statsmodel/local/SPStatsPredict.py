# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, String, Bool
import pandas as pd
import numpy as np
from statsmodels.tsa.base.tsa_model import TimeSeriesResultsWrapper
from statsmodels.tsa.ar_model import ARResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
from statsmodels.tsa.arima_model import ARMAResultsWrapper, ARIMAResultsWrapper
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.discrete.discrete_model import (
    BinaryResultsWrapper,
    MultinomialResultsWrapper,
)
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.input(SklearnModel(key="inputModel"))
@dc.column(String(key="featureColumns", default=["a", "b", "c", "d"]))
@dc.column(String(key="predictColumn", default="prediction"))
@dc.param(String(key="start", default="2000-11-30"))
@dc.param(String(key="end", default="2001-05-31"))
@dc.param(Bool(key="dynamic", default=True))
@dc.output(Csv(key="outputData"))
def SPStatsPredict(context):
    args = context.args

    model = args.inputModel
    if isinstance(
        model,
        (
            ARResultsWrapper,
            ARMAResultsWrapper,
            ARIMAResultsWrapper,
            SARIMAXResultsWrapper,
        ),
    ):
        print("Time series model loaded")
        start = args.start
        end = args.end
        dynamic = args.dynamic
        predictions = model.predict(start, end, dynamic=dynamic)
        res = pd.DataFrame(predictions.values, columns=[args.predictColumn])
        res["date"] = predictions.index
        print(res)
    elif isinstance(model, RegressionResultsWrapper):
        print("Linear regression model loaded")
        res = args.inputData
        featureColumns = args.featureColumns
        predictColumn = args.predictColumn
        X = res[featureColumns].values
        predictions = model.predict(X)
        print(predictions)
        if isinstance(model, BinaryResultsWrapper):
            res[predictColumn] = np.around(predictions)
        elif isinstance(model, MultinomialResultsWrapper):
            res[predictColumn] = np.argmax(predictions, axis=1)
        else:
            res[predictColumn] = predictions
    else:
        print("Wrong input model.")
        return None

    return res


if __name__ == "__main__":
    SPStatsPredict()  # pylint: disable=no-value-for-parameter
