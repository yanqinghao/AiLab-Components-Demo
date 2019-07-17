# coding=utf-8
from __future__ import absolute_import, print_function

from arguments import SklearnModel
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, ListOfString
from utils import common_util


def validatePredictColumns(predictColumns, actualColumnCount):
    if not predictColumns:
        predictColumns = [
            "prediction_{}".format(str(i)) for i in range(actualColumnCount)
        ]

    if len(predictColumns) != actualColumnCount:
        raise RecursionError(
            "Actual predict column count is: {}, but the length of predict columns given is: {}".format(
                actualColumnCount, len(predictColumns)
            )
        )

    return predictColumns


@dc.input(Csv(key="inputData"))
@dc.input(SklearnModel(key="inputModel"))
@dc.column(ListOfString(key="featureColumns", default=[]))
@dc.column(ListOfString(key="predictColumns", default="prediction"))
@dc.output(Csv(key="outputData"))
def SPPredict(context):
    args = context.args

    df = args.inputData

    model = common_util.encapsulateModel(args.inputModel)
    featureColumns = args.featureColumns
    predictColumns = args.predictColumns

    X = df[featureColumns].values if len(featureColumns)>0 else df.values

    predictions = model.predict(X)
    isEstimator = model.isEstimator()

    if isEstimator:
        predictColumnCount = predictions.shape[0]
        predictColumns = validatePredictColumns(predictColumns, predictColumnCount)

        for index in range(predictColumnCount):
            predictColumn = predictColumns[index]
            prediction = predictions[index]
            df[predictColumn] = prediction
    else:
        df[featureColumns] = predictions

    return df


if __name__ == "__main__":
    SPPredict()  # pylint: disable=no-value-for-parameter
