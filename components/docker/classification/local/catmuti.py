# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import (
    Int,
    String,
    Csv,
    Model,
    Bool,
    Float,
    ListOfString,
    Table,
)
import pandas as pd
import os
from catboost import CatBoostClassifier
import joblib
from suanpan.components import Result


class SklearnModel(Model):
    FILETYPE = "model"

    def format(self, context):
        super(SklearnModel, self).format(context)
        if self.filePath:
            self.value = joblib.load(self.filePath)

        return self.value

    def save(self, context, result):
        joblib.dump(result.value, self.filePath)

        return super(SklearnModel, self).save(
            context, Result.froms(value=self.filePath)
        )


# 定义输入
@dc.input(Csv(key="inputData", required=True))
@dc.column(ListOfString(key="featureColumns", default=["a", "b", "c", "d"]))
@dc.column(String(key="labelColumn", default="e"))
@dc.param(
    Int(
        key="iterations",
        default=1000,
        help="The maximum number of trees that can be built when solving machine learning problems.",
    )
)
@dc.param(Float(key="learningRate", default=0.03, help="The learning rate."))
@dc.param(Int(key="depth", default=6, help="Depth of the tree."))
@dc.param(
    Float(
        key="l2LeafReg",
        default=3.0,
        help="Coefficient at the L2 regularization term of the cost function.",
    )
)
@dc.param(
    Float(
        key="rsm",
        default=1,
        help="Random subspace method. The percentage of features to use at each split selection, when features are selected over again at random.",
    )
)
@dc.param(
    String(
        key="lossFunction",
        default="MultiClass",
        help="The metric to use in training. MultiClass,MultiClassOneVsAll",
    )
)
@dc.param(
    Int(
        key="odWait",
        default=20,
        help="The number of iterations to continue the training after the iteration with the optimal metric value.",
    )
)
@dc.param(
    String(
        key="odType",
        default="IncToDec",
        help="The type of the overfitting detector to use.IncToDec,Iter",
    )
)
@dc.param(Int(key="randomSeed", default=0, help="The random seed used for training."))
@dc.param(
    Int(
        key="metricPeriod",
        default=1,
        help="The frequency of iterations to calculate the values of objectives and metrics. ",
    )
)
@dc.param(Bool(key="useBestModel", default=True, help="Use Best Model."))
@dc.param(
    Float(
        key="baggingTemperature",
        default=1,
        help="Defines the settings of the Bayesian bootstrap. ",
    )
)
@dc.param(
    String(
        key="evalMetric",
        default="LossFunction",
        help="The metric used for overfitting detection and best model selection. Accuracy,HingeLoss,LossFunction",
    )
)
@dc.param(Bool(key="needTrain", default=True))

# 定义输出
@dc.output(SklearnModel(key="outputModel"))
def catBoostCLFMuti(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData1 数据

    df = args.inputData

    featureColumns = args.featureColumns
    labelColumn = args.labelColumn

    features = df[featureColumns].values
    label = df[labelColumn].values

    iterations = args.iterations
    learningRate = args.learningRate
    depth = args.depth
    l2LeafReg = args.l2LeafReg
    rsm = args.rsm
    lossFunction = args.lossFunction
    odWait = args.odWait
    odType = args.odType
    randomSeed = args.randomSeed
    evalMetric = (
        args.lossFunction if args.evalMetric == "LossFunction" else args.evalMetric
    )
    useBestModel = args.useBestModel
    baggingTemperature = args.baggingTemperature
    metricPeriod = args.metricPeriod

    catclf = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learningRate,
        depth=depth,
        l2_leaf_reg=l2LeafReg,
        rsm=rsm,
        loss_function=lossFunction,
        od_wait=odWait,
        od_type=odType,
        random_seed=randomSeed,
        metric_period=metricPeriod,
        eval_metric=evalMetric,
        use_best_model=useBestModel,
        bagging_temperature=baggingTemperature,
    )
    if args.needTrain:
        catclf.fit(features, label, eval_set=(features, label))

    return catclf


if __name__ == "__main__":
    catBoostCLFMuti()
