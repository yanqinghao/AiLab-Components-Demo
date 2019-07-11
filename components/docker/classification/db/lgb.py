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
import lightgbm as lgb
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


@dc.input(
    Table(
        key="inputData", table="inputTable", partition="inputPartition", required=True
    )
)
@dc.column(ListOfString(key="featureColumns", default=["f1", "f2", "f3", "f4"]))
@dc.column(String(key="labelColumn", default="label"))
@dc.param(Int(key="maxDepth", default=-1, help="Maximum tree depth for base learners"))
@dc.param(
    String(
        key="boostingType",
        default="gbdt",
        help="Specify which booster to use: 'goss', 'rf' or 'dart'",
    )
)
@dc.param(
    Int(key="numLeaves", default=31, help="Maximum tree leaves for base learners.")
)
@dc.param(Float(key="learningRate", default=0.1, help="Boosting learning rate."))
@dc.param(Int(key="nEstimators", default=100, help="Number of boosted trees to fit."))
@dc.param(
    Int(
        key="subsampleForBin",
        default=200000,
        help="Number of samples for constructing bins.",
    )
)
@dc.param(
    Float(
        key="minSplitGain",
        default=0.0,
        help="Minimum loss reduction required to make a further partition on a leaf node of the tree.",
    )
)
@dc.param(
    Float(
        key="minChildWeight",
        default=1e-3,
        help="Minimum sum of instance weight (hessian) needed in a child (leaf).",
    )
)
@dc.param(
    Int(
        key="minChildSamples",
        default=20,
        help="Minimum number of data needed in a child (leaf).",
    )
)
@dc.param(
    Float(
        key="subsample", default=1.0, help="Subsample ratio of the training instance."
    )
)
@dc.param(
    Int(
        key="subsampleFreq",
        default=0,
        help="Frequence of subsample, <=0 means no enable.",
    )
)
@dc.param(
    Float(
        key="colsampleBytree",
        default=1.0,
        help="Subsample ratio of columns when constructing each tree.",
    )
)
@dc.param(Float(key="regAlpha", default=0.0, help="L1 regularization term on weights."))
@dc.param(
    Float(key="regLambda", default=0.0, help="L2 regularization term on weights.")
)
@dc.param(Int(key="randomState", default=0, help="Random number seed."))
@dc.param(Int(key="nJobs", default=-1, help="Number of parallel threads."))
@dc.param(
    Bool(
        key="silent",
        default=True,
        help="Whether to print messages while running boosting.",
    )
)
@dc.param(Bool(key="needTrain", default=True))
@dc.output(SklearnModel(key="outputModel"))
def LightGBMClf(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据

    df = args.inputData

    featureColumns = args.featureColumns
    labelColumn = args.labelColumn

    features = df[featureColumns].values
    label = df[labelColumn].values

    maxDepth = args.maxDepth
    boostingType = args.boostingType
    numLeaves = args.numLeaves
    learningRate = args.learningRate
    nEstimators = args.nEstimators
    subsampleForBin = args.subsampleForBin
    minSplitGain = args.minSplitGain
    minChildWeight = args.minChildWeight
    minChildSamples = args.minChildSamples
    subsample = args.subsample
    subsampleFreq = args.subsampleFreq
    colsampleBytree = args.colsampleBytree
    regAlpha = args.regAlpha
    regLambda = args.regLambda
    randomState = args.randomState
    nJobs = args.nJobs
    silent = args.silent

    gbm = lgb.LGBMClassifier(
        boosting_type=boostingType,
        n_estimators=nEstimators,
        num_leaves=numLeaves,
        colsample_bytree=colsampleBytree,
        subsample=subsample,
        max_depth=maxDepth,
        reg_alpha=regAlpha,
        reg_lambda=regLambda,
        min_split_gain=minSplitGain,
        n_jobs=nJobs,
        silent=silent,
        min_child_weight=minChildWeight,
        learning_rate=learningRate,
        subsample_for_bin=subsampleForBin,
        min_child_samples=minChildSamples,
        subsample_freq=subsampleFreq,
        random_state=randomState,
    )
    if args.needTrain:
        gbm.fit(features, label)

    return gbm


# if __name__ == "__main__":
#     LightGBMClf()
