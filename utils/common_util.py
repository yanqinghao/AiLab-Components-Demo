# coding=utf-8
from __future__ import absolute_import, print_function

from generic_model import GenericModel


def validateArgs(df, featureColumns, labelColumns, needTrain):
    result = df is not None and featureColumns and labelColumns if needTrain else True
    if not result:
        raise RuntimeError(
            "If estimator need to train, train data, feature columns and label column(s) are required."
        )


def getXY(df, featureColumns, labelColumns, needTrain):
    validateArgs(df, featureColumns, labelColumns, needTrain)
    return (df[featureColumns], df[labelColumns]) if needTrain else (None, None)


def encapsulateModel(model):
    if not isinstance(model, GenericModel):
        model = GenericModel.fromModel(model)

    return model


def unpackModel(model):
    if isinstance(model, GenericModel):
        model = model.model

    return model
