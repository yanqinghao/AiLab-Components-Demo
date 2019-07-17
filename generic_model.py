# coding=utf-8
from __future__ import absolute_import, print_function

from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
from sklearn.mixture.base import BaseMixture
from catboost import CatBoostRegressor, CatBoostClassifier

from suanpan.log import logger
from suanpan.model import Model


class GenericModel(Model):
    def __init__(self):
        super(GenericModel, self).__init__()
        self.needTrain = None
        self.needCrossVal = None
        self.cvInstance = None
        self.modelInstance = None

    @classmethod
    def createModel(cls, modelClass, **kwargs):
        return cls().createModelInstance(modelClass, **kwargs)

    @classmethod
    def fromModel(cls, modelInstance):
        return cls().setModelInstance(modelInstance)

    def train(self, X, y=None):
        if self.needTrain:
            if self.needCrossVal and self.cvInstance:
                maxScore = 0
                logger.info("Using cross validation...")
                for trainIndex, testIndex in self.cvInstance.split(X):
                    xTrain, xTest = X[trainIndex], X[testIndex]
                    yTrain, yTest = y[trainIndex], y[testIndex]
                    model = self.modelInstance
                    model.fit(xTrain, yTrain)
                    score = model.score(xTest, yTest)
                    if score > maxScore:
                        xTrainMax = xTrain
                        yTrainMax = yTrain
                self.modelInstance.fit(xTrainMax, yTrainMax)
            else:
                self.modelInstance.fit(X, y)

    def predict(self, X):
        return self._predict(X) if self.isEstimator() else self._transform(X)

    @property
    def model(self):
        return self.modelInstance

    def setModelInstance(self, modelInstance):
        self.modelInstance = modelInstance
        return self

    def createModelInstance(self, modelClass, **kwargs):
        self.needTrain = kwargs.pop("needTrain", True)
        self.needCrossVal = kwargs.pop("needCrossVal", False)
        self.cvInstance = kwargs.pop("cvInstance", None)
        self.modelInstance = modelClass(**kwargs)
        return self

    def isEstimator(self):
        return isinstance(
            self.modelInstance,
            (
                ClassifierMixin,
                RegressorMixin,
                ClusterMixin,
                BaseMixture,
                CatBoostRegressor,
                CatBoostClassifier,
            ),
        )

    def isTransformer(self):
        return isinstance(self.model, TransformerMixin)

    def _predict(self, X):
        logger.info("model is an estimator, use predict()")
        predictions = self.modelInstance.predict(X)
        labelCount = 1 if len(predictions.shape) == 1 else predictions.shape[1]
        predictions = (
            predictions.T
            if labelCount > 1
            else predictions.reshape(1, len(predictions))
        )

        return predictions

    def _transform(self, X):
        logger.info("model is an transformer, use transform()")
        return self.modelInstance.transform(X)
