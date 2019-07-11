# coding=utf-8
from __future__ import absolute_import, print_function

import joblib

from suanpan.components import Result
from suanpan.storage.arguments import Model


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
