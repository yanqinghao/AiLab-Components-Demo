# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv
import statsmodels.api as sm


@dc.output(Csv(key="outputData"))
def SPMacroData(context):
    dta = sm.datasets.macrodata.load_pandas().data

    return dta


if __name__ == "__main__":
    SPMacroData()
