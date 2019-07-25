# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Csv, Float, ListOfFloat, Bool, String
from statsmodels.tsa.arima_process import arma_generate_sample
import statsmodels.api as sm
import numpy as np
import pandas as pd


@dc.param(
    ListOfFloat(
        key="ar",
        default=[0.75, -0.25],
        help="coefficient for autoregressive lag polynomial, including zero lag",
    )
)
@dc.param(
    ListOfFloat(
        key="ma",
        default=[0.65, 0.35],
        help="coefficient for moving-average lag polynomial, including zero lag",
    )
)
@dc.param(Int(key="nsample", default=250, help="length of simulated time series"))
@dc.param(Float(key="sigma", default=1.0, help="standard deviation of noise"))
@dc.param(Int(key="randomSeed", default=12345, help="random seed"))
@dc.param(Bool(key="dateCol", default=True, help="date in dataset"))
@dc.param(
    String(
        key="startDate",
        default="19800131",
        help="The first abbreviated date, for instance, '1965q1' or '1965m1'",
    )
)
@dc.param(String(key="freq", default="M", help="DateOffset"))
@dc.output(Csv(key="outputData"))
def SPARMASample(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    np.random.seed(args.randomSeed)
    arparams = np.array(args.ar)
    maparams = np.array(args.ma)
    nobs = args.nsample
    sample = arma_generate_sample(arparams, maparams, nobs, sigma=args.sigma)
    if args.dateCol:
        dates = pd.date_range(start=args.startDate, periods=nobs, freq=args.freq).values
        sample = pd.DataFrame(sample, columns=["y"])
        sample["date"] = dates
    else:
        sample = pd.DataFrame(sample, columns=["y"])
    return sample


if __name__ == "__main__":
    SPARMASample()
