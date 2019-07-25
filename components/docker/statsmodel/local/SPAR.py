# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Csv, String, Bool
import statsmodels.api as sm
import pandas as pd
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.column(Bool(key="timestampIndex", default=False))
@dc.column(String(key="timestampColumn", default="date"))
@dc.column(String(key="labelColumn", default="y"))
@dc.param(
    String(
        key="missing",
        default="none",
        help="Available options are ‘none’, ‘drop’, and ‘raise’.",
    )
)
@dc.param(
    String(
        key="trend",
        default="c",
        help="Whether to include a constant or not. ‘c’ includes constant, ‘nc’ no constant.",
    )
)
@dc.param(String(key="method", default="cmle", help="‘cmle’, ‘mle’"))
@dc.param(
    Int(key="maxiter", default=35, help="The maximum number of function evaluations.")
)
@dc.param(
    Int(key="disp", default=1, help="If True, convergence information is output.")
)
@dc.param(
    Int(
        key="maxlag",
        default=None,
        help="If ic is None, then maxlag is the lag length used in fit.",
    )
)
@dc.output(SklearnModel(key="outputModel"))
def SPAR(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    inputdata = args.inputData
    inputdata = (
        pd.Series(inputdata[args.labelColumn].values, index=inputdata.index)
        if args.timestampIndex
        else pd.Series(
            inputdata[args.labelColumn].values,
            index=inputdata[args.timestampColumn].values,
        )
    )
    arma_mod = sm.tsa.AR(inputdata, missing=args.missing)
    arma_res = arma_mod.fit(
        maxlag=args.maxlag,
        trend=args.trend,
        disp=args.disp,
        method=args.method,
        maxiter=args.maxiter,
    )

    return arma_res


if __name__ == "__main__":
    SPAR()
