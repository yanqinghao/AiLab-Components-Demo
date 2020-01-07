# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Int, Csv, ListOfInt, String, Bool
import statsmodels.api as sm
import pandas as pd
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.column(Bool(key="timestampIndex", default=False))
@dc.column(String(key="timestampColumn", default="date"))
@dc.column(String(key="labelColumn", default="y"))
@dc.param(
    ListOfInt(
        key="order",
        default=[2, 0],
        help="The (p,q) order of the model for the number of AR parameters, differences, and MA parameters to use.",
    )
)
@dc.param(
    String(
        key="trend",
        default="c",
        help="Whether to include a constant or not. ‘c’ includes constant, ‘nc’ no constant.",
    )
)
@dc.param(
    String(
        key="method",
        default="css-mle",
        help="This is the loglikelihood to maximize.‘css-mle’,’mle’,’css’",
    )
)
@dc.param(
    Int(key="maxiter", default=500, help="The maximum number of function evaluations.")
)
@dc.param(
    Int(key="disp", default=5, help="If True, convergence information is printed.")
)
@dc.output(SklearnModel(key="outputModel"))
def SPARMA(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    inputdata = args.inputData
    inputdata = (
        pd.DataFrame(inputdata[args.labelColumn].values, index=inputdata.index)
        if args.timestampIndex
        else pd.DataFrame(
            inputdata[args.labelColumn].values,
            index=inputdata[args.timestampColumn].values,
        )
    )
    print(inputdata)
    order = (args.order[0], args.order[1])
    arma_mod = sm.tsa.ARMA(inputdata, order=order)
    arma_res = arma_mod.fit(
        trend=args.trend, disp=args.disp, maxiter=args.maxiter, method=args.method
    )

    return arma_res


if __name__ == "__main__":
    SPARMA()
