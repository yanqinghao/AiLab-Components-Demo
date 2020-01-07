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
        default=[1, 0, 0],
        help="The (p,d,q) order of the model for the number of AR parameters, "
        "differences, and MA parameters.",
    )
)
@dc.param(
    ListOfInt(
        key="seasonalOrder",
        default=[0, 0, 0, 0],
        help="The (P,D,Q,s) order of the seasonal component of the model for the"
        " AR parameters, differences, MA parameters, and periodicity.",
    )
)
@dc.param(
    String(
        key="trend",
        default=None,
        help="Parameter controlling the deterministic trend polynomial A(t)."
        " ‘n’,’c’,’t’,’ct’",
    )
)
@dc.param(
    Bool(
        key="measurementError",
        default=False,
        help="Whether or not to assume the endogenous observations endog were"
        " measured with error.",
    )
)
@dc.param(
    Bool(
        key="timeVaryingRegression",
        default=False,
        help="Used when an explanatory variables, exog, are provided provided to"
        " select whether or not coefficients on the exogenous regressors are"
        " allowed to vary over time.",
    )
)
@dc.param(
    Bool(
        key="mleRegression",
        default=True,
        help="Whether or not to use estimate the regression coefficients for the"
        " exogenous variables as part of maximum likelihood estimation or "
        "through the Kalman filter",
    )
)
@dc.param(
    Int(
        key="trendOffset",
        default=1,
        help="The offset at which to start time trend values.",
    )
)
@dc.param(
    Int(key="disp", default=5, help="If True, convergence information is printed.")
)
@dc.param(
    Int(key="maxiter", default=50, help="The maximum number of function evaluations.")
)
@dc.param(
    String(
        key="method",
        default="lbfgs",
        help="The method determines which solver from scipy.optimize is used "
        "‘newton’, ‘bfgs’, ‘lbfgs’, ‘powell’, ‘cg’, ‘ncg’, ‘basinhopping’",
    )
)
@dc.output(SklearnModel(key="outputModel"))
def SPSARIMAX(context):
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
    order = (args.order[0], args.order[1], args.order[2])
    seasonalOrder = (
        args.seasonalOrder[0],
        args.seasonalOrder[1],
        args.seasonalOrder[2],
        args.seasonalOrder[3],
    )
    arma_mod = sm.tsa.statespace.SARIMAX(
        inputdata,
        order=order,
        seasonal_order=seasonalOrder,
        measurement_error=args.measurementError,
        time_varying_regression=args.mtimeVaryingRegression,
        mle_regression=args.mleRegression,
        trend_offset=args.trendOffset,
        trend=args.trend,
    )
    arma_res = arma_mod.fit(disp=args.disp, maxiter=args.maxiter, method=args.method)

    return arma_res


if __name__ == "__main__":
    SPSARIMAX()
