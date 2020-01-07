# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, ListOfString, String, Int
import statsmodels.api as sm
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.column(ListOfString(key="featureColumns", default=["a", "b", "c", "d"]))
@dc.column(String(key="labelColumn", default="e"))
@dc.param(Int(key="rho", default=1, help="Order of the autoregressive covariance"))
@dc.param(
    String(
        key="missing",
        default="none",
        help="Available options are ‘none’, ‘drop’, and ‘raise’.",
    )
)
@dc.param(String(key="method", default="pinv", help="Can be “pinv”, “qr”. "))
@dc.output(SklearnModel(key="outputModel"))
def SPGLSAR(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    df = args.inputData

    featureColumns = args.featureColumns
    labelColumn = args.labelColumn

    features = df[featureColumns].values
    label = df[labelColumn].values

    arma_mod = sm.GLSAR(label, features, rho=args.rho, missing=args.missing)
    arma_res = arma_mod.fit(method=args.method)

    return arma_res


if __name__ == "__main__":
    SPGLSAR()
