# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, ListOfString, String, Int
import statsmodels.api as sm
import statsmodels
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.column(ListOfString(key="featureColumns", default=["a", "b", "c", "d"]))
@dc.column(String(key="labelColumn", default="e"))
@dc.param(
    String(
        key="M",
        default="HuberT",
        help="The default is LeastSquares. HuberT, RamsayE, AndrewWave, TrimmedMean"
             "Hampel, TukeyBiweight",
    )
)
@dc.param(
    String(
        key="missing",
        default="none",
        help="Available options are ‘none’, ‘drop’, and ‘raise’.",
    )
)
@dc.param(Int(key="maxiter", default=50, help="The maximum number of iterations to try."))
@dc.output(SklearnModel(key="outputModel"))
def SPRLM(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    df = args.inputData

    featureColumns = args.featureColumns
    labelColumn = args.labelColumn

    features = df[featureColumns].values
    label = df[labelColumn].values
    M = args.M
    result = getattr(statsmodels.robust.norms, M)()

    arma_mod = sm.RLM(label, features, M=result, missing=args.missing)
    arma_res = arma_mod.fit(maxiter=args.maxiter)

    return arma_res


if __name__ == "__main__":
    SPRLM()
