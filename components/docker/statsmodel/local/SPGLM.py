# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Csv, ListOfString, String, Int
import statsmodels.api as sm
from arguments import SklearnModel


@dc.input(Csv(key="inputData"))
@dc.column(ListOfString(key="featureColumns", default=["a", "b", "c", "d"]))
@dc.column(String(key="labelColumn", default="e"))
@dc.param(
    String(
        key="family",
        default="Gaussian",
        help="The default is Gaussian. Binomial, Gamma, Gaussian, InverseGaussian"
             "NegativeBinomial, Poisson, Tweedie",
    )
)
@dc.param(
    String(
        key="missing",
        default="none",
        help="Available options are ‘none’, ‘drop’, and ‘raise’.",
    )
)
@dc.param(Int(key="maxiter", default=100, help="Default is 100."))
@dc.output(SklearnModel(key="outputModel"))
def SPGLM(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData 数据
    df = args.inputData

    featureColumns = args.featureColumns
    labelColumn = args.labelColumn

    features = df[featureColumns].values
    label = df[labelColumn].values
    family = args.family
    result = getattr(sm.families, family)()

    arma_mod = sm.GLM(label, features, family=result, missing=args.missing)
    arma_res = arma_mod.fit(maxiter=args.maxiter)

    return arma_res


if __name__ == "__main__":
    SPGLM()
