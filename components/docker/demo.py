# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, Table
import utils


# 定义输入
# @dc.input(Folder(key="inputData1", required=True))
@dc.input(Table(key="inputData1", table="inputTable1", partition="inputPartition1", required=True))
# 定义输出
@dc.output(Table(key="outputData1", table="outputTable1", partition="outputPartition1", required=True))
def Demo(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData1 数据
    print(args.inputData1)
    a = args.inputData1

    # 自定义代码
    utils.hello()

    # 将 args.outputData1 作为输出发送给下一节点
    return a


if __name__ == "__main__":
    suanpan.run(Demo)
