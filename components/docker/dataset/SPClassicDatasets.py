# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, String
from suanpan.storage import storage

DATESET_PATH_PREFIX = "common/data"


@dc.param(
    String(
        key="dataset",
        required=True,
        help="allowed values: ['boston_housing', 'breast_cancer', 'california_housing', "
        "'covertype', 'diabetes', 'digits', 'iris', 'kddcup', 'linnerud', 'wine', 'titanic'"
        ", 'sun_spots', 'macrodata']",
    )
)
@dc.output(Folder(key="outputDir"))
def SPClassicDatasets(context):
    args = context.args

    remotePath = storage.storagePathJoin(DATESET_PATH_PREFIX, args.dataset)
    storage.download(remotePath, args.outputDir)

    return args.outputDir


if __name__ == "__main__":
    SPClassicDatasets()  # pylint: disable=no-value-for-parameter
