NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/mllib_components:$1 -f docker/docker_yanqing/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/mllib_components_stream:$1 -f docker/stream_yanqing/Dockerfile .

    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/mllib_components:$1
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/mllib_components_stream:$1
done
