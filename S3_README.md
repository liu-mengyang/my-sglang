# S3

## How to run

```
## Fetch git repo
apt update
apt install git -y
git clone -b s3-dev https://github.com/liu-mengyang/my-sglang
cd my-sglang

## Start docker container
docker run -it -d -v /home/chatchat/model_hub/:/models -v /hkust/:/workspace --network=host --ipc=host --gpus all --shm-size 6g --name s3 s3
docker exec -it s3 bash

## Pad vllm
pip uninstall vllm
pip install vllm==0.6.3.post1
cd python
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.3.post1
python3 python_only_dev.py
cd ..
cp vllm_padding/fused_moe.py ./vllm/vllm/model_executor/layers/fused_moe/fused_moe.py
cp vllm_padding/layer.py ./vllm/vllm/model_executor/layers/fused_moe/layer.py

## Start to collect
pip install safetensors
python3 collect_xsum.py
python3 collect_humaneval.py
```