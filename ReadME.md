<h1 align='center'>✨Qwen3-VL✨ From Scratch</h1>

In the repository, you will find the inference code for Qwen3-VL. 


## Environment Setup

```bash
uv sync 
source .venv/bin/activate
uv pip install -e . 
```


Here is the example:

<video src='https://github.com/YYZhang2025/Qwen3-VL/blob/main/assets/demo-cli.mp4' controls='controls' style='max-width: 100%; height: auto;' ></video>




```Bash
docker run --rm -it \
  -p 7860:7860 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  qwen3vl:latest
```