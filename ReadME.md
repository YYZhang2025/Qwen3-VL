<h1 align='center'>✨Qwen3-VL✨ From Scratch</h1>

In the repository, you will find the inference code for Qwen3-VL. 


## Environment Setup

```bash
uv sync 
source .venv/bin/activate
uv pip install -e . 
```


Here is the example:





https://github.com/user-attachments/assets/2e5ff59f-2a96-4b5d-9e99-12d63ef0f034



```Bash
docker run --rm -it \
  -p 7860:7860 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  qwen3vl:latest
```
