import torch

from src.model.qwen3vl import Qwen3VL
from src.processors.preprocessor import Processor

if __name__ == "__main__":
    model = Qwen3VL.from_pretrained("./checkpoints/Qwen3VL-2B-Instruct")
    print("Model loaded successfully.")

    messages = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image",
                #     "url": "./assets/cat.jpg",
                # },
                {"type": "text", "text": "What animal is this?"},
            ],
        }
    ]
    processor = Processor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    inputs = processor(messages, device=next(model.parameters()).device)
    print("Inputs prepared successfully.")

    prompt_len = inputs["input_ids"].shape[1]
    for _ in range(50):
        outputs = model.generate(**inputs, max_new_tokens=50)
        new_token = outputs[:, -1:]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], new_token], dim=1)

        generated_ids = outputs[0, prompt_len:]  # only the new part
        generated_ids = generated_ids.detach().cpu().numpy()
        print("Generated output:", processor.tokenizer.decode(generated_ids, skip_special_tokens=True))
