import os

import torch

from src.model.qwen3vl import Qwen3VL
from src.processors.preprocessor import Processor


def build_user_message(text: str, image_path: str | None = None):
    """
    Build a single user message in the same format as your current run.py.

    If image_path is provided, it is added first as a vision content block.
    """
    content = []
    if image_path is not None:
        # You can also convert to absolute path if you like
        content.append(
            {
                "type": "image",
                "url": image_path,
            }
        )
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def stream_assistant_reply(model, processor, messages, device, max_new_tokens=256):
    """
    Given the full message history, call processor with add_generation_prompt=True,
    stream the assistant reply, and return the final decoded string.
    """
    inputs = processor(
        messages,
        add_generation_prompt=True,
        device=device,
    )

    generated_token_ids = []

    print("Assistant: ", end="", flush=True)

    # stream tokens
    for token in model.generate_stream(**inputs, max_new_tokens=max_new_tokens):
        # token is typically a scalar tensor
        if isinstance(token, torch.Tensor):
            token_id = token.item()
        else:
            token_id = int(token)

        generated_token_ids.append(token_id)

        # incremental decode (we keep special tokens off the screen)
        piece = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)

    print()  # newline after streaming

    # full decoded text for storing in history
    assistant_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return assistant_text, generated_token_ids


def main():
    # 1. Load model & processor
    model = Qwen3VL.from_pretrained("./checkpoints/Qwen3VL-2B-Instruct")
    print("Model loaded successfully.")

    processor = Processor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    device = next(model.parameters()).device
    print("Processor loaded successfully. Device:", device)

    # 2. Conversation history (list of messages)
    messages: list[dict] = []

    print(
        "\nQwen3-VL Chat CLI\n"
        "---------------------------------\n"
        "Commands:\n"
        "  /exit                quit\n"
        "  /image PATH          send a message with an image\n"
        "\nExamples:\n"
        "  You: Hello!\n"
        "  You: /image ./assets/cat.jpg\n"
        "  You: What is this? /image ./assets/cat.jpg\n"
        "---------------------------------\n"
    )

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        image_path = None
        text = user_input

        # Two ways to attach an image:
        # 1) Pure command:      /image ./path/to/img.jpg   (then we ask for caption)
        # 2) Inline in a line:  What is this? /image ./path/to/img.jpg
        if user_input.startswith("/image "):
            # Mode 1: pure command, no text yet
            image_path = user_input[len("/image ") :].strip()

            if not image_path:
                print("Please provide an image path after /image")
                continue

            # Ask for the text part of the message
            text = input("Caption / question about the image: ").strip()
            if not text:
                text = "Describe the image."

        elif "/image " in user_input:
            # Mode 2: inline command at the end of the sentence
            text_part, image_part = user_input.split("/image ", 1)
            text = text_part.strip()
            image_path = image_part.strip()

            if not image_path:
                print("Please provide an image path after /image")
                continue

            if not text:
                text = "Describe the image."

        # 3. Add user message to history
        user_msg = build_user_message(text, image_path=image_path)
        messages.append(user_msg)

        # 4. Generate assistant reply (streaming)
        try:
            assistant_text, generated_ids = stream_assistant_reply(
                model, processor, messages, device, max_new_tokens=256
            )
        except KeyboardInterrupt:
            print("\n[Generation interrupted]")
            # You may want to pop the last user message if you treat it as cancelled:
            # messages.pop()
            continue

        # 5. Add assistant reply back into history so the chat is multi-turn
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            }
        )


if __name__ == "__main__":
    main()
