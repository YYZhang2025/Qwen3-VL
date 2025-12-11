import torch
from rich.console import Console

from src.model.qwen3vl import Qwen3VL
from src.processors.preprocessor import Processor


def parse_image_paths(raw: str) -> list[str]:
    """
    Parse one or more image paths from a raw string.

    Supports both space- and comma-separated paths, e.g.:
      "/image img1.jpg img2.png"
      "img1.jpg, img2.png"
    """
    raw = raw.strip()
    if not raw:
        return []
    # allow comma-separated or space-separated
    parts = raw.replace(",", " ").split()
    return [p for p in parts if p]


def build_user_message(text: str, image_paths: list[str] | None = None):
    """
    Build a single user message in the same format as your current run.py.

    If one or more image paths are provided, each is added as a vision
    content block (in order), followed by the text block.
    """
    content: list[dict] = []

    if image_paths:
        for path in image_paths:
            content.append(
                {
                    "type": "image",
                    "url": path,
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

    console = Console()
    console.print("[bold magenta]Assistant:[/bold magenta] ", end="")

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
    print()

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

    console = Console()

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
        user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        image_paths: list[str] = []
        text = user_input

        # Two ways to attach image(s):
        # 1) Pure command:      /image path1 path2 ...     (then we ask for caption)
        # 2) Inline in a line:  What is this? /image path1 path2 ...
        if user_input.startswith("/image "):
            # Mode 1: pure command, no text yet
            raw_paths = user_input[len("/image ") :].strip()
            image_paths = parse_image_paths(raw_paths)

            if not image_paths:
                print("Please provide at least one image path after /image")
                continue

            # Ask for the text part of the message
            text = input("Caption / question about the image(s): ").strip()
            if not text:
                text = "Describe the image."

        elif "/image " in user_input:
            # Mode 2: inline command at the end of the sentence
            text_part, image_part = user_input.split("/image ", 1)
            text = text_part.strip()
            image_paths = parse_image_paths(image_part)

            if not image_paths:
                print("Please provide at least one image path after /image")
                continue

            if not text:
                text = "Describe the image."

        # 3. Add user message to history
        user_msg = build_user_message(text, image_paths=image_paths if image_paths else None)
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
