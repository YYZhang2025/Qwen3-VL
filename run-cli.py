import rich
import torch
from rich.console import Console

from src.model.qwen3vl import Qwen3VL
from src.processors.preprocessor import Processor

# -------------------------
# Utilities
# -------------------------


def parse_paths(raw: str) -> list[str]:
    """
    Parse one or more paths from a raw string.
    Supports space- or comma-separated paths.
    """
    raw = raw.strip()
    if not raw:
        return []
    parts = raw.replace(",", " ").split()
    return [p for p in parts if p]


def build_user_message(
    text: str,
    image_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
):
    """
    Build a user message with multimodal content blocks.
    Order matters: vision blocks first, then text.
    """
    content: list[dict] = []

    if image_paths:
        for path in image_paths:
            content.append({"type": "image", "url": path})

    if video_paths:
        for path in video_paths:
            content.append({"type": "video", "video": path})

    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


# -------------------------
# Streaming generation
# -------------------------


def stream_assistant_reply(model, processor, messages, device, max_new_tokens=256):
    """
    Prepare inputs via Processor, stream model output, and return final text.
    """
    inputs = processor(
        messages,
        add_generation_prompt=True,
        device=device,
    )

    generated_token_ids = []
    console = Console()
    console.print("[bold magenta]Assistant:[/bold magenta] ", end="")

    for token in model.generate_stream(**inputs, max_new_tokens=max_new_tokens):
        if isinstance(token, torch.Tensor):
            token_id = token.item()
        else:
            token_id = int(token)

        generated_token_ids.append(token_id)
        piece = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)

    print()
    assistant_text = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return assistant_text, generated_token_ids


# -------------------------
# Main CLI
# -------------------------


def main():
    # 1. Load model & processor
    model = Qwen3VL.from_pretrained("./checkpoints/Qwen3VL-2B-Instruct")
    rich.print("[bold green]Model loaded successfully.[/bold green]")

    processor = Processor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    device = next(model.parameters()).device
    rich.print(f"[bold green]Processor loaded successfully. Device: {device}[/bold green]")

    console = Console()

    # 2. Conversation history
    messages: list[dict] = []

    print(
        "\nQwen3-VL Chat CLI\n"
        "---------------------------------\n"
        "Commands:\n"
        "  /exit                quit\n"
        "  /image PATHS         send image(s)\n"
        "  /video PATHS         send video(s)\n"
        "\nExamples:\n"
        "  You: Hello!\n"
        "  You: /image ./assets/cat.jpg\n"
        "  You: /video ./assets/test_video.mp4\n"
        "  You: What happens here? /video ./assets/test_video.mp4\n"
        "  You: Compare /image a.jpg /video b.mp4\n"
        "---------------------------------\n"
    )

    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye!")
            break

        image_paths: list[str] = []
        video_paths: list[str] = []
        text = user_input

        # -------------------------
        # Pure commands
        # -------------------------
        if user_input.startswith("/image "):
            image_paths = parse_paths(user_input[len("/image ") :])
            if not image_paths:
                print("Please provide image paths after /image")
                continue
            text = input("Caption / question: ").strip() or "Describe the image."

        elif user_input.startswith("/video "):
            video_paths = parse_paths(user_input[len("/video ") :])
            if not video_paths:
                print("Please provide video paths after /video")
                continue
            text = input("Caption / question: ").strip() or "Describe the video."

        # -------------------------
        # Inline commands
        # -------------------------
        else:
            if "/image " in text:
                text_part, image_part = text.split("/image ", 1)
                text = text_part.strip()
                image_paths = parse_paths(image_part)

            if "/video " in text:
                text_part, video_part = text.split("/video ", 1)
                text = text_part.strip()
                video_paths = parse_paths(video_part)

            if (image_paths or video_paths) and not text:
                text = "Describe the content."

        # -------------------------
        # Build message
        # -------------------------
        user_msg = build_user_message(
            text=text,
            image_paths=image_paths if image_paths else None,
            video_paths=video_paths if video_paths else None,
        )
        messages.append(user_msg)

        # -------------------------
        # Generate reply
        # -------------------------
        try:
            assistant_text, _ = stream_assistant_reply(
                model,
                processor,
                messages,
                device,
                max_new_tokens=256,
            )
        except KeyboardInterrupt:
            print("\n[Generation interrupted]")
            messages.pop()
            continue

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )


if __name__ == "__main__":
    main()
