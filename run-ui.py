import os
from typing import Any

import gradio as gr
import rich
import torch

from src.model.qwen3vl import Qwen3VL
from src.processors.preprocessor import Processor

# ============================================================
# Load model once
# ============================================================


def _load_model_and_processor():
    model = Qwen3VL.from_pretrained("./checkpoints/Qwen3VL-2B-Instruct")
    model = torch.compile(model)
    processor = Processor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    device = next(model.parameters()).device
    rich.print(f"[bold green]Model loaded. Device: {device}[/bold green]")
    return model, processor, device


MODEL, PROCESSOR, DEVICE = _load_model_and_processor()


# ============================================================
# Helpers: map Gradio uploaded files -> Processor message format
# ============================================================


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _file_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    # Default: treat as image-like for display in chatbot, but you can extend this.
    return "file"


def build_user_message_for_processor(text: str, files: list[str] | None) -> dict[str, Any]:
    """Build a user message for your Processor (vision blocks first, then text)."""
    content: list[dict[str, Any]] = []

    if files:
        for p in files:
            kind = _file_kind(p)
            if kind == "image":
                content.append({"type": "image", "url": p})
            elif kind == "video":
                # Your existing Processor uses {"type": "video", "video": path}
                content.append({"type": "video", "video": p})
            else:
                # Fallback: pass as image path if your Processor only supports vision
                content.append({"type": "image", "url": p})

    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def build_assistant_message_for_processor(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def _prepare_inputs(model_messages: list[dict[str, Any]]):
    return PROCESSOR(
        model_messages,
        add_generation_prompt=True,
        device=DEVICE,
    )


def _stream_model_reply(model_messages: list[dict[str, Any]], max_new_tokens: int):
    """Yield decoded token pieces."""
    inputs = _prepare_inputs(model_messages)
    for token in MODEL.generate_stream(**inputs, max_new_tokens=max_new_tokens):
        if isinstance(token, torch.Tensor):
            token_id = int(token.item())
        else:
            token_id = int(token)
        yield PROCESSOR.tokenizer.decode([token_id], skip_special_tokens=True)


# ============================================================
# Gradio Blocks app (ChatGPT-like): upload first + streaming
# Based on Gradio guide: "Creating a Custom Chatbot with Blocks"
# ============================================================


def add_message(
    chat_history: list[dict[str, Any]],
    mm: dict[str, Any] | None,
    model_messages: list[dict[str, Any]],
):
    """Add user's multimodal message to BOTH:

    1) UI chat_history (OpenAI-style dicts so Chatbot can display files)
    2) model_messages (your Processor's multimodal format)

    This runs instantly (queue=False) so the user sees their message immediately.
    """
    chat_history = chat_history or []
    model_messages = model_messages or []

    mm = mm or {}
    text = (mm.get("text") or "").strip()
    files = mm.get("files") or []

    # If user uploads files first and doesn't type, mimic ChatGPT behavior.
    if not text and files:
        text = "Describe the content."

    # If nothing at all, do nothing.
    if not text and not files:
        return chat_history, gr.MultimodalTextbox(value=None), model_messages

    # 1) UI message (Gradio Chatbot can render files if content has {"path": ...})
    ui_user_msg: dict[str, Any] = {"role": "user", "content": []}
    for f in files:
        ui_user_msg["content"].append({"path": f})
    if text:
        ui_user_msg["content"].append(text)
    chat_history.append(ui_user_msg)

    # 2) Processor message
    model_messages.append(build_user_message_for_processor(text=text, files=files))

    # Clear multimodal input after submit
    return chat_history, gr.MultimodalTextbox(value=None), model_messages


def bot(
    chat_history: list[dict[str, Any]],
    model_messages: list[dict[str, Any]],
    max_new_tokens: int,
):
    """Stream assistant response.

    - Append an empty assistant message to UI history, then progressively fill it.
    - Append final assistant text to model_messages after streaming completes.
    """
    chat_history = chat_history or []
    model_messages = model_messages or []

    chat_history.append({"role": "assistant", "content": ""})

    assistant_text = ""
    for piece in _stream_model_reply(model_messages, max_new_tokens=max_new_tokens):
        assistant_text += piece
        chat_history[-1]["content"] = assistant_text
        yield chat_history, model_messages

    model_messages.append(build_assistant_message_for_processor(assistant_text))
    yield chat_history, model_messages


def clear_all():
    return [], gr.MultimodalTextbox(value=None), []


def main():
    with gr.Blocks(title="Qwen3-VL Chat") as demo:
        gr.Markdown(
            "# Qwen3-VL Chat\nChatGPT-like UI (Gradio): upload image/video first, then chat with streaming."
        )

        chatbot = gr.Chatbot(height=520)

        # Multimodal input (upload-first)
        mm = gr.MultimodalTextbox(
            placeholder="Type a message or upload files…",
            file_types=["image", "video"],
            file_count="multiple",
        )

        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=16,
                maximum=2048,
                value=256,
                step=16,
                label="max_new_tokens",
            )
            clear_btn = gr.Button("Clear")

        # Keep Processor conversation history separate from UI history
        model_messages = gr.State([])

        # Streaming pattern from Gradio guide:
        # 1) add user message immediately (queue=False)
        # 2) then stream bot response
        mm.submit(
            add_message,
            [chatbot, mm, model_messages],
            [chatbot, mm, model_messages],
            queue=False,
        ).then(
            bot,
            [chatbot, model_messages, max_new_tokens],
            [chatbot, model_messages],
        )

        clear_btn.click(clear_all, None, [chatbot, mm, model_messages], queue=False)

    # If you want public link: demo.launch(share=True)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
