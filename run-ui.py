# run-ui.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import rich
import torch
import torch.nn.functional as F

from src.model.qwen3vl import KVCache, Qwen3VL
from src.processors.preprocessor import Processor

# ============================================================
# Load model once
# ============================================================

MODEL_DIR = "./checkpoints/Qwen3VL-2B-Instruct"
TOKENIZER_REPO = "Qwen/Qwen3-VL-2B-Instruct"

USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "1") == "1"
PORT = int(os.getenv("PORT", "7860"))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _load_model_and_processor():
    model = Qwen3VL.from_pretrained(MODEL_DIR)
    if USE_TORCH_COMPILE:
        model = torch.compile(model)
    processor = Processor.from_pretrained(TOKENIZER_REPO)
    device = next(model.parameters()).device
    rich.print(f"[bold green]Model loaded.[/bold green] Device: [bold]{device}[/bold]")
    return model, processor, device


MODEL, PROCESSOR, DEVICE = _load_model_and_processor()


# ============================================================
# Session state (per Gradio session)
# ============================================================


@dataclass
class SessionCache:
    # 1) media preproc cache: key -> (patches_np, grid_t, grid_h, grid_w)
    media_cache: Dict[str, Any] = field(default_factory=dict)

    # 2) vision feature cache: vision_key -> (vision_embed, deepstack_features)
    #    NOTE: stored on MODEL device/dtype
    vision_cache: Dict[str, Any] = field(default_factory=dict)

    # 3) KV cache for language model (persist across turns)
    kv_cache: KVCache = field(default_factory=KVCache)

    # 4) cached prompt token ids (on DEVICE)
    cached_input_ids: Optional[torch.Tensor] = None  # [1, T_cached]

    # 5) remember whether cached prefix was built with vision (not strictly required)
    cached_has_vision: bool = False


def _file_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    return "file"


def build_user_message_for_processor(text: str, files: List[str] | None) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if files:
        for p in files:
            kind = _file_kind(p)
            if kind == "image":
                content.append({"type": "image", "image": p})
            elif kind == "video":
                content.append({"type": "video", "video": p})
            else:
                content.append({"type": "image", "image": p})
    if text:
        content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def build_assistant_message_for_processor(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


# ============================================================
# Core: vision cache + incremental prefill + streaming decode
# ============================================================


@torch.no_grad()
def _get_vision_features(
    pixels: torch.Tensor,
    d_image: torch.Tensor,
    vision_key: str,
    session: SessionCache,
) -> Tuple[torch.Tensor, Any]:
    if vision_key in session.vision_cache:
        return session.vision_cache[vision_key]

    dtype = next(MODEL.parameters()).dtype
    v_embed, deep = MODEL.model.visual(pixels=pixels.to(dtype), d_image=d_image)
    session.vision_cache[vision_key] = (v_embed, deep)

    # simple cap to avoid unbounded growth (tune if you like)
    if len(session.vision_cache) > 4:
        k0 = next(iter(session.vision_cache.keys()))
        session.vision_cache.pop(k0, None)

    return v_embed, deep


def _starts_with(a: torch.Tensor, prefix: torch.Tensor) -> bool:
    # a: [1, T], prefix: [1, Tp]
    if a is None or prefix is None:
        return False
    if a.shape[1] < prefix.shape[1]:
        return False
    return torch.equal(a[:, : prefix.shape[1]], prefix)


# @torch.no_grad()
# def _prefill_incremental(
#     full_input_ids: torch.Tensor,  # [1, T_full]
#     d_image: Optional[torch.Tensor],
#     session: SessionCache,
#     vision_embed: Optional[torch.Tensor],
#     deepstack_features: Optional[Any],
#     image_token_id: int,
#     video_token_id: int,
# ) -> torch.Tensor:
#     """
#     Ensure session.kv_cache contains KV for full_input_ids, using incremental prefill if possible.
#     Return logits for the last token of full_input_ids (shape [1, V]).
#     """

#     # Decide whether we can reuse existing KV
#     if session.cached_input_ids is None or not _starts_with(full_input_ids, session.cached_input_ids):
#         # Reset + full prefill
#         session.kv_cache.reset()
#         session.cached_input_ids = None
#         cached_len = 0
#     else:
#         cached_len = session.cached_input_ids.shape[1]

#     # Tokens to prefill this turn
#     new_ids = full_input_ids[:, cached_len:]  # [1, T_new]
#     if new_ids.numel() == 0:
#         # No new tokens: just compute logits for last cached token? We can reuse by running 1 step
#         # For simplicity, run one step on the last token (safe & cheap)
#         last_id = full_input_ids[:, -1:]
#         pos_full = Qwen3VL.get_position_ids(full_input_ids, d_image=d_image, image_token_id=image_token_id)
#         new_pos = pos_full[:, :, -1:]
#         new_embed = MODEL.get_input_embeddings()(last_id)
#         hidden = MODEL.model.language_model(
#             input_embed=new_embed, kv_cache=session.kv_cache, position_ids=new_pos
#         )
#         last_hidden = hidden[:, -1, :]
#     else:
#         # Position ids: compute once for full prompt, slice new range
#         pos_full = Qwen3VL.get_position_ids(full_input_ids, d_image=d_image, image_token_id=image_token_id)
#         pos_new = pos_full[:, :, cached_len:]  # [3,1,T_new]

#         new_embed = MODEL.get_input_embeddings()(new_ids)

#         # If this is the very first prefill (cached_len == 0) and we have vision, run with vision once
#         if cached_len == 0 and vision_embed is not None:
#             vision_mask = (full_input_ids == image_token_id) | (full_input_ids == video_token_id)
#             hidden = MODEL.model.language_model(
#                 input_embed=new_embed,
#                 vision_embed=vision_embed,
#                 deepstack_features=deepstack_features,
#                 kv_cache=session.kv_cache,
#                 vision_mask=vision_mask,
#                 position_ids=pos_new,
#             )
#             session.cached_has_vision = True
#         else:
#             hidden = MODEL.model.language_model(
#                 input_embed=new_embed,
#                 kv_cache=session.kv_cache,
#                 position_ids=pos_new,
#             )

#         last_hidden = hidden[:, -1, :]

#         # Update cached_input_ids
#         if session.cached_input_ids is None:
#             session.cached_input_ids = full_input_ids
#         else:
#             session.cached_input_ids = full_input_ids

#     # Compute logits
#     if MODEL.lm_head is None:
#         logits = last_hidden @ MODEL.model.language_model.embed_tokens.weight.T
#     else:
#         logits = MODEL.lm_head(last_hidden)
#     return logits


@torch.no_grad()
def _prefill_incremental(
    full_input_ids: torch.Tensor,  # [1, T_full]
    d_image: Optional[torch.Tensor],
    session: SessionCache,
    vision_embed: Optional[torch.Tensor],
    deepstack_features: Optional[Any],
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """
    Cross-turn KV incremental prefill.

    Key rule for correctness with multi-image/video in a conversation:
      - If the NEWLY appended tokens contain vision placeholders (image/video token),
        we must RESET kv_cache and do a full prefill once with vision embeddings.

    Returns logits for the last token: [1, V]
    """

    def _has_vision_tokens(x: torch.Tensor) -> bool:
        return bool((x == image_token_id).any().item() or (x == video_token_id).any().item())

    # Decide whether we can reuse existing KV by prefix check
    can_reuse = session.cached_input_ids is not None and _starts_with(
        full_input_ids, session.cached_input_ids
    )

    if not can_reuse:
        session.kv_cache.reset()
        session.cached_input_ids = None
        cached_len = 0
    else:
        cached_len = session.cached_input_ids.shape[1]

    # Tokens newly appended this turn
    new_ids = full_input_ids[:, cached_len:]  # [1, T_new]

    # ---- CRITICAL FIX ----
    # If we are reusing KV (cached_len>0) but the NEW part contains vision tokens,
    # incremental prefill without vision injection is incorrect.
    # So we reset and do full prefill (cached_len=0).
    if cached_len > 0 and new_ids.numel() > 0 and _has_vision_tokens(new_ids):
        session.kv_cache.reset()
        session.cached_input_ids = None
        cached_len = 0
        new_ids = full_input_ids  # full prefill

    # Now prefill either incrementally (text-only new_ids) or fully (cached_len=0)
    if new_ids.numel() == 0:
        # No new tokens -> run one step on last token to get logits
        last_id = full_input_ids[:, -1:]
        pos_full = Qwen3VL.get_position_ids(full_input_ids, d_image=d_image, image_token_id=image_token_id)
        new_pos = pos_full[:, :, -1:]
        new_embed = MODEL.get_input_embeddings()(last_id)
        hidden = MODEL.model.language_model(
            input_embed=new_embed,
            kv_cache=session.kv_cache,
            position_ids=new_pos,
        )
        last_hidden = hidden[:, -1, :]
    else:
        # Compute position ids once for the whole prompt, then slice the needed segment
        pos_full = Qwen3VL.get_position_ids(full_input_ids, d_image=d_image, image_token_id=image_token_id)
        pos_new = pos_full[:, :, cached_len:]  # [3,1,T_new]

        new_embed = MODEL.get_input_embeddings()(new_ids)

        # If this segment includes vision tokens, we must run with vision params.
        # (This will happen in the full-prefill case above.)
        if vision_embed is not None and _has_vision_tokens(new_ids):
            vision_mask_full = (full_input_ids == image_token_id) | (full_input_ids == video_token_id)
            vision_mask_new = vision_mask_full[:, cached_len:]  # [1, T_new]
            hidden = MODEL.model.language_model(
                input_embed=new_embed,
                vision_embed=vision_embed,
                deepstack_features=deepstack_features,
                kv_cache=session.kv_cache,
                vision_mask=vision_mask_new,
                position_ids=pos_new,
            )
            session.cached_has_vision = True
        else:
            hidden = MODEL.model.language_model(
                input_embed=new_embed,
                kv_cache=session.kv_cache,
                position_ids=pos_new,
            )

        last_hidden = hidden[:, -1, :]

        # Update cached_input_ids to the full prompt we just cached into kv
        session.cached_input_ids = full_input_ids

    # Compute logits
    if MODEL.lm_head is None:
        logits = last_hidden @ MODEL.model.language_model.embed_tokens.weight.T
    else:
        logits = MODEL.lm_head(last_hidden)
    return logits


@torch.no_grad()
def stream_chat(
    model_messages: List[Dict[str, Any]],
    session: SessionCache,
    max_new_tokens: int,
) -> Generator[str, None, str]:
    """
    Streaming generation with:
      - session media_cache (Processor)
      - session vision_cache (VisionEncoder outputs)
      - session kv_cache + cached_input_ids (cross-turn incremental prefill)

    Yields text chunks; returns final assistant_text (as generator return value).
    """

    # 1) Build full prompt ids (+ pixels/d_image) with media_cache; also return media_keys for vision_key
    proc_out = PROCESSOR(
        model_messages,
        add_generation_prompt=True,
        device=DEVICE,
        media_cache=session.media_cache,
        return_media_keys=True,
    )
    full_input_ids: torch.Tensor = proc_out["input_ids"]
    pixels: Optional[torch.Tensor] = proc_out.get("pixels", None)
    d_image: Optional[torch.Tensor] = proc_out.get("d_image", None)
    media_keys: List[str] = proc_out.get("media_keys", [])

    # 2) Vision features (cached)
    vision_embed = None
    deepstack = None
    if pixels is not None and d_image is not None and len(media_keys) > 0:
        vision_key = "|".join(media_keys)
        vision_embed, deepstack = _get_vision_features(pixels, d_image, vision_key, session)

    # 3) Incremental prefill to extend KV to full_input_ids, get logits for last prompt token
    image_token_id = MODEL.config.image_token_id
    video_token_id = MODEL.config.video_token_id
    logits = _prefill_incremental(
        full_input_ids=full_input_ids,
        d_image=d_image,
        session=session,
        vision_embed=vision_embed,
        deepstack_features=deepstack,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )

    # 4) Decode loop (token-by-token)
    #    We avoid recomputing full position_ids each token by using the last prompt pos + step.
    #    For text-only generation, position in 3 dims all increments by 1 each new token.
    if d_image is None:
        last_pos_val = full_input_ids.shape[1] - 1
    else:
        pos_full = Qwen3VL.get_position_ids(full_input_ids, d_image=d_image, image_token_id=image_token_id)
        last_pos_val = int(pos_full[0, 0, -1].item())

    assistant_text = ""
    flush_buf = ""

    # def sample_next(logits_1v: torch.Tensor) -> torch.Tensor:
    # greedy
    # return torch.argmax(logits_1v, dim=-1, keepdim=True)  # [1,1]
    def sample_top_p(logits: torch.Tensor, temperature=0.7, top_p=0.9):
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        # keep smallest set with cumulative prob >= top_p
        cutoff = cum > top_p
        cutoff[..., 0] = False
        sorted_probs[cutoff] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_in_sorted)
        return next_token

    stop_tokens = [151645, 151644, 151643]  # <|im_end|>, <|im_start|>, <|endoftext|>

    for _ in range(max_new_tokens):
        next_token = sample_top_p(logits)  # [1,1]
        token_id = int(next_token[0, 0].item())

        # update cached_input_ids (so next turn prompt can be checked for prefix match)
        if session.cached_input_ids is None:
            session.cached_input_ids = torch.cat([full_input_ids, next_token], dim=1)
        else:
            session.cached_input_ids = torch.cat([session.cached_input_ids, next_token], dim=1)

        # decode piece
        piece = PROCESSOR.tokenizer.decode([token_id], skip_special_tokens=True)
        assistant_text += piece
        flush_buf += piece

        if len(flush_buf) >= 16:
            yield flush_buf
            flush_buf = ""

        if token_id in stop_tokens:
            break

        # prepare next step
        last_pos_val += 1
        new_pos = torch.tensor(
            [[[last_pos_val]], [[last_pos_val]], [[last_pos_val]]], device=DEVICE, dtype=torch.long
        )
        new_embed = MODEL.get_input_embeddings()(next_token)

        hidden = MODEL.model.language_model(
            input_embed=new_embed,
            kv_cache=session.kv_cache,
            position_ids=new_pos,
        )
        last_hidden = hidden[:, -1, :]

        if MODEL.lm_head is None:
            logits = last_hidden @ MODEL.model.language_model.embed_tokens.weight.T
        else:
            logits = MODEL.lm_head(last_hidden)

    if flush_buf:
        yield flush_buf

    return assistant_text


# ============================================================
# Gradio app
# ============================================================


def add_message(
    chat_history: list[dict[str, Any]],
    mm: dict[str, Any] | None,
    model_messages: list[dict[str, Any]],
):
    chat_history = chat_history or []
    model_messages = model_messages or []

    mm = mm or {}
    text = (mm.get("text") or "").strip()
    files = mm.get("files") or []

    if not text and files:
        text = "Describe the content."
    if not text and not files:
        return chat_history, gr.MultimodalTextbox(value=None), model_messages

    ui_user_msg: dict[str, Any] = {"role": "user", "content": []}
    for f in files:
        ui_user_msg["content"].append({"path": f})
    if text:
        ui_user_msg["content"].append(text)
    chat_history.append(ui_user_msg)

    model_messages.append(build_user_message_for_processor(text=text, files=files))
    return chat_history, gr.MultimodalTextbox(value=None), model_messages


def bot(
    chat_history: list[dict[str, Any]],
    model_messages: list[dict[str, Any]],
    session: SessionCache,
    max_new_tokens: int,
):
    chat_history = chat_history or []
    model_messages = model_messages or []

    # UI: append empty assistant message
    chat_history.append({"role": "assistant", "content": ""})
    assistant_text = ""

    # stream
    gen = stream_chat(model_messages, session=session, max_new_tokens=max_new_tokens)
    try:
        for chunk in gen:
            assistant_text += chunk
            chat_history[-1]["content"] = assistant_text
            yield chat_history, model_messages, session
    except StopIteration as e:
        # python generator return value
        if e.value:
            assistant_text = e.value

    # persist assistant message
    model_messages.append(build_assistant_message_for_processor(assistant_text))
    yield chat_history, model_messages, session


def clear_all():
    # IMPORTANT: clear KV cache + caches
    return [], gr.MultimodalTextbox(value=None), [], SessionCache()


def _disable_inputs():
    return gr.update(interactive=False), gr.update(interactive=False)


def _enable_inputs():
    return gr.update(interactive=True), gr.update(interactive=True)


def main():
    with gr.Blocks(title="Qwen3-VL Chat (Session Cache + KV Incremental Prefill)") as demo:
        gr.Markdown("# Qwen3-VL Chat\n")

        chatbot = gr.Chatbot(height=560, show_label=False)

        mm = gr.MultimodalTextbox(
            placeholder="Type a message or upload image/video…",
            file_types=["image", "video"],
            file_count="multiple",
            show_label=False,
        )

        with gr.Row():
            clear_btn = gr.Button("Clear", variant="secondary")
            max_new_tokens = gr.Slider(16, 2048, value=512, step=16, label="max_new_tokens")
            send_btn = gr.Button("Send", variant="primary")

        model_messages = gr.State([])
        session_state = gr.State(SessionCache())

        # button click
        send_btn.click(
            add_message,
            inputs=[chatbot, mm, model_messages],
            outputs=[chatbot, mm, model_messages],
            queue=False,
        ).then(_disable_inputs, None, [send_btn, mm], queue=False).then(
            bot,
            inputs=[chatbot, model_messages, session_state, max_new_tokens],
            outputs=[chatbot, model_messages, session_state],
        ).then(_enable_inputs, None, [send_btn, mm], queue=False)

        # enter submit
        mm.submit(
            add_message,
            inputs=[chatbot, mm, model_messages],
            outputs=[chatbot, mm, model_messages],
            queue=False,
        ).then(_disable_inputs, None, [send_btn, mm], queue=False).then(
            bot,
            inputs=[chatbot, model_messages, session_state, max_new_tokens],
            outputs=[chatbot, model_messages, session_state],
        ).then(_enable_inputs, None, [send_btn, mm], queue=False)

        clear_btn.click(clear_all, None, [chatbot, mm, model_messages, session_state], queue=False)

    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)


if __name__ == "__main__":
    main()
