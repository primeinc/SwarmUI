"""Test-only ComfyUI module for SwarmUITests ComfyLiveTests and `just app`, copied into the pinned ComfyUI's custom_nodes by SwarmUITests/ComfyLive/justfile (`just hook`). Never installed with SwarmUI.

ComfyUI derives diffusion model and VAE dimensions from tensor shapes, but builds Qwen3-VL text encoders from fixed configs:
comfy/text_encoders/llama.py Qwen3VL_8BConfig for the language model, comfy/text_encoders/qwen3vl.py QWEN3VL_VISION for the vision tower.
This reads those dimensions from the state dict being loaded and builds the encoder at them, the language model through the override
SDClipModel already accepts (comfy/sd1_clip.py, model_options["<model_name>_model_config"]), so the zero-weight text encoder written by
SwarmUITests/NullModels.cs, which has small tensors, builds, loads and runs. A released encoder yields its released dimensions.
"""
import inspect
import logging

import comfy.sd
import comfy.text_encoders.qwen3vl as qwen3vl

original_load_text_encoder_state_dicts = comfy.sd.load_text_encoder_state_dicts
load_signature = inspect.signature(original_load_text_encoder_state_dicts)

# comfy/text_encoders/llama.py: head_dim is a class constant of the config, not a field the override can set.
QWEN3VL_HEAD_DIM = 128


def count_indexed(sd, prefix, suffix):
    count = 0
    while f"{prefix}{count}{suffix}" in sd:
        count += 1
    return count


def qwen3vl_dims(sd):
    head_dim = sd["model.layers.0.self_attn.q_norm.weight"].shape[0]
    if head_dim != QWEN3VL_HEAD_DIM:
        raise ValueError(f"Qwen3-VL head width is fixed at {QWEN3VL_HEAD_DIM}, the state dict has {head_dim}")
    text = {
        "vocab_size": sd["model.embed_tokens.weight"].shape[0],
        "hidden_size": sd["model.embed_tokens.weight"].shape[1],
        "intermediate_size": sd["model.layers.0.mlp.gate_proj.weight"].shape[0],
        "num_hidden_layers": count_indexed(sd, "model.layers.", ".input_layernorm.weight"),
        "num_attention_heads": sd["model.layers.0.self_attn.q_proj.weight"].shape[0] // head_dim,
        "num_key_value_heads": sd["model.layers.0.self_attn.k_proj.weight"].shape[0] // head_dim,
    }
    vision = {
        "hidden_size": sd["model.visual.patch_embed.proj.weight"].shape[0],
        "intermediate_size": sd["model.visual.blocks.0.mlp.linear_fc1.weight"].shape[0],
        "depth": count_indexed(sd, "model.visual.blocks.", ".norm1.weight"),
    }
    return text, vision


def load_text_encoder_state_dicts(*args, **kwargs):
    bound = load_signature.bind(*args, **kwargs)
    bound.apply_defaults()
    state_dicts = bound.arguments["state_dicts"]
    # comfy/sd.py detect_te_model: DeepStack is unique to Qwen3-VL, and the merger output width tells 4B from 8B.
    if len(state_dicts) != 1 or "model.visual.deepstack_merger_list.0.norm.weight" not in state_dicts[0]:
        return original_load_text_encoder_state_dicts(*args, **kwargs)
    sd = state_dicts[0]
    model_type = "qwen3vl_4b" if sd["model.visual.merger.linear_fc2.weight"].shape[0] == 2560 else "qwen3vl_8b"
    text, vision = qwen3vl_dims(sd)
    logging.info(f"[swarm_null_models] building {model_type} text encoder at {text}, vision tower at {vision}")
    bound.arguments["model_options"] = {**bound.arguments["model_options"], f"{model_type}_model_config": text}
    fixed_vision = qwen3vl.QWEN3VL_VISION[model_type]
    qwen3vl.QWEN3VL_VISION[model_type] = {**fixed_vision, **vision}
    try:
        return original_load_text_encoder_state_dicts(*bound.args, **bound.kwargs)
    finally:
        qwen3vl.QWEN3VL_VISION[model_type] = fixed_vision


comfy.sd.load_text_encoder_state_dicts = load_text_encoder_state_dicts

NODE_CLASS_MAPPINGS = {}
