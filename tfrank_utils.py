"""
Utility functions for TFRank, from the official implementation.
Adapted to work with vLLM library instead of API streaming.
"""

import math
from typing import Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoConfig


def get_max_model_tokens(model_path):
    """Official implementation from authors."""
    try:
        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
        elif hasattr(config, "n_positions"):
            return config.n_positions
        elif hasattr(config, "seq_length"):
            return config.seq_length
        else:
            return 40960  # fallback
    except Exception as e:
        print(f"Warning: cannot load model config from {model_path}, use default 40960")
        return 40960


def softmax(logits):
    """Official implementation from authors."""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def get_logit(top_logprob_dict, candidates, fallback):
    """Official implementation from authors."""
    for token in candidates:
        if token in top_logprob_dict:
            return top_logprob_dict[token]
    return fallback


def parse_yesno(top_logprob_dict):
    """Official implementation from authors."""
    yes_candidates = ['yes', 'Yes', 'YES']
    no_candidates = ['no', 'No', 'NO']
    min_logprob = min(top_logprob_dict.values()) if top_logprob_dict else -100

    true_logit = get_logit(top_logprob_dict, yes_candidates, min_logprob - 10)
    false_logit = get_logit(top_logprob_dict, no_candidates, min_logprob - 10)

    scores = softmax([true_logit, false_logit])
    return {'yes': scores[0], 'no': scores[1]}


def parse_label(top_logprob_dict, target_label_tokens_str):
    """Official implementation from authors."""
    min_logprob = min(top_logprob_dict.values()) if top_logprob_dict else -100
    logits = []
    for num_str in target_label_tokens_str:
        logp = top_logprob_dict.get(num_str, min_logprob-10)
        logits.append(logp)
    probs = softmax(logits)
    return {int(num): prob for num, prob in zip(target_label_tokens_str, probs)}


def convert_vllm_logprobs(vllm_logprobs_dict, tokenizer):
    """
    Convert vLLM logprobs dict to simple {token_str: logprob} dict.
    
    vLLM format: {token_id: Logprob(logprob=..., decoded_token=...), ...}
    Target format: {token_str: logprob_value, ...}
    """
    result = {}
    if not isinstance(vllm_logprobs_dict, dict):
        return result
    
    for token_id, logprob_obj in vllm_logprobs_dict.items():
        # Get token string
        if hasattr(logprob_obj, 'decoded_token'):
            token_str = logprob_obj.decoded_token
        else:
            try:
                token_str = tokenizer.decode([int(token_id)])
            except:
                continue
        
        # Get logprob value
        if hasattr(logprob_obj, 'logprob'):
            logprob_val = logprob_obj.logprob
        else:
            logprob_val = float(logprob_obj) if isinstance(logprob_obj, (int, float)) else -100
        
        result[token_str] = logprob_val
    
    return result


def get_next_token_probs(response_stream, tokenizer, think_mode=False, debug=False):
    """
    Official implementation from authors, adapted for direct vLLM processing.
    
    Processes vLLM RequestOutput directly without streaming simulation.
    Uses original marker-based logic to find yes/no and label positions.
    """
    # Extract from vLLM RequestOutput
    if not hasattr(response_stream, 'outputs') or len(response_stream.outputs) == 0:
        if debug:
            print("ERROR: No outputs in vLLM response")
        return None
    
    output = response_stream.outputs[0]
    
    # Get token IDs and logprobs
    if not hasattr(output, 'token_ids') or not hasattr(output, 'logprobs'):
        if debug:
            print("ERROR: Missing token_ids or logprobs in vLLM output")
        return None
    
    token_ids = output.token_ids
    logprobs_list = output.logprobs
    
    if not token_ids or not logprobs_list:
        if debug:
            print("ERROR: Empty token_ids or logprobs")
        return None
    
    # Decode tokens and build cumulative text (matches original logic)
    buffer_tokens = []
    found_label_end = False
    found_yesno_end = False if think_mode else True
    label_finish = False
    yesno_finish = False
    label_result = None
    yesno_result = None
    target_label_tokens_str = ['0', '1', '2', '3', '4']
    
    # Process each token (original author's logic)
    for i, token_id in enumerate(token_ids):
        # Decode token
        try:
            token = tokenizer.decode([token_id])
        except:
            continue
        
        buffer_tokens.append(token)
        text_so_far = ''.join(buffer_tokens)  # Build cumulative text
        
        # Extract yes/no logprobs when marker found (original logic)
        if found_yesno_end and not yesno_finish:
            if i < len(logprobs_list):
                top_logprob_dict = convert_vllm_logprobs(logprobs_list[i], tokenizer)
                yesno_result = parse_yesno(top_logprob_dict)
                yesno_finish = True
        
        # Extract label logprobs when marker found (original logic)
        if found_label_end and not label_finish:
            if i < len(logprobs_list):
                top_logprob_dict = convert_vllm_logprobs(logprobs_list[i], tokenizer)
                label_result = parse_label(top_logprob_dict, target_label_tokens_str)
                label_finish = True
        
        if debug:
            print(text_so_far)
        
        # Find markers (original author's logic)
        if "</think>\n\n" in text_so_far:
            found_yesno_end = True
        
        if "yes(" in text_so_far or "no(" in text_so_far:
            found_label_end = True
        if "yes (" in text_so_far or "no (" in text_so_far:
            found_label_end = True
        if "Yes(" in text_so_far or "No(" in text_so_far:
            found_label_end = True
        if "Yes (" in text_so_far or "No (" in text_so_far:
            found_label_end = True
        
        # Return when both found (original logic)
        if label_result and yesno_result:
            return label_result, yesno_result, text_so_far
    
    return None


def build_completion_prompt(
    input_text: str,
    system_prompt: str,
    tokenizer,
    max_model_tokens: int = 40960,
    reserved_tokens: int = 1024,
    think_mode: bool = True,
    reasoning_model: bool = True,
    try_num: int = 0
) -> Tuple[str, int]:
    """Official implementation from authors."""
    if think_mode:
        fixed_prompt = (
            "<|im_start|>system\n"
            + system_prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + "\n<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    else:
        if reasoning_model:
            fixed_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "\n<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            fixed_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + "\n<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
    
    fixed_tokens = len(tokenizer.encode(fixed_prompt))
    max_input_tokens = max_model_tokens - reserved_tokens - fixed_tokens
    input_tokens = tokenizer.encode(input_text)
    
    if len(input_tokens) > max_input_tokens:
        input_tokens = input_tokens[:max_input_tokens]
    
    truncated_input = tokenizer.decode(input_tokens)
    
    if try_num >= 1:
        truncated_input += '\nPlease directly output the relevance judgment (yes or no), followed by the relevance score in parentheses, e.g., yes(score) or no(score).'
    
    if reasoning_model:
        prompt = truncated_input + (' /think' if think_mode else ' /no think')
    else:
        prompt = truncated_input
    
    if think_mode:
        completion_prompt = (
            "<|im_start|>system\n"
            + system_prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + prompt + "\n"
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
    else:
        if reasoning_model:
            completion_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        else:
            completion_prompt = (
                "<|im_start|>system\n"
                + system_prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + prompt + "\n"
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
    
    return completion_prompt, len(tokenizer.encode(completion_prompt))
