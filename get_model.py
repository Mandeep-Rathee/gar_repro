

import os
from rank1_prompt import get_prompt
from qwen_reranker_vllm import QwenRerankerVLLM
from tfrank_reranker import TFRankReranker
from rank1 import Rank1Reranker

from pyterrier_t5 import MonoT5ReRanker
from ranklamma_pyterrier import RankLammaReRanker

from qwen_prompts import rerank_prompts





tf_rank_models_map = {
    "tfrank-0.6b": "Johnnyfans/TFRank-GRPO-Qwen3-0.6B",
    "tfrank-1.7b": "Johnnyfans/TFRank-GRPO-Qwen3-1.7B",
    "tfrank-4b": "Johnnyfans/TFRank-GRPO-Qwen3-4B",
    "tfrank-8b": "Johnnyfans/TFRank-GRPO-Qwen3-8B",
    "tfrank-0.6b-think": "Johnnyfans/TFRank-GRPO-Qwen3-0.6B",
    "tfrank-1.7b-think": "Johnnyfans/TFRank-GRPO-Qwen3-1.7B",
    "tfrank-4b-think": "Johnnyfans/TFRank-GRPO-Qwen3-4B",
    "tfrank-8b-think": "Johnnyfans/TFRank-GRPO-Qwen3-8B"
}


qwen_models_map = {
    "qwen-0.6b": "Qwen/Qwen3-Reranker-0.6B",
    "qwen-4b": "Qwen/Qwen3-Reranker-4B",
    "qwen-8b": "Qwen/Qwen3-Reranker-8B",
}

rank1_url = {"rank1-7b":"http://rank1-7b.macavaney.us/",
        "rank1-0.5b":"http://rank1-0-5b.macavaney.us/",
        "rank1-1.5b":"http://rank1-1-5b.macavaney.us/",
        }




def GetRanker(model_name, model_type, batch_size, task, reasoning=False):
    """
    Factory function to create and return the appropriate reranker based on model_name.
    """
    import torch
    
    # Set default CUDA device to GPU 0 for all PyTorch-based rankers
    # This ensures rankers use GPU 0 while retriever can use GPU 1
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    dataset_prompt = get_prompt("BRIGHT", task)

    if "rank1" in model_name.lower():
        return Rank1Reranker(model_name_or_path=f"jhu-clsp/{model_name}", batch_size=batch_size, context_size=32000, dataset_prompt=dataset_prompt)
    
    elif "monot5-base" in model_name.lower():
        return MonoT5ReRanker(verbose=False, batch_size=batch_size)

    elif "monot5-3b" in model_name.lower():
        return MonoT5ReRanker(model="castorini/monot5-3b-msmarco", verbose=False, batch_size=batch_size)

    elif "llama" in model_name.lower():
        return RankLammaReRanker(batch_size=batch_size, verbose=False, device='cuda:0')

    elif "qwen" in model_name.lower():
        return QwenRerankerVLLM(
            qwen_models_map[model_name],
            instruction=rerank_prompts[task]["instructions"]["query"],
            batch_size=batch_size, 
            max_length=8192,
            max_model_len=10000,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.7,  # Reduced from 0.8 to leave more memory headroom
            verbose=False,
            device='cuda:0'
        ), "pointwise"
    

    elif "tfrank" in model_name.lower():
        return TFRankReranker(
            #model_name="Johnnyfans/TFRank-GRPO-Qwen3-8B",
            model_name = tf_rank_models_map[model_name],
            batch_size=batch_size,
            max_length=8192,
            max_model_len=16384,  # Increased to handle longer documents (was 10000)
            enable_prefix_caching=True,
            gpu_memory_utilization=0.7,  # Leave memory headroom like Qwen
            verbose=False,
            device='cuda:0',  # Same GPU as other rankers
            think_mode=reasoning,  # Use /no think mode
            temperature=0.0,
            score_mode="combined"  # Average of fine-grained and yes/no scores
        ), "pointwise"
    
    else:
        raise ValueError(f"Model {model_name} not supported. Please choose from gpt, zephyr, vicuna, rank1, monot5, llama, qwen, tfrank")

