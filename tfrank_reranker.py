# Requires vllm>=0.8.5
import torch
import pandas as pd
import pyterrier as pt
import math
from typing import Tuple, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt

# Import official TFRank utility functions
from tfrank_utils import (
    build_completion_prompt,
    get_next_token_probs,
    get_max_model_tokens,
)


class TFRankReranker(pt.Transformer):
    """
    PyTerrier transformer that uses TFRank with vLLM for efficient document reranking.
    
    TFRank uses a two-level scoring system:
    - Fine-grained: 0-4 scale (0=completely irrelevant, 4=completely relevant)
    - Binary: yes/no (where 0-1 maps to "no", 2-4 maps to "yes")
    
    The final score is typically the average of normalized fine-grained score and yes probability.
    
    Usage:
        reranker = TFRankReranker(
            model_name="Johnnyfans/TFRank-GRPO-Qwen3-8B"
        )
        pipeline = bm25 >> pt.text.get_text(index, 'text') >> reranker
        results = pipeline.search('your query')
    """
    
    def __init__(
        self,
        model_name: str = "Johnnyfans/TFRank-GRPO-Qwen3-8B",
        system_prompt: str = (
            "Based on the relevance of the Documents to the Query and "
            "the Instruct provided to complete the task."
        ),
        batch_size: int = 16,
        max_length: int = 8192,
        tensor_parallel_size: int = None,
        max_model_len: int = 10000,
        max_new_tokens: int = 1024,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.7,
        text_field: str = "text",
        verbose: bool = False,
        device: str = None,
        think_mode: bool = False,
        temperature: float = 0.0,
        score_mode: str = "combined",  # "combined", "fine_grained", or "yes_prob"
    ):
        """
        Args:
            model_name: HuggingFace model identifier for TFRank
            system_prompt: System prompt for the model
            batch_size: Number of query-document pairs to process at once
            max_length: Maximum sequence length for input
            tensor_parallel_size: Number of GPUs to use (auto-detected if None, ignored if device is specified)
            max_model_len: Maximum model length for vLLM
            enable_prefix_caching: Whether to enable prefix caching in vLLM
            gpu_memory_utilization: GPU memory utilization for vLLM
            text_field: Column name containing document text
            verbose: Whether to print progress information
            device: Device to use (e.g., 'cuda:0', 'cuda:1', or None for auto). Overrides tensor_parallel_size.
            think_mode: Whether to use thinking mode (/think) or not (/no think)
            temperature: Sampling temperature
            score_mode: Which score to use as final score:
                - "combined": average of fine-grained and yes probability (default)
                - "fine_grained": only use 0-4 score normalized to [0,1]
                - "yes_prob": only use yes probability
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len  # Store for safety checks
        self.text_field = text_field
        self.verbose = verbose
        self.device = device
        self.think_mode = think_mode
        self.temperature = temperature
        self.score_mode = score_mode
        
        # Handle device specification for vLLM using distributed backend
        import os
        gpu_id = None
        original_cuda_visible = None
        physical_gpu_id = None
        
        if device is not None and 'cuda' in str(device):
            # Extract logical GPU index from device string (e.g., 'cuda:0' -> 0, 'cuda:1' -> 1)
            gpu_id = int(str(device).split(':')[-1]) if ':' in str(device) else 0
            
            # Map logical GPU ID to physical GPU ID from CUDA_VISIBLE_DEVICES
            original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if original_cuda_visible is not None:
                # Parse available GPUs (e.g., "0,1" or "2,5")
                available_gpus = [int(g.strip()) for g in original_cuda_visible.split(',')]
                if gpu_id < len(available_gpus):
                    physical_gpu_id = available_gpus[gpu_id]
                else:
                    physical_gpu_id = available_gpus[0]  # Fallback to first GPU
            else:
                # No CUDA_VISIBLE_DEVICES set, use logical ID directly
                physical_gpu_id = gpu_id
            
            if verbose:
                print(f"Assigning TFRankReranker to logical GPU {gpu_id} (physical GPU {physical_gpu_id})")
                print(f"Original CUDA_VISIBLE_DEVICES: {original_cuda_visible}")
            
            # When device is specified, use single GPU
            self.tensor_parallel_size = 1
        else:
            # Determine tensor parallel size
            if tensor_parallel_size is None:
                self.tensor_parallel_size = torch.cuda.device_count()
            else:
                self.tensor_parallel_size = tensor_parallel_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get max model tokens using official function
        self.max_model_tokens = get_max_model_tokens(model_name)  # Model's max context (e.g., 32768)
        self.reserved_tokens = self.max_new_tokens  # Reserve tokens for generation (matches max_tokens in SamplingParams)
        
        # Calculate effective max tokens to prevent exceeding vLLM's max_model_len
        # This ensures prompt + generation never exceeds vLLM's limit
        self.effective_max_tokens = min(max_length, max_model_len - self.reserved_tokens)
        
        if verbose:
            print(f"Model max tokens: {self.max_model_tokens}")
            print(f"Reserved tokens: {self.reserved_tokens}")
            print(f"vLLM max_model_len: {max_model_len}")
            print(f"Effective max for inputs: {self.effective_max_tokens}")
        
        # Temporarily set to specific GPU for this model initialization
        if physical_gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
            if verbose:
                print(f"Temporarily setting CUDA_VISIBLE_DEVICES={physical_gpu_id} for model initialization")
        
        # Load model with vLLM
        try:
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=gpu_memory_utilization
            )
        finally:
            # Restore original CUDA_VISIBLE_DEVICES after model initialization
            if physical_gpu_id is not None:
                if original_cuda_visible is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
                if verbose:
                    print(f"Restored CUDA_VISIBLE_DEVICES to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        
        # Prepare suffix tokens based on think mode
        if self.think_mode:
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n/think\n\n"
        else:
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n/no think\n\n"
        
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # Token IDs for scoring
        # Fine-grained scores: 0, 1, 2, 3, 4
        self.label_tokens = {
            '0': self.tokenizer("0", add_special_tokens=False).input_ids[0],
            '1': self.tokenizer("1", add_special_tokens=False).input_ids[0],
            '2': self.tokenizer("2", add_special_tokens=False).input_ids[0],
            '3': self.tokenizer("3", add_special_tokens=False).input_ids[0],
            '4': self.tokenizer("4", add_special_tokens=False).input_ids[0],
        }
        
        # Binary scores: yes, no
        self.yes_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.no_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        if verbose:
            print(f"Label token IDs: {self.label_tokens}")
            print(f"Yes token ID: {self.yes_token}, No token ID: {self.no_token}")
        
        # Sampling parameters
        # Note: We need to generate full response (not just 1 token) for original parsing logic
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,  # Generate full response: yes(2) or <think>...</think>\n\nyes(2)
            logprobs=20,  # Get top 20 logprobs for each token
        )
    
    def _build_input(self, query: str, document: str) -> str:
        """
        Construct input text consistent with TFRank training/evaluation format.
        Includes 0-4 scoring explanation and yes/no rules.
        """
        return (
            "<Instruct>: Please judge the relevance strength between the query and the document, "
            "and directly output the relevance judgment (yes or no), followed by the relevance "
            "score in parentheses, e.g., yes(score) or no(score).\n\n\n"
            "- Relevance scores are represented by numbers from 0 to 4, with the following meanings:\n"
            "0 means completely irrelevant,\n"
            "1 means weakly relevant,\n"
            "2 means moderately relevant,\n"
            "3 means strongly relevant,\n"
            "4 means completely relevant.\n\n"
            "- For binary relevance judgment (yes or no), the rule is:\n"
            "Scores 0 and 1 are considered irrelevant and represented as \"no\",\n"
            "Scores 2, 3, and 4 are considered relevant and represented as \"yes\".\n\n\n"
            f"<Query>: {query}\n\n\n"
            f"<Document>: {document}"
        )
    
    def _process_inputs(self, pairs: list) -> Tuple[list, list]:
        """
        Tokenize and prepare inputs for vLLM using official build_completion_prompt.
        
        Returns:
            Tuple of (token_prompts, prompt_token_counts)
        """
        token_prompts = []
        prompt_token_counts = []
        
        for query, doc in pairs:
            # Build input text using the standard format
            input_text = self._build_input(query, doc)
            
            # Use official build_completion_prompt function
            # Use effective_max_tokens to ensure we never exceed vLLM's max_model_len
            completion_prompt, prompt_token_count = build_completion_prompt(
                input_text=input_text,
                system_prompt=self.system_prompt,
                tokenizer=self.tokenizer,
                max_model_tokens=self.effective_max_tokens,  # Guaranteed to fit in vLLM
                reserved_tokens=self.reserved_tokens,
                think_mode=self.think_mode,
                reasoning_model=True,  # TFRank uses reasoning model format
                try_num=0,
            )
            
            # Tokenize the prompt
            tokens = self.tokenizer.encode(completion_prompt, add_special_tokens=False)
            
            # Safety check: ensure prompt doesn't exceed vLLM's limit
            if len(tokens) > self.max_model_len:
                if self.verbose:
                    print(f"WARNING: Prompt too long ({len(tokens)} tokens), truncating to {self.max_model_len - self.reserved_tokens}")
                # Hard truncate if somehow still too long
                tokens = tokens[:self.max_model_len - self.reserved_tokens]
            
            # Create TokensPrompt for vLLM
            token_prompt = TokensPrompt(prompt_token_ids=tokens)
            token_prompts.append(token_prompt)
            prompt_token_counts.append(len(tokens))
        
        return token_prompts, prompt_token_counts
    
    def _compute_scores(self, messages: list) -> list:
        """
        Compute relevance scores for the inputs using vLLM and official parsing.
        
        Uses get_next_token_probs to extract scores exactly as the authors intended.
        
        Returns list of tuples: (final_score, fg_score, yes_score)
        """
        outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
        
        scores = []
        for i in range(len(outputs)):
            # Use official get_next_token_probs to parse the response
            result = get_next_token_probs(
                response_stream=outputs[i],  # Note: parameter name is response_stream
                tokenizer=self.tokenizer,
                think_mode=self.think_mode,
                debug=self.verbose,  # Show debug output when verbose
            )
            
            if result is None:
                # Parsing failed, use default scores
                scores.append((0.5, 0.5, 0.5))
                continue
            
            label_probs, yesno_probs, response_text = result
            
            if self.verbose:
                print(f"\n  Pair {i+1} scoring:")
                print(f"    Response: {response_text}")
                print(f"    Label probs: {label_probs}")
                print(f"    Yes/No probs: {yesno_probs}")
            
            # Compute fine-grained score (0-4 scale) using author's method
            # Expected value: sum(label * prob), then normalize to [0,1]
            fg_raw = sum(float(k) * float(p) for k, p in label_probs.items())
            fg_score = max(0.0, min(fg_raw / 4.0, 1.0))
            
            # Get yes probability
            yes_score = float(yesno_probs.get("yes", 0.5))
            yes_score = max(0.0, min(yes_score, 1.0))
            
            # Compute final score based on mode (author's default is combined)
            if self.score_mode == "fine_grained":
                final_score = fg_score
            elif self.score_mode == "yes_prob":
                final_score = yes_score
            else:  # combined (author's default)
                final_score = (fg_score + yes_score) / 2.0
            
            scores.append((
                float(final_score),
                float(fg_score),
                float(yes_score)
            ))
        
        return scores
    
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank documents in the input DataFrame.
        
        Args:
            df: DataFrame with columns 'query' and text_field (default 'text')
                Must also have 'qid' and 'docno' columns for proper PyTerrier integration
        
        Returns:
            DataFrame with updated 'score' column and re-ordered by score
            Also adds 'tfrank_fg_score' and 'tfrank_yes_score' columns with component scores
        """
        if len(df) == 0:
            return df
        
        # Ensure required columns exist
        if 'query' not in df.columns:
            raise ValueError("DataFrame must have a 'query' column")
        if self.text_field not in df.columns:
            raise ValueError(f"DataFrame must have a '{self.text_field}' column")
        
        # Create query-document pairs
        queries = df['query'].tolist()
        documents = df[self.text_field].tolist()
        pairs = list(zip(queries, documents))
        
        # Process in batches
        all_final_scores = []
        all_fg_scores = []
        all_yes_scores = []
        
        mini_batch_size = 1  # just to make sure the batch size is not too large
        for i in range(0, len(pairs), mini_batch_size):
            batch = pairs[i:i + mini_batch_size]    
            if self.verbose:
                print(f"Processing batch {i // mini_batch_size + 1}/{(len(pairs) - 1) // mini_batch_size + 1}")
            
            # Process inputs using official build_completion_prompt
            inputs, prompt_token_counts = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            
            # Unpack scores
            for final_score, fg_score, yes_score in scores:
                all_final_scores.append(final_score)
                all_fg_scores.append(fg_score)
                all_yes_scores.append(yes_score)
        
        # Update DataFrame
        result = df.copy()
        result['score'] = all_final_scores
        result['tfrank_fg_score'] = all_fg_scores
        result['tfrank_yes_score'] = all_yes_scores
        
        # Sort by qid and score (descending)
        if 'qid' in result.columns:
            result = result.sort_values(['qid', 'score'], ascending=[True, False])
            # Update rank column if it exists
            result['rank'] = result.groupby('qid').cumcount()
        else:
            result = result.sort_values('score', ascending=False)
        
        return result.reset_index(drop=True)
    
    def __del__(self):
        """Cleanup vLLM resources."""
        try:
            destroy_model_parallel()
        except:
            pass
    
    def __repr__(self):
        return f"TFRankReranker(model={self.model_name}, batch_size={self.batch_size}, tensor_parallel={self.tensor_parallel_size})"


# Example usage
if __name__ == "__main__":
    if not pt.started():
        pt.init()
    
    # Create sample data as PyTerrier would provide
    sample_df = pd.DataFrame({
        'qid': ['q1', 'q1', 'q2', 'q2'],
        'docno': ['d1', 'd2', 'd3', 'd4'],
        'query': [
            "What is the capital of China?",
            "What is the capital of China?",
            "Explain gravity",
            "Explain gravity",
        ],
        'text': [
            "The capital of China is Beijing.",
            "China is a large country in Asia.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
            "Newton discovered the laws of motion.",
        ],
        'score': [1.0, 0.9, 1.0, 0.9],  # Initial BM25-like scores
        'rank': [0, 1, 0, 1],
    })
    
    print("Input DataFrame:")
    print(sample_df)
    print()
    
    # Initialize reranker
    reranker = TFRankReranker(
        model_name="Johnnyfans/TFRank-GRPO-Qwen3-0.6B",
        verbose=False,
        batch_size=2,
        think_mode=True,
    )
    
    # Apply reranking
    result_df = reranker.transform(sample_df)
    
    print("\nReranked DataFrame:")
    print(result_df[['qid', 'docno', 'query', 'score', 'tfrank_fg_score', 'tfrank_yes_score', 'rank']])
    
    # Cleanup
    destroy_model_parallel()

