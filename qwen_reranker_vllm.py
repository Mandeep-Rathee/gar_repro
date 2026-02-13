# Requires vllm>=0.8.5
import torch
import pandas as pd
import pyterrier as pt
import math
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt


class QwenRerankerVLLM(pt.Transformer):
    """
    PyTerrier transformer that uses Qwen3-Reranker with vLLM for efficient document reranking.
    
    Usage:
        reranker = QwenRerankerVLLM()
        pipeline = bm25 >> pt.text.get_text(index, 'text') >> reranker
        results = pipeline.search('your query')
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        batch_size: int = 32,
        max_length: int = 8192,
        tensor_parallel_size: int = None,
        max_model_len: int = 10000,
        enable_prefix_caching: bool = True,
        gpu_memory_utilization: float = 0.8,
        text_field: str = "text",
        verbose: bool = False,
        device: str = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for the Qwen reranker
            instruction: Task instruction for the reranker
            batch_size: Number of query-document pairs to process at once
            max_length: Maximum sequence length for input
            tensor_parallel_size: Number of GPUs to use (auto-detected if None, ignored if device is specified)
            max_model_len: Maximum model length for vLLM
            enable_prefix_caching: Whether to enable prefix caching in vLLM
            gpu_memory_utilization: GPU memory utilization for vLLM
            text_field: Column name containing document text
            verbose: Whether to print progress information
            device: Device to use (e.g., 'cuda:0', 'cuda:1', or None for auto). Overrides tensor_parallel_size.
        """
        self.model_name = model_name
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_field = text_field
        self.verbose = verbose
        self.device = device
        
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
                print(f"Assigning QwenRerankerVLLM to logical GPU {gpu_id} (physical GPU {physical_gpu_id})")
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
        
        # Prepare suffix tokens
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        # Token IDs for yes/no
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )
    
    def _format_instruction(self, query: str, doc: str) -> list:
        """Format a query-document pair with the instruction as chat messages."""
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text
    
    def _process_inputs(self, pairs: list) -> list:
        """Tokenize and prepare inputs for vLLM."""
        messages = [self._format_instruction(query, doc) for query, doc in pairs]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        # Truncate to max_length and add suffix
        max_len = self.max_length - len(self.suffix_tokens)
        messages = [ele[:max_len] + self.suffix_tokens for ele in messages]
        # Convert to TokensPrompt for vLLM
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages
    
    def _compute_scores(self, messages: list) -> list:
        """Compute relevance scores for the inputs using vLLM."""
        outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        for i in range(len(outputs)):
            final_logits = outputs[i].outputs[0].logprobs[-1]
            
            # Get logits for true and false tokens
            if self.true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self.true_token].logprob
            
            if self.false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self.false_token].logprob
            
            # Compute normalized score
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)
        
        return scores
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rerank documents in the input DataFrame.
        
        Args:
            df: DataFrame with columns 'query' and text_field (default 'text')
                Must also have 'qid' and 'docno' columns for proper PyTerrier integration
        
        Returns:
            DataFrame with updated 'score' column and re-ordered by score
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
        all_scores = []
        batch_size = 1
        for i in range(0, len(pairs),   batch_size):
            batch = pairs[i:i + batch_size]
            if self.verbose:
                print(f"Processing batch {i // batch_size + 1}/{(len(pairs) - 1) // batch_size + 1}")
            
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores)
        
        # Update DataFrame
        result = df.copy()
        result['score'] = all_scores
        
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
        return f"QwenRerankerVLLM(model={self.model_name}, batch_size={self.batch_size}, tensor_parallel={self.tensor_parallel_size})"


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
    reranker = QwenRerankerVLLM(verbose=True)
    
    # Apply reranking
    result_df = reranker.transform(sample_df)
    
    print("\nReranked DataFrame:")
    print(result_df[['qid', 'docno', 'query', 'score', 'rank']])
    
    # Cleanup
    destroy_model_parallel()
