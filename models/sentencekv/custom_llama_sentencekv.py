""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import weakref
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import LlamaConfig
import bisect
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)
import time
import numpy as np
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)



class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config: Optional[LlamaConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.45"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaLinearScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does linear scaling (simply pass the model config to __init__)."
        )
        kwargs["rope_type"] = "linear"
        super().__init__(*args, **kwargs)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`LlamaDynamicNTKScalingRotaryEmbedding` is deprecated an will be removed in v4.45. Please use "
            "`LlamaRotaryEmbedding`, which now also does dynamic ntk scaling (simply pass the model config to "
            "__init__)."
        )
        kwargs["rope_type"] = "dynamic"
        super().__init__(*args, **kwargs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import bisect
from typing import Optional, Tuple

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.kv_seq_len = 0
        self.sentence_cache = []
        self.token_index_in_sentence = 0

        self.punctuations_ids = getattr(config, 'punctuations_ids')
        self.length_threshold = getattr(config, 'length_threshold', 1)
        self.start_budget = getattr(config, 'start_budget', 32)
        self.recent_budget = getattr(config, 'recent_budget', 32)
        self.token_budget = getattr(config, 'max_capacity_prompts', 128)
        self.semantic_factor = getattr(config, 'semantic_factor', 5)
        self.semantic_budget = int(self.token_budget * self.semantic_factor)
        
        self.parent_model = None
        self.window_size = 32
        self.kernel_size = 7
        self.pooling = 'maxpool'
        self.generate = True

        # Cache for vectorized sentence processing
        self.head_mean_k_tensor = None
        self.head_token_indices_tensor = None
        self.head_sentence_lengths = None

    def _vectorized_sentence_processing(self, key_states, all_indices, sentences_ranges, num_sentences):
        """Vectorized sentence processing to avoid per-head and per-sentence loops"""
        bsz, num_heads, compressed_seq_len, head_dim = key_states.shape
        device = key_states.device
        dtype = key_states.dtype
        
        # Pre-allocate result tensors
        max_tokens_per_sentence = max(100, (compressed_seq_len // num_sentences) * 2)
        head_mean_k_tensor = torch.zeros(
            (num_heads, num_sentences, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=True
        )
        head_token_indices_tensor = torch.full(
            (num_heads, num_sentences, max_tokens_per_sentence),
            -1,
            dtype=torch.long,
            device=device
        )
        head_sentence_lengths = torch.zeros(
            (num_heads, num_sentences),
            dtype=torch.long,
            device=device
        )
        
        # Create sentence range tensor
        sentences_ranges_tensor = torch.tensor(sentences_ranges, device=device)
        
        # Process all heads in batch
        all_indices_flat = all_indices[0]
        
        # Compute token-to-sentence mapping for all heads
        sentence_assignments = torch.searchsorted(
            sentences_ranges_tensor.unsqueeze(0).expand(num_heads, -1),
            all_indices_flat,
            right=True
        ) - 1
        sentence_assignments = sentence_assignments.clamp(0, num_sentences - 1)
        
        # Use scatter operations to aggregate tokens for each sentence
        for s_idx in range(num_sentences):
            mask = (sentence_assignments == s_idx)
            
            # Process all heads in parallel
            for h in range(num_heads):
                token_positions = torch.where(mask[h])[0]
                if len(token_positions) == 0:
                    continue
                
                # Store token indices
                num_tokens = min(len(token_positions), max_tokens_per_sentence)
                head_token_indices_tensor[h, s_idx, :num_tokens] = token_positions[:num_tokens]
                head_sentence_lengths[h, s_idx] = num_tokens
                
                # Compute mean(K) vectorized
                k_vecs = key_states[0, h, token_positions, :]
                mean_k = k_vecs.mean(dim=0).cpu()
                head_mean_k_tensor[h, s_idx, :] = mean_k
        
        return head_mean_k_tensor, head_token_indices_tensor, head_sentence_lengths


    def _parallel_sentence_mean_k(self, key_states, all_indices, sentences_ranges, num_sentences):
        """
        Fully parallelized mean(K) computation - completely loop-free version.
        """
        bsz, num_heads, compressed_seq_len, head_dim = key_states.shape
        device = key_states.device
        dtype = key_states.dtype
        
        if num_sentences > 0:
            est_max_len = (compressed_seq_len // num_sentences) * 2 + 50
            max_tokens_per_sentence = max(100, est_max_len)
        else:
            max_tokens_per_sentence = compressed_seq_len
        
        # Pre-allocate result tensors
        head_token_indices_tensor = torch.full(
            (num_heads, num_sentences, max_tokens_per_sentence),
            -1,
            dtype=torch.long,
            device=device
        )
        
        # Compute sentence assignments
        sentences_ranges_tensor = torch.tensor(sentences_ranges, device=device)
        all_indices_flat = all_indices[0]
        
        sentence_assignments = torch.searchsorted(
            sentences_ranges_tensor.unsqueeze(0).expand(num_heads, -1),
            all_indices_flat,
            right=True
        ) - 1
        sentence_assignments = sentence_assignments.clamp(0, num_sentences - 1)
        
        # Compute mean(K)
        one_hot = F.one_hot(sentence_assignments, num_classes=num_sentences).to(dtype)
        k_expanded = key_states[0]
        sum_k = torch.einsum('hsd,hsn->hnd', k_expanded, one_hot)
        counts = one_hot.sum(dim=1).clamp(min=1)
        mean_k = sum_k / counts.unsqueeze(-1)
        head_mean_k_tensor = mean_k
        head_sentence_lengths = counts.long()
        
        # Fully vectorized token indices filling - using scatter method
        # Flatten all dimensions for processing
        flat_assignments = sentence_assignments.reshape(-1)  # [num_heads * compressed_seq_len]
        flat_positions = torch.arange(compressed_seq_len, device=device).repeat(num_heads)
        flat_head_idx = torch.arange(num_heads, device=device).unsqueeze(1).expand(-1, compressed_seq_len).reshape(-1)
        
        # Use stable sort to maintain original order
        sort_keys = flat_head_idx * (num_sentences * compressed_seq_len) + flat_assignments * compressed_seq_len + flat_positions
        sorted_idx = sort_keys.argsort()
        
        sorted_heads = flat_head_idx[sorted_idx]
        sorted_sentences = flat_assignments[sorted_idx]
        sorted_positions = flat_positions[sorted_idx]
        
        # Compute each token's position within sentence - fully vectorized
        # Create unique (head, sentence) combination identifier
        group_ids = sorted_heads * num_sentences + sorted_sentences
        
        # Use diff to find group boundaries
        group_changes = torch.cat([
            torch.tensor([True], device=device),
            group_ids[1:] != group_ids[:-1]
        ])
        
        # Use cumsum and indexing tricks to compute within-group positions
        group_starts = torch.where(group_changes)[0]
        group_ids_expanded = torch.zeros(len(sorted_idx), dtype=torch.long, device=device)
        group_ids_expanded[group_starts] = 1
        group_ids_cumsum = group_ids_expanded.cumsum(0) - 1
        
        # Compute each element's position within its group
        positions_in_group = torch.arange(len(sorted_idx), device=device) - group_starts[group_ids_cumsum]
        
        # Only keep first max_tokens_per_sentence tokens
        valid_mask = positions_in_group < max_tokens_per_sentence
        
        # Fill result tensor
        if valid_mask.any():
            head_token_indices_tensor[
                sorted_heads[valid_mask],
                sorted_sentences[valid_mask],
                positions_in_group[valid_mask]
            ] = sorted_positions[valid_mask]
        
        # Truncate sentence lengths
        head_sentence_lengths = head_sentence_lengths.clamp(max=max_tokens_per_sentence)
        
        return head_mean_k_tensor, head_token_indices_tensor, head_sentence_lengths
    
    def _batch_similarity_computation(self, partial_q_view, head_mean_k_tensor, 
                                     head_token_indices_tensor, head_sentence_lengths,
                                     cutoff_val_left, cutoff_val_right, max_tokens_needed):

        num_heads, num_sentences, head_dim = head_mean_k_tensor.shape
        device = partial_q_view.device
        dtype = partial_q_view.dtype
        
        # Simplified tensor transfer since it's already on correct device with likely correct dtype
        mean_k_gpu = head_mean_k_tensor.to(dtype=dtype)
        
        similarities = torch.bmm(
            mean_k_gpu,
            partial_q_view.unsqueeze(-1)
        ).squeeze(-1)
        
        sorted_sims, sorted_indices = torch.sort(similarities, dim=1, descending=True)
        
        # Prepare batch token selection
        selected_tokens_list = []
        
        # Vectorized token selection
        for h in range(num_heads):
            sorted_indices_h = sorted_indices[h]
            
            # Batch retrieve tokens from all sentences
            all_tokens = head_token_indices_tensor[h, sorted_indices_h, :]
            all_lengths = head_sentence_lengths[h, sorted_indices_h]
            
            # Create validity mask
            position_indices = torch.arange(all_tokens.shape[-1], device=device)
            length_mask = position_indices.unsqueeze(0) < all_lengths.unsqueeze(-1)
            range_mask = (all_tokens > cutoff_val_left) & (all_tokens < cutoff_val_right)
            valid_mask = length_mask & range_mask
            
            # Extract valid tokens
            valid_tokens = all_tokens[valid_mask]
            if len(valid_tokens) > max_tokens_needed:
                valid_tokens = valid_tokens[:max_tokens_needed]
            
            selected_tokens_list.append(valid_tokens)
        
        del mean_k_gpu
        return selected_tokens_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        parent_model_instance = self.parent_model()
        if parent_model_instance is None:
            raise ValueError("Parent model reference is lost")
        input_ids = parent_model_instance.current_input_ids
        if input_ids is None:
            input_ids = self.config.current_input_ids
            self.generate = False
        
        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Layer index required for caching")
            if hasattr(self, "kv_seq_len"):
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if q_len > 1:
                is_prefill = True
            else:
                is_prefill = False
            
            if not is_prefill:
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states, key_states, value_states = (
                x.to(target_dtype) for x in [query_states, key_states, value_states]
            )

        # Prefill stage
        if is_prefill:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # Apply attention mask in prefill stage
            if attention_mask is not None:
                if attention_mask.size() != attn_weights.size():
                    attention_mask = attention_mask[:, :, :q_len, :kv_seq_len]
                attn_weights += attention_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            # SnapKV compression logic
            if kv_seq_len >= self.window_size:
                head_dim, num_heads = self.head_dim, self.num_heads
                last_window_q = query_states[..., -self.window_size:, :]
                attn_weights_temp = torch.matmul(last_window_q, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights_temp.dtype).min, device=attn_weights_temp.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights_temp.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                attn_weights_temp[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
                attn_weights_temp = F.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights_temp[..., :kv_seq_len - self.window_size].sum(dim=-2)
                pooling_fn = F.avg_pool1d if self.pooling == 'avgpool' else F.max_pool1d
                attn_cache = pooling_fn(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
                k_value = self.semantic_budget - self.window_size
                if k_value <= 0: k_value = 1
                if k_value > attn_cache.size(-1): k_value = attn_cache.size(-1)
                topk_indices = attn_cache.topk(k_value, dim=-1).indices
                topk_indices, _ = torch.sort(topk_indices, dim=-1)
                k_past, v_past = key_states[..., :kv_seq_len - self.window_size, :], value_states[..., :kv_seq_len - self.window_size, :]
                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress, v_past_compress = k_past.gather(dim=2, index=topk_indices_expanded), v_past.gather(dim=2, index=topk_indices_expanded)
                k_cur, v_cur = key_states[..., kv_seq_len - self.window_size:, :], value_states[..., kv_seq_len - self.window_size:, :]
                key_states, value_states = torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.kv_seq_len = past_key_value.seen_tokens
                cur_range = torch.arange(kv_seq_len - self.window_size, kv_seq_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, self.window_size)
                all_indices = torch.cat([topk_indices, cur_range], dim=-1)
            else:
                 past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                 self.kv_seq_len = past_key_value.seen_tokens
            

            seq_len_input = input_ids.size(1)
            punctuation_positions = [idx for idx, token_id in enumerate(input_ids[0]) if token_id.item() in self.punctuations_ids]
            sentences_ranges = [0] + [pos + 1 for pos in punctuation_positions]
            if not sentences_ranges or sentences_ranges[-1] < seq_len_input:
                sentences_ranges.append(seq_len_input)
            num_sentences = len(sentences_ranges) - 1
            
            if kv_seq_len < self.window_size:
                all_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_heads, -1)
            
            self.head_mean_k_tensor, self.head_token_indices_tensor, self.head_sentence_lengths = \
                self._parallel_sentence_mean_k(key_states, all_indices, sentences_ranges, num_sentences)
            
            self.sentence_cache = []
            self.token_index_in_sentence = past_key_value.seen_tokens

        # Decoding stage
        else:
            attn_weights_full = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # Apply attention mask in decoding stage
            if attention_mask is not None:
                if attention_mask.shape[-1] != kv_seq_len:
                    attention_mask = attention_mask[:, :, -q_len:, :kv_seq_len]
                attn_weights_full += attention_mask

            local_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
            bucket_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
            total_processed = kv_seq_len - q_len
            
            # Decoding loop
            for seq_idx in range(q_len):
                global_idx = total_processed + seq_idx
                token_id = input_ids[0, -1] if self.generate else input_ids[0, seq_idx]
                new_token_q_state = query_states[0, :, seq_idx, :]
                self.sentence_cache.append((new_token_q_state.contiguous().view(-1), global_idx))
                if token_id in self.punctuations_ids and self.sentence_cache:
                    sentence_len = len(self.sentence_cache)
                    start_idx = global_idx - sentence_len + 1
                    k_indices = torch.arange(start_idx, global_idx + 1, device=key_states.device)
                    k_selected = key_states[0, :, k_indices, :]
                    per_head_mean_k = k_selected.mean(dim=1)
                    self.head_mean_k_tensor = torch.cat([self.head_mean_k_tensor, per_head_mean_k.unsqueeze(1)], dim=1)
                    token_indices = [item[1] for item in self.sentence_cache]
                    num_new_tokens, max_len = len(token_indices), self.head_token_indices_tensor.size(-1)
                    new_sentence_data = torch.full((self.num_heads, 1, max_len), -1, dtype=torch.long, device=key_states.device)
                    if num_new_tokens <= max_len:
                        new_sentence_data[:, 0, :num_new_tokens] = torch.tensor(token_indices, device=key_states.device)
                    self.head_token_indices_tensor = torch.cat([self.head_token_indices_tensor, new_sentence_data], dim=1)
                    new_lengths = torch.full((self.num_heads, 1), min(num_new_tokens, max_len), dtype=torch.long, device=key_states.device)
                    self.head_sentence_lengths = torch.cat([self.head_sentence_lengths, new_lengths], dim=1)
                    self.sentence_cache = []
                full_seq_idx = kv_seq_len - q_len + seq_idx
                if len(self.sentence_cache) <= self.length_threshold:
                    recent_indices = torch.arange(max(0, full_seq_idx - self.recent_budget + 1), full_seq_idx + 1)
                    start_indices = torch.arange(0, min(self.start_budget, kv_seq_len))
                    positions_local = torch.cat([start_indices, recent_indices]).unique()
                    local_attention_mask[0, :, seq_idx, positions_local.to(local_attention_mask.device)] = True
                    bucket_attention_mask[0, :, seq_idx, :] = True
                else:
                    q_list_partial = [item[0] for item in self.sentence_cache]
                    partial_q = torch.mean(torch.stack(q_list_partial, dim=0), dim=0).view(self.num_heads, self.head_dim)
                    max_tokens_needed = self.token_budget - self.recent_budget - self.start_budget
                    cutoff_val_right = kv_seq_len - self.recent_budget
                    cutoff_val_left = self.start_budget
                    selected_tokens_list = self._batch_similarity_computation(
                        partial_q, self.head_mean_k_tensor,
                        self.head_token_indices_tensor, self.head_sentence_lengths,
                        cutoff_val_left, cutoff_val_right, max_tokens_needed
                    )
                    for h, valid_tokens in enumerate(selected_tokens_list):
                        if len(valid_tokens) > 0:
                            bucket_attention_mask[0, h, seq_idx, valid_tokens] = True
                    recent_indices = torch.arange(max(0, full_seq_idx - self.recent_budget + 1), full_seq_idx + 1)
                    start_indices = torch.arange(0, min(self.start_budget, kv_seq_len))
                    positions_local = torch.cat([start_indices, recent_indices]).unique()
                    local_attention_mask[0, :, seq_idx, positions_local.to(local_attention_mask.device)] = True
            
            bucket_attention_mask = bucket_attention_mask & ~local_attention_mask
            attn_weights = torch.full_like(attn_weights_full, float('-inf'))
            attn_weights[local_attention_mask] = attn_weights_full[local_attention_mask]
            attn_weights[bucket_attention_mask] = attn_weights_full[bucket_attention_mask]

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)


        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

            
class LlamaFlashAttention2(LlamaAttention):
    """Optimized version using vectorized operations to reduce loops and improve parallel processing"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
        self.max_tokens_per_sentence = 256  
        self.current_sentence_q_sum = None
        self.current_sentence_len = 0
        
        self.head_mean_k_tensor = None
        self.head_token_indices_tensor = None
        self.head_sentence_lengths = None


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        

        parent_model_instance = self.parent_model()
        if parent_model_instance is None:
            raise ValueError("Parent model reference is lost")
        input_ids = parent_model_instance.current_input_ids
        if input_ids is None:
            input_ids = self.config.current_input_ids
            self.generate = False

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Layer index required for caching")
            if hasattr(self, "kv_seq_len"):
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if key_states.shape[-2] == kv_seq_len:
                is_prefill = True
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
                dropout_rate = self.attention_dropout if self.training else 0.0
            else:
                is_prefill = False
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Prefill stage - using optimized parallel processing
        if is_prefill:
            with torch.no_grad():
                # Flash attention computation
                attn_output = _flash_attention_forward(
                    query_states, key_states, value_states, attention_mask,
                    q_len, position_ids=position_ids, dropout=dropout_rate,
                    sliding_window=getattr(self, "sliding_window", None),
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=self.is_causal,
                )
                attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
                attn_output = self.o_proj(attn_output)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                if kv_seq_len < self.window_size:
                    past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                    self.kv_seq_len = past_key_value.seen_tokens
                else:
                    head_dim = self.head_dim
                    num_heads = self.num_heads
                    
                    last_window_q = query_states[..., -self.window_size:, :]
                    attn_weights_temp = torch.matmul(last_window_q, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                    
                    mask = torch.full(
                        (self.window_size, self.window_size),
                        torch.finfo(attn_weights_temp.dtype).min,
                        device=attn_weights_temp.device
                    )
                    mask_cond = torch.arange(mask.size(-1), device=attn_weights_temp.device)
                    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                    attn_weights_temp[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
                    
                    attn_weights_temp = F.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    attn_weights_sum = attn_weights_temp[..., :kv_seq_len - self.window_size].sum(dim=-2)
                    
                    if self.pooling == 'avgpool':
                        attn_cache = F.avg_pool1d(
                            attn_weights_sum,
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2,
                            stride=1
                        )
                    elif self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(
                            attn_weights_sum,
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2,
                            stride=1
                        )
                    else:
                        raise ValueError('Pooling method not supported')
                    
                    k_value = self.semantic_budget - self.window_size
                    if k_value <= 0:
                        raise ValueError(f"Invalid topk value: {k_value}")
                    if k_value > attn_cache.size(-1):
                        k_value = attn_cache.size(-1)
                    
                    topk_indices = attn_cache.topk(k_value, dim=-1).indices
                    topk_indices, _ = torch.sort(topk_indices, dim=-1)
                    
                    k_past = key_states[..., :kv_seq_len - self.window_size, :]
                    v_past = value_states[..., :kv_seq_len - self.window_size, :]
                    
                    topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k_past_compress = k_past.gather(dim=2, index=topk_indices_expanded)
                    v_past_compress = v_past.gather(dim=2, index=topk_indices_expanded)
                    
                    k_cur = key_states[..., kv_seq_len - self.window_size:, :]
                    v_cur = value_states[..., kv_seq_len - self.window_size:, :]
                    
                    key_states = torch.cat([k_past_compress, k_cur], dim=2)
                    value_states = torch.cat([v_past_compress, v_cur], dim=2)
                    
                    del attn_weights_temp, k_past, v_past, k_past_compress, v_past_compress, k_cur, v_cur
                    torch.cuda.empty_cache()
                    
                    past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                    self.kv_seq_len = past_key_value.seen_tokens
                    
                    topk_indices_orig = topk_indices
                    cur_range = torch.arange(
                        kv_seq_len - self.window_size, kv_seq_len,
                        device=key_states.device
                    ).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, self.window_size)
                    all_indices = torch.cat([topk_indices_orig, cur_range], dim=-1)

                # Sentence segmentation and mean(K) computation - using optimized parallel version
                seq_len_input = input_ids.size(1)
                punctuation_positions = []
                for idx in range(seq_len_input):
                    if input_ids[0, idx].item() in self.punctuations_ids:
                        punctuation_positions.append(idx)

                sentences_ranges = [0] + [idx_ + 1 for idx_ in punctuation_positions]
                if sentences_ranges[-1] < seq_len_input:
                    sentences_ranges.append(seq_len_input)
                num_sentences = len(sentences_ranges) - 1

                # Use optimized parallel processing function
                if kv_seq_len >= self.window_size:
                    self.head_mean_k_tensor, self.head_token_indices_tensor, self.head_sentence_lengths = \
                        self._parallel_sentence_mean_k(key_states, all_indices, sentences_ranges, num_sentences)

                else:
                    compressed_seq_len = key_states.shape[2]
                    bsz, num_heads, _, head_dim = key_states.shape
                    all_indices = torch.arange(
                        kv_seq_len, device=key_states.device
                    ).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, kv_seq_len)
                    
                    self.head_mean_k_tensor, self.head_token_indices_tensor, self.head_sentence_lengths = \
                        self._parallel_sentence_mean_k(key_states, all_indices, sentences_ranges, num_sentences)

                self.sentence_cache = []
                self.token_index_in_sentence = past_key_value.seen_tokens

        # Decoding stage - optimized batch processing
        else:
            with torch.no_grad():
                attn_weights_full = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                if attention_mask is not None:
                    if attention_mask.shape[-1] != kv_seq_len:
                        causal_mask = attention_mask[:, :, :, :kv_seq_len]
                        attn_weights_full += causal_mask
                    else:
                        attn_weights_full += attention_mask
                
                local_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
                bucket_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
                total_processed = kv_seq_len - q_len

                # Decoding loop
                for seq_idx in range(q_len):
                    global_idx = total_processed + seq_idx
                    if self.generate == True:
                        token_id = input_ids[0, -1]
                    else:
                        token_id = input_ids[0, seq_idx]
                    
                    num_heads = self.num_heads
                    head_dim = self.head_dim
                    

                    new_token_q_state = query_states[0, :, seq_idx, :]
                    q_state_flat = new_token_q_state.contiguous().view(-1)
                    self.sentence_cache.append((q_state_flat, self.token_index_in_sentence))
                    self.token_index_in_sentence += 1
                    

                    if token_id in self.punctuations_ids:
                        q_list = [item[0] for item in self.sentence_cache]
                        if len(q_list) > 0:

                            stacked_q = torch.stack(q_list, dim=0)
                            sentence_len = stacked_q.size(0)
                            start_idx = global_idx - sentence_len + 1
                            
                            k_indices = torch.arange(start_idx, global_idx + 1, device=key_states.device)
                            k_selected = key_states[0, :, k_indices, :]
                            per_head_mean_k = k_selected.mean(dim=1)
                            
                            token_indices = [it[1] for it in self.sentence_cache]
                            new_mean_k = per_head_mean_k.unsqueeze(1)
                            self.head_mean_k_tensor = torch.cat([self.head_mean_k_tensor, new_mean_k], dim=1)
                            

                            num_new_tokens = len(token_indices)
                            new_sentence_data = torch.full(
                                (num_heads, 1, self.head_token_indices_tensor.size(-1)),
                                -1, dtype=torch.long, device=key_states.device
                            )
                            if num_new_tokens <= self.head_token_indices_tensor.size(-1):
                                new_sentence_data[:, 0, :num_new_tokens] = torch.tensor(token_indices, device=key_states.device)
                            
                            self.head_token_indices_tensor = torch.cat([self.head_token_indices_tensor, new_sentence_data], dim=1)
                            
                            new_lengths = torch.full((num_heads, 1), min(num_new_tokens, self.head_token_indices_tensor.size(-1)),
                                                dtype=torch.long, device=key_states.device)
                            self.head_sentence_lengths = torch.cat([self.head_sentence_lengths, new_lengths], dim=1)
                        
                        self.sentence_cache = []
                        self.token_index_in_sentence = past_key_value.seen_tokens
                    

                    full_seq_idx = kv_seq_len - q_len + seq_idx
                    start_budget = self.start_budget
                    recent_budget = self.recent_budget
                    
                    if len(self.sentence_cache) <= self.length_threshold:
                        # Local attention
                        recent_indices = torch.arange(
                            max(0, full_seq_idx - recent_budget + 1),
                            full_seq_idx + 1,
                            device=attn_weights_full.device
                        )
                        start_indices = torch.arange(
                            0,
                            min(start_budget, kv_seq_len),
                            device=attn_weights_full.device
                        )
                        positions_local = torch.cat([start_indices, recent_indices]).unique()
                        local_attention_mask[0, :, seq_idx, positions_local] = True
                        bucket_attention_mask[0, :, seq_idx, :] = True
                    else:
                        # Use optimized batch similarity computation
                        
                        q_list_partial = [item[0] for item in self.sentence_cache]
                        partial_q = torch.mean(torch.stack(q_list_partial, dim=0), dim=0)
                        partial_q_view = partial_q.view(num_heads, head_dim)
                        
                        if getattr(self, "token_budget", None) is not None:
                            max_tokens_needed = self.token_budget - self.recent_budget - self.start_budget
                        else:
                            max_tokens_needed = 999999999999
                        
                        cutoff_val_right = kv_seq_len - self.recent_budget
                        cutoff_val_left = self.start_budget
                        

                        selected_tokens_list = self._batch_similarity_computation(
                            partial_q_view, self.head_mean_k_tensor,
                            self.head_token_indices_tensor, self.head_sentence_lengths,
                            cutoff_val_left, cutoff_val_right, max_tokens_needed
                        )

                        for h, valid_tokens in enumerate(selected_tokens_list):
                            if len(valid_tokens) > 0:
                                bucket_attention_mask[0, h, seq_idx, valid_tokens] = True
                        

                        recent_indices = torch.arange(
                            max(0, full_seq_idx - recent_budget + 1),
                            full_seq_idx + 1,
                            device=attn_weights_full.device
                        )
                        start_indices = torch.arange(
                            0,
                            min(start_budget, kv_seq_len),
                            device=attn_weights_full.device
                        )
                        positions_local = torch.cat([start_indices, recent_indices]).unique()
                        local_attention_mask[0, :, seq_idx, positions_local] = True
                

                bucket_attention_mask = bucket_attention_mask & ~local_attention_mask
                attn_weights = torch.full_like(attn_weights_full, float('-inf'))
                attn_weights[local_attention_mask] = attn_weights_full[local_attention_mask]
                attn_weights[bucket_attention_mask] = attn_weights_full[bucket_attention_mask]
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, value_states)
                

                attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class LlamaSdpaAttention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        parent_model_instance = self.parent_model()
        if parent_model_instance is None:
            raise ValueError("Parent model reference is lost")
        input_ids = parent_model_instance.current_input_ids
        if input_ids is None:
            input_ids = self.config.current_input_ids
            self.generate = False

        if output_attentions:
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("Layer index required for caching")
            if hasattr(self, "kv_seq_len"):
                if self.kv_seq_len != 0:
                    kv_seq_len += self.kv_seq_len
                else:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
            
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            if key_states.shape[-2] == kv_seq_len:
                is_prefill = True
            else:
                is_prefill = False
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states, key_states, value_states = (
                x.to(target_dtype) for x in [query_states, key_states, value_states]
            )


        if is_prefill:

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            if kv_seq_len >= self.window_size:
                head_dim, num_heads = self.head_dim, self.num_heads
                last_window_q = query_states[..., -self.window_size:, :]
                attn_weights_temp = torch.matmul(last_window_q, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights_temp.dtype).min, device=attn_weights_temp.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights_temp.device)
                mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                attn_weights_temp[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
                attn_weights_temp = F.softmax(attn_weights_temp, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights_temp[..., :kv_seq_len - self.window_size].sum(dim=-2)
                
                pooling_fn = F.avg_pool1d if self.pooling == 'avgpool' else F.max_pool1d
                attn_cache = pooling_fn(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
                
                k_value = self.semantic_budget - self.window_size
                if k_value <= 0: k_value = 1
                if k_value > attn_cache.size(-1): k_value = attn_cache.size(-1)
                
                topk_indices = attn_cache.topk(k_value, dim=-1).indices
                topk_indices, _ = torch.sort(topk_indices, dim=-1)

                k_past, v_past = key_states[..., :kv_seq_len - self.window_size, :], value_states[..., :kv_seq_len - self.window_size, :]
                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                k_past_compress, v_past_compress = k_past.gather(dim=2, index=topk_indices_expanded), v_past.gather(dim=2, index=topk_indices_expanded)
                
                k_cur, v_cur = key_states[..., kv_seq_len - self.window_size:, :], value_states[..., kv_seq_len - self.window_size:, :]
                
                key_states, value_states = torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)
                
                past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.kv_seq_len = past_key_value.seen_tokens

                cur_range = torch.arange(kv_seq_len - self.window_size, kv_seq_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, self.window_size)
                all_indices = torch.cat([topk_indices, cur_range], dim=-1)
            else:
                 past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                 self.kv_seq_len = past_key_value.seen_tokens


            seq_len_input = input_ids.size(1)
            punctuation_positions = [idx for idx, token_id in enumerate(input_ids[0]) if token_id.item() in self.punctuations_ids]
            sentences_ranges = [0] + [pos + 1 for pos in punctuation_positions]
            if not sentences_ranges or sentences_ranges[-1] < seq_len_input:
                sentences_ranges.append(seq_len_input)
            num_sentences = len(sentences_ranges) - 1

            if kv_seq_len < self.window_size:
                all_indices = torch.arange(key_states.shape[2], device=key_states.device).unsqueeze(0).unsqueeze(0).expand(bsz, self.num_heads, -1)


            self.head_mean_k_tensor, self.head_token_indices_tensor, self.head_sentence_lengths = \
                self._parallel_sentence_mean_k(key_states, all_indices, sentences_ranges, num_sentences)
            
            self.sentence_cache = []
            self.token_index_in_sentence = past_key_value.seen_tokens


        else:
          
            attn_weights_full = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights_full += attention_mask

            local_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
            bucket_attention_mask = torch.zeros_like(attn_weights_full, dtype=torch.bool)
            total_processed = kv_seq_len - q_len
            
            for seq_idx in range(q_len):
                global_idx = total_processed + seq_idx
                token_id = input_ids[0, -1] if self.generate else input_ids[0, seq_idx]
                
                new_token_q_state = query_states[0, :, seq_idx, :]
                self.sentence_cache.append((new_token_q_state.contiguous().view(-1), global_idx))
                
                if token_id in self.punctuations_ids and self.sentence_cache:
                    sentence_len = len(self.sentence_cache)
                    start_idx = global_idx - sentence_len + 1
                    
                    k_indices = torch.arange(start_idx, global_idx + 1, device=key_states.device)
                    k_selected = key_states[0, :, k_indices, :]
                    per_head_mean_k = k_selected.mean(dim=1)
                    
                    self.head_mean_k_tensor = torch.cat([self.head_mean_k_tensor, per_head_mean_k.unsqueeze(1)], dim=1)
                    
                    token_indices = [item[1] for item in self.sentence_cache]
                    num_new_tokens, max_len = len(token_indices), self.head_token_indices_tensor.size(-1)
                    new_sentence_data = torch.full((self.num_heads, 1, max_len), -1, dtype=torch.long, device=key_states.device)
                    if num_new_tokens <= max_len:
                        new_sentence_data[:, 0, :num_new_tokens] = torch.tensor(token_indices, device=key_states.device)
                    
                    self.head_token_indices_tensor = torch.cat([self.head_token_indices_tensor, new_sentence_data], dim=1)
                    
                    new_lengths = torch.full((self.num_heads, 1), min(num_new_tokens, max_len), dtype=torch.long, device=key_states.device)
                    self.head_sentence_lengths = torch.cat([self.head_sentence_lengths, new_lengths], dim=1)
                    self.sentence_cache = []

                full_seq_idx = kv_seq_len - q_len + seq_idx
                if len(self.sentence_cache) <= self.length_threshold:
                    recent_indices = torch.arange(max(0, full_seq_idx - self.recent_budget + 1), full_seq_idx + 1)
                    start_indices = torch.arange(0, min(self.start_budget, kv_seq_len))
                    local_indices = torch.cat([start_indices, recent_indices]).unique()
                    local_attention_mask[0, :, seq_idx, local_indices.to(local_attention_mask.device)] = True
                    bucket_attention_mask[0, :, seq_idx, :] = True
                else:
                    q_list_partial = [item[0] for item in self.sentence_cache]
                    partial_q = torch.mean(torch.stack(q_list_partial, dim=0), dim=0).view(self.num_heads, self.head_dim)
                    
                    max_tokens_needed = self.token_budget - self.recent_budget - self.start_budget
                    cutoff_val_right = kv_seq_len - self.recent_budget
                    cutoff_val_left = self.start_budget

                    selected_tokens_list = self._batch_similarity_computation(
                        partial_q, self.head_mean_k_tensor,
                        self.head_token_indices_tensor, self.head_sentence_lengths,
                        cutoff_val_left, cutoff_val_right, max_tokens_needed
                    )
                    
                    for h, valid_tokens in enumerate(selected_tokens_list):
                        if len(valid_tokens) > 0:
                            bucket_attention_mask[0, h, seq_idx, valid_tokens] = True
                    
                    recent_indices = torch.arange(max(0, full_seq_idx - self.recent_budget + 1), full_seq_idx + 1)
                    start_indices = torch.arange(0, min(self.start_budget, kv_seq_len))
                    local_indices = torch.cat([start_indices, recent_indices]).unique()
                    local_attention_mask[0, :, seq_idx, local_indices.to(local_attention_mask.device)] = True
            
            bucket_attention_mask = bucket_attention_mask & ~local_attention_mask
            attn_weights = torch.full_like(attn_weights_full, float('-inf'))
            attn_weights[local_attention_mask] = attn_weights_full[local_attention_mask]
            attn_weights[bucket_attention_mask] = attn_weights_full[bucket_attention_mask]

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)


        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


        # 增加一个属性来存储当前输入序列
        self.current_input_ids = None  
        # 给每一层的self_attn设置parent_model为self，以便访问current_input_ids
        for layer in self.layers:
            layer.self_attn.parent_model = weakref.ref(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._full_input_ids = None  # 用于在生成时存储完整输入序列
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None:
                batch_size, sequence_length = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        
        # 更新_full_input_ids, 保持解码全程的序列记录
        # if past_key_values is None:
        if past_key_values.seen_tokens == 0:
            # 首步
            self._full_input_ids = input_ids
        else:
            # 后续步只追加最后一个token
            self._full_input_ids = torch.cat([self._full_input_ids, input_ids[:, -1:]], dim=1)

        # 将完整序列传给model的current_input_ids
        self.model.current_input_ids = self._full_input_ids
        return model_inputs

@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )