"""
Implementation for Marian architecture.
TODO: add docstring
"""

import dataclasses
import math
from typing import Any, Dict, Optional

from typing import Tuple
from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)

MAX_VOCAB_SIZE = 100000

"""
HuggingFace's implementation of MarianMT:https://github.com/huggingface/transformers/blob/main/src/transformers/models/marian/configuration_marian.py

"""


@dataclasses.dataclass
class MarianConfig(ConfigBase):
    vocab_size: int 
    encoder_layers: int
    encoder_attention_heads: int
    decoder_layers: int
    decoder_attention_heads: int
    decoder_ffn_dim: int
    encoder_ffn_dim: int
    d_model: int
    pad_token_id: int
    scale_embedding: bool
    max_position_embeddings: int
    max_length: int
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vocab_size = MAX_VOCAB_SIZE
        if self.d_model % self.decoder_attention_heads != 0:
            raise ValueError(
                f"d_model must be divisible by decoder_attention_heads (got `d_model`: {self.d_model}"
                f" and `decoder_attention_heads`: {self.decoder_attention_heads})."
            )
        if self.d_model % self.encoder_attention_heads != 0:
            raise ValueError(
                f"d_model must be divisible by encoder_attention_heads (got `d_model`: {self.d_model}"
                f" and `encoder_attention_heads`: {self.encoder_attention_heads})."
            )
        if self.context_window_size == 0:
            for name in ["n_positions", "max_length"]:
                if name in self.kwargs or hasattr(self, name):
                    self.context_window_size = (
                        self.kwargs.pop(name) if name in self.kwargs else getattr(self, name)
                    )
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maxmimum sequence length, because none of "
                    "`context_window_size`, `n_positions`, `max_sequence_length` or `max_target_positions` is "
                    "provided in `config.json`."
                )

            if self.prefill_chunk_size == 0:
                # chunk size same as context window size by default
                self.prefill_chunk_size = self.context_window_size


"""
Implementation for Marian architecture.
TODO: add docstring
"""

"""
HuggingFace's implementation of MarianMT:https://github.com/huggingface/transformers/blob/main/src/transformers/models/marian/configuration_marian.py

"""


class MarianPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.weight = nn.Parameter((max_seq_len, embed_dim))

    def forward(self, x: Tensor, offset: tir.Var):
        def te_op(x: te.Tensor, embed: te.Tensor, offset: tir.Var):
            """
            This function defines a tensor expression operation. It takes three arguments: `x`,
            `embed`, and `offset`. `x` is the input tensor, `embed` is the embedding tensor,
            and `offset` is the offset for the positional embedding.
            """

            def compute(i: tir.Var, j: tir.Var, k: tir.Var):
                """
                The `compute` function
                inside `te_op` returns the value of the embedding at the position `offset + j`
                for each element in the input tensor `x`
                """
                return embed[offset + j, k]

            """
            This function is used to create a new tensor by computing a function over its indices. 
            Here, it creates a new tensor with the same shape as `x` and the last dimension of `embed`,
            and the values are computed by the `compute` function.
            As a lambda function we can write it as
            te.compute([*x.shape, embed.shape[-1]], lambda i, j, k: embed[offset+j, k], name="position_embedding")
            """
            return te.compute([*x.shape, embed.shape[-1]], compute, name="position_embedding")

        """
        This function is used to create a new tensor by applying the tensor expression operation `
        te_op` to the input tensor `x`, the weight tensor `self.weight`, and the offset `offset`. 
        The result is a tensor with the same shape as `x` and the last dimension of `self.weight`, 
        and the values are the positional embeddings.
        """
        pos_embed = nn.tensor_expr_op(te_op, "position_embedding", args=[x, self.weight, offset])
        return pos_embed


class MarianAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, kv_cache_len: int, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if kv_cache_len > 0:
            self.k_cache = nn.KVCache(kv_cache_len, [self.num_heads, self.head_dim])
            self.v_cache = nn.KVCache(kv_cache_len, [self.num_heads, self.head_dim])

    def forward(
        self,
        hidden_states: Tensor,
        total_seq_len: Optional[tir.Var] = None,
        key_value_states: Optional[Tensor] = None,
        cached_cross_attn_states: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states and cached_cross_attn_states).

        Args:

            hidden_states: hidden states of the encoder or decoder. Shape: [bsz, seq_len, embed_dim]
            key_value_states: hidden states of the encoder.
                              These are the key and value vectors for the current attention block.
                              If the layer is being used for self-attention, key_value_states would
                              be the same as hidden_states. If the layer is being used for
                              cross-attention (like in a decoder attending to encoder outputs),
                              key_value_states would be the output from the encoder. cross attention
                              input to the layer of shape `(batch, seq_len, embed_dim)`

            cached_cross_attn_states: cached key and value states of the encoder. Tuple of two tensors(k and v) and each of shape [bsz, seq_len, num_heads, head_dim]
            attention_mask: attention mask. Shape: [bsz, seq_len, seq_len]
            total_seq_len: total length of the sequence. Used for caching the key and value states of the encoder.
        """

        is_cross_attention = key_value_states is not None or cached_cross_attn_states is not None

        h, d = self.num_heads, self.head_dim
        bsz, q_len, _ = hidden_states.shape
        assert bsz == 1, "Only support batch size 1 at this moment."

        q = nn.reshape(self.q_proj(hidden_states) * self.scaling, (bsz, q_len, h, d))

        dtype = q.dtype

        # initialize the cached_kv to 0
        def _initialize(q: Tensor):
            bsz, q_len, h, d = q.shape
            return te.compute([bsz, q_len, h, d], lambda i, j, k, l: 0)

        cached_kv = (
            op.tensor_expr_op(_initialize, name_hint="k", args=[q]),
            op.tensor_expr_op(_initialize, name_hint="v", args=[q]),
        )

        if is_cross_attention:
            # cross attention
            if cached_cross_attn_states is None:
                # no cache, cross attentions
                kv_len = key_value_states.shape[1]

                # Need to change the dtype of key_value_states to that for the quantizations
                key_value_states = key_value_states.astype(dtype)
                k = nn.reshape(self.k_proj(key_value_states), (bsz, kv_len, h, d))
                v = nn.reshape(self.v_proj(key_value_states), (bsz, kv_len, h, d))
                cached_kv = (k, v)

            else:
                # reuse cached k,v, cross_attentions
                k, v = cached_cross_attn_states

                # Need to chnage the dtype for the quantization
                k = k.astype(dtype)
                v = v.astype(dtype)

        else:
            # self attention
            k = nn.reshape(self.k_proj(hidden_states), (bsz, q_len, h, d))
            v = nn.reshape(self.v_proj(hidden_states), (bsz, q_len, h, d))

            if total_seq_len is not None:
                # reuse cached k, v, self_attention
                self.k_cache.append(nn.squeeze(k, axis=0))
                self.v_cache.append(nn.squeeze(v, axis=0))
                k = nn.reshape(self.k_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
                v = nn.reshape(self.v_cache.view(total_seq_len), (bsz, total_seq_len, h, d))
            else:
                # encode self attention, no cache
                # self attention
                ...

        q = nn.permute_dims(q, [0, 2, 1, 3])  # [b, h, q_len, d]
        k = nn.permute_dims(k, [0, 2, 1, 3])  # [b, h, q_len, d]
        v = nn.permute_dims(v, [0, 2, 1, 3])  # [b, h, q_len, d]

        attn_weights = nn.matmul(q, (nn.permute_dims(k, [0, 1, 3, 2])))  # [b, h, q_len, q_len]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        dtype = attn_weights.dtype
        attn_weights = attn_weights.maximum(tir.min_value(dtype))
        attn_weights = attn_weights.minimum(tir.max_value(dtype))
        if dtype == "float32":
            attn_weights = nn.softmax(attn_weights, axis=-1)
        else:
            attn_weights = nn.softmax(attn_weights.astype("float32"), axis=-1).astype(dtype)
        attn_output = nn.matmul(attn_weights, v)  # [b, h, q_len, d]

        attn_output = nn.permute_dims(attn_output, [0, 2, 1, 3])  # [b, q_len, h, d]
        attn_output = nn.reshape(attn_output, (bsz, q_len, self.embed_dim))  # [b, q_len, h * d]

        attn_output = self.out_proj(attn_output)
        # op.print_(attn_output)

        return attn_output, cached_kv


class EncoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            kv_cache_len=0,  # no need for kv_cache
            bias=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim, bias=True)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
            total_seq_len=None,
        )
        hidden_states = (
            residual + hidden_states
        )  # [bsz, seq_len, d_model] + [bsz, seq_len, d_model] -> [bsz, seq_len, d_model]
        hidden_states = self.self_attn_layer_norm(
            hidden_states
        )  # [bsz, seq_len, d_model] -> [bsz, seq_len, d_model]

        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = hidden_states.maximum(tir.min_value(hidden_states.dtype)).minimum(
            tir.max_value(hidden_states.dtype)
        )

        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            kv_cache_len=config.max_position_embeddings,  # kv_cache_len is set to max_position_embeddings
            bias=True,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            kv_cache_len=config.max_position_embeddings,  # kv_cache_len is set to max_position_embeddings
            bias=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim, bias=True)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim, bias=True)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cached_encoder_hidden_states: Tensor,
        total_seq_len: tir.Var,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor]]]:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=None,
            cached_cross_attn_states=None,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        hidden_states, cross_attn_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            total_seq_len=total_seq_len,
            key_value_states=encoder_hidden_states,
            cached_cross_attn_states=cached_encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = nn.silu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if cached_encoder_hidden_states is None:
            return hidden_states, cross_attn_key_value
        else:
            return hidden_states, None


class MarianEncoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(MAX_VOCAB_SIZE, config.d_model)
        self.embed_positions = MarianPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )

        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, input_ids: Tensor) -> Tensor:
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids, offset=0)

        hidden_states = inputs_embeds + embed_pos

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class MarianDecoder(nn.Module):
    def __init__(self, config: MarianConfig):
        super().__init__()

        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(MAX_VOCAB_SIZE, config.d_model)
        self.embed_positions = MarianPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])

    def forward(
        self,
        input_ids: Tensor,
        total_seq_len: Optional[tir.Var] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        cached_encoder_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        # total_seq_len = Length of generated tokens
        input_embeds = self.embed_tokens(input_ids) * self.embed_scale
        past_seq_len = total_seq_len - 1
        position_embeds = self.embed_positions(input_ids, offset=past_seq_len)

        hidden_states = input_embeds + position_embeds

        all_encoder_key_value = ()
        for idx, decoder_layer in enumerate(self.layers):
            ith_cached_encoder_key_value = (
                cached_encoder_key_value[idx] if cached_encoder_key_value is not None else None
            )
            hidden_states, encoder_key_value = decoder_layer(
                hidden_states=hidden_states,
                total_seq_len=total_seq_len,
                encoder_hidden_states=encoder_hidden_states,
                cached_encoder_hidden_states=ith_cached_encoder_key_value,
                attention_mask=attention_mask,
            )
            if cached_encoder_key_value is None:
                all_encoder_key_value += (encoder_key_value,)

        if cached_encoder_key_value is None:
            return hidden_states, all_encoder_key_value
        else:
            return hidden_states, None


class MarianModel(nn.Module):
    def __init__(self, config: MarianConfig):
        self.encoder = MarianEncoder(config)
        self.decoder = MarianDecoder(config)

    def forward(
        self,
        input_ids: Tensor,
        total_seq_len: Optional[tir.Var] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        cached_encoder_key_value: Optional[Tuple[Tuple[Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tuple[Tensor]]]]:
        encoder_hidden_states = self.encoder(input_ids)
        decoder_hidden_states, decoder_key_value = self.decoder(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=encoder_hidden_states,
            cached_encoder_key_value=cached_encoder_key_value,
            attention_mask=attention_mask,
        )
        return decoder_hidden_states, decoder_key_value


class MarianMT(nn.Module):
    def __init__(self, config: MarianConfig):
        self.config = config
        self.model = MarianModel(config)
        self.lm_head = nn.Linear(config.d_model, MAX_VOCAB_SIZE, bias=False)
        self.final_logits_bias = nn.Parameter((1, MAX_VOCAB_SIZE))
        self.dtype = "float32"

    def forward(self, input_ids: Tensor, total_seq_len: Optional[tir.Var] = None) -> Tensor:
        hidden_states, _ = self.model(input_ids=input_ids, total_seq_len=total_seq_len)

        def _index(x: te.Tensor):
            """
            x[:-1,:]. Extract the last hidden state of the sequence for each batch (x[i, seq_len - 1, k]).
            The shape (bsz, 1, d_embed) suggests that it reshapes the tensor to keep only the final
            hidden state for each item in the batch.
            """
            bsz, seq_len, d_embed = x.shape
            return te.compute(
                (bsz, 1, d_embed),
                lambda i, _, k: x[i, seq_len - 1, k],
                name="index",
            )

        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def encode(self, input_ids: Tensor) -> Tensor:
        return self.model.encoder(input_ids)

    # def softmax_with_temperature(self, logits: Tensor, temperature: Tensor):
    #     """Softmax."""
    #     return op.softmax(logits / temperature, axis=-1)

    def decode(
        self, input_ids: Tensor, total_seq_len: int, encoder_hidden_states: Tensor
    ) -> Tuple[Tensor, Tuple[Tuple[Tensor]]]:
        hidden_states, all_encoder_key_value = self.model.decoder(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=encoder_hidden_states,
            cached_encoder_key_value=None,
            attention_mask=None,
        )
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits, all_encoder_key_value

    def prefill(
        self, input_ids: Tensor, total_seq_len: int, cached_encoder_key_value: Tuple[Tuple[Tensor]]
    ) -> Tensor:
        hidden_states, _ = self.model.decoder.forward(
            input_ids=input_ids,
            total_seq_len=total_seq_len,
            encoder_hidden_states=None,
            cached_encoder_key_value=cached_encoder_key_value,
            attention_mask=None,
        )
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        return logits

    def get_default_spec(self):
        """Needed for ``export_tvm()``."""
        batch_size = 1

        mod_spec = {
            "prefill": {
                "input_ids": nn.spec.Tensor([batch_size, "seq_len"], "int32"),
                "total_seq_len": int,
                "cached_encoder_key_value": tuple(
                    tuple(
                        nn.spec.Tensor(
                            [
                                batch_size,
                                "source_seq_len",
                                self.config.decoder_attention_heads,
                                self.config.d_model // self.config.decoder_attention_heads,
                            ],
                            "float32",
                        )
                        for i2 in range(2)
                    )
                    for i1 in range(self.config.encoder_layers)
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "encode": {
                "input_ids": nn.spec.Tensor(
                    [batch_size, "seq_len"],
                    "int32",
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
            "decode": {
                "input_ids": nn.spec.Tensor([batch_size, 1], "int32"),
                "total_seq_len": int,
                "encoder_hidden_states": nn.spec.Tensor(
                    [batch_size, "source_seq_len", self.config.d_model],
                    "float32",
                ),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "packed",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
