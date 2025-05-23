""" PyTorch QuantizedLLaMA model."""
import math
import warnings
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from QQQ.smooth.quantization import Quantizer, QuantizedLayer, QuantizedModule
from QQQ.smooth.migration.migration_llama import migration

logger = logging.getLogger("QQQ")


class QuantizedLlamaMLP(LlamaMLP, QuantizedModule):
    def __init__(
        self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"
    ):
        super(LlamaMLP, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.hidden_size = org_module.hidden_size
        self.intermediate_size = org_module.intermediate_size
        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.gate_proj = QuantizedLayer(
            org_module.gate_proj, None, w_qconfig, a_qconfig, self.qinput
        )
        self.up_proj = QuantizedLayer(
            org_module.up_proj, None, w_qconfig, a_qconfig, self.qinput
        )
        if getattr(self.a_qconfig, "disable_down_proj", False):
            self.down_proj = org_module.mlp.down_proj
        else:
            self.a_qconfig.disable_down_proj = False
            self.down_proj = QuantizedLayer(
                org_module.down_proj, None, w_qconfig, a_qconfig, True
            )
        self.act_fn = org_module.act_fn

    def forward(self, hidden_states, **kwargs):
        observation_mask = kwargs["observation_mask"]
        if self.cac_migrate:
            logger.info(
                "the original min range is {}, the original max range is {}".format(
                    hidden_states.min(), hidden_states.max()
                )
            )

            # calculate scale
            weight_list = torch.cat(
                [self.gate_proj.module.weight, self.up_proj.module.weight]
            )
            extra_dict = {"observation_mask": observation_mask, "act_fn": self.act_fn}
            best_scale = migration(
                hidden_states,
                weight_list,
                self.a_qconfig,
                self.w_qconfig,
                "up_and_gate",
                extra_dict,
            )
            # update scale
            hidden_states /= best_scale
            self.gate_proj.module.weight.data *= best_scale
            self.up_proj.module.weight.data *= best_scale

        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)

        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(
            hidden_states
        )

        if not self.a_qconfig.disable_down_proj and self.cac_migrate:
            logger.info(
                "the original min range is {}, the original max range is {}".format(
                    hidden_states.min(), hidden_states.max()
                )
            )
            weight_list = torch.cat([self.down_proj.module.weight])
            extra_dict = {
                "observation_mask": observation_mask,
            }
            best_scale = migration(
                hidden_states,
                weight_list,
                self.a_qconfig,
                self.w_qconfig,
                "down_proj",
                extra_dict,
            )
            # update scale
            hidden_states /= best_scale
            self.down_proj.module.weight.data *= best_scale
        hidden_states = self.down_proj(hidden_states, observation_mask, 1)
        return hidden_states


# class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
#     def __init__(
#         self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"
#     ):
#         super(LlamaAttention, self).__init__()
#         QuantizedModule.__init__(self, backend=backend)
#         self.w_qconfig = w_qconfig
#         self.a_qconfig = a_qconfig
#         self.config = org_module.config
#         self.qinput = qinput
#         self.layer_idx = org_module.layer_idx

#         if self.layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )
#         self.attention_dropout = org_module.attention_dropout
#         self.hidden_size = org_module.hidden_size
#         self.num_heads = org_module.num_heads
#         self.head_dim = org_module.head_dim
#         self.num_key_value_heads = org_module.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = org_module.max_position_embeddings
#         self.rope_theta = org_module.rope_theta
#         self.is_causal = org_module.is_causal

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )

#         self.act_fake_quant = Quantizer(None, a_qconfig)
#         self.q_proj = QuantizedLayer(
#             org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput
#         )
#         self.k_proj = QuantizedLayer(
#             org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput
#         )
#         self.v_proj = QuantizedLayer(
#             org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput
#         )
#         self.o_proj = QuantizedLayer(
#             org_module.o_proj, None, w_qconfig, a_qconfig, True
#         )
#         self._init_rope()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         observation_mask = kwargs["observation_mask"]
#         bsz, q_len, _ = hidden_states.size()
#         cos, sin = self.rotary_emb(hidden_states, position_ids)
#         # gamma migration
#         causal = self._update_causal_mask(attention_mask, hidden_states, cache_position=cache_position)
        
#         if self.cac_migrate:
#             logger.info(
#                 "the original min range is {}, the original max range is {}".format(
#                     hidden_states.min(), hidden_states.max()
#                 )
#             )
#             # calculate scale
#             weight_list = torch.cat(
#                 [
#                     self.q_proj.module.weight,
#                     self.k_proj.module.weight,
#                     self.v_proj.module.weight,
#                 ]
#             )
#             extra_dict = {
#                 "num_heads": self.num_heads,
#                 "num_key_value_heads": self.num_key_value_heads,
#                 "num_key_value_groups": self.num_key_value_groups,
#                 "cos_cached": cos,
#                 "sin_cached": sin,
#                 "head_dim": self.head_dim,
#                 "position_ids": position_ids,
#                 "attention_mask": attention_mask[:, :, cache_position, :q_len],
#                 "observation_mask": observation_mask,
#             }
#             # update scale
#             best_scale = migration(
#                 hidden_states,
#                 weight_list,
#                 self.a_qconfig,
#                 self.w_qconfig,
#                 "qkv",
#                 extra_dict,
#             )
#             hidden_states /= best_scale
#             self.q_proj.module.weight.data *= best_scale
#             self.k_proj.module.weight.data *= best_scale
#             self.v_proj.module.weight.data *= best_scale

#         hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )
#         past_key_value = getattr(self, "past_key_value", past_key_value)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; position_ids needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(
#             query_states, key_states.transpose(2, 3)
#         ) / math.sqrt(self.head_dim)
            

#         if attention_mask is not None:
#             # 2D(attn_mask[B,S])면 (B,1,1,S)로, 4D면 그대로 사용
#             am = attention_mask if attention_mask.dim()==4 \
#                  else attention_mask[:,None,None,:]
#             # bool OR 연산 후 마스크 더하기
#             causal = causal | am.to(causal.dtype)
#             attn_weights = attn_weights + causal

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(
#             attn_weights, dim=-1, dtype=torch.float32
#         ).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(
#             attn_weights, p=self.attention_dropout, training=self.training
#         )
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         # out migration
#         if self.cac_migrate:
#             logger.info(
#                 "the original min range is {}, the original max range is {}".format(
#                     attn_output.min(), attn_output.max()
#                 )
#             )
#             weight_list = torch.cat([self.o_proj.module.weight])
#             extra_dict = {
#                 "observation_mask": observation_mask,
#             }
#             best_scale = migration(
#                 attn_output,
#                 weight_list,
#                 self.a_qconfig,
#                 self.w_qconfig,
#                 "o_proj",
#                 extra_dict,
#             )
#             # update scale
#             attn_output /= best_scale
#             self.o_proj.module.weight.data *= best_scale

#         attn_output = self.o_proj(attn_output, observation_mask, 1)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value

class QuantizedLlamaAttention(LlamaAttention, QuantizedModule):
    def __init__(self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"):
        super(LlamaAttention, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.config = org_module.config
        self.qinput = qinput
        self.layer_idx = org_module.layer_idx

        self.attention_dropout = org_module.attention_dropout
        self.hidden_size = org_module.hidden_size
        self.num_heads = org_module.num_heads
        self.head_dim = org_module.head_dim
        self.num_key_value_heads = org_module.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = org_module.max_position_embeddings
        self.rope_theta = org_module.rope_theta
        self.is_causal = org_module.is_causal

        self.act_fake_quant = Quantizer(None, a_qconfig)
        self.q_proj = QuantizedLayer(org_module.q_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.k_proj = QuantizedLayer(org_module.k_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.v_proj = QuantizedLayer(org_module.v_proj, None, w_qconfig, a_qconfig, self.qinput)
        self.o_proj = QuantizedLayer(org_module.o_proj, None, w_qconfig, a_qconfig, True)

        self._init_rope()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        observation_mask = kwargs.get("observation_mask", None)
        bsz, q_len, _ = hidden_states.size()

        # 1) RoPE 연산
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # 2) 인과 마스크(bool) 생성
        #    LlamaModel 또는 LlamaAttention에 붙인 _update_causal_mask을 호출
        causal = self._update_causal_mask(
            attention_mask,          # (B,S) 또는 (B,1,S,S)
            hidden_states,           
            cache_position=cache_position
        )                         # (1,1,S,S) bool

        # 3) GPTQ 마이그레이션(생략 가능)
        if self.cac_migrate:
            weight_list = torch.cat([self.q_proj.module.weight,
                                     self.k_proj.module.weight,
                                     self.v_proj.module.weight])
            extra = {
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "num_key_value_groups": self.num_key_value_groups,
                "cos_cached": cos, "sin_cached": sin,
                "head_dim": self.head_dim,
                "position_ids": position_ids,
                "attention_mask": attention_mask[:, :, cache_position, :q_len] if cache_position is not None else None,
                "observation_mask": observation_mask,
            }
            best_scale = migration(hidden_states, weight_list,
                                   self.a_qconfig, self.w_qconfig,
                                   "qkv", extra)
            hidden_states /= best_scale
            for proj in (self.q_proj, self.k_proj, self.v_proj):
                proj.module.weight.data *= best_scale

        # 4) quant + proj
        hidden_states = self.act_fake_quant(hidden_states, observation_mask, 1)
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) \
                       / math.sqrt(self.head_dim)

        # 5) 패딩 마스크 합치기(bool OR)
        if attention_mask is not None:
            am = attention_mask if attention_mask.dim() == 4 else attention_mask[:, None, None, :]
            causal = causal | am.to(causal)

        # 6) bool → float 마스크로 변환 후 더하기
        float_mask = (~causal).to(attn_weights.dtype) * -1e9
        attn_weights = attn_weights + float_mask

        # 7) softmax → dropout → output
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)

        # 8) Output projection & optional migration
        if self.cac_migrate:
            best_scale = migration(attn_output, self.o_proj.module.weight,
                                   self.a_qconfig, self.w_qconfig,
                                   "o_proj", {"observation_mask": observation_mask})
            attn_output /= best_scale
            self.o_proj.module.weight.data *= best_scale

        attn_output = self.o_proj(attn_output, observation_mask, 1)

        return attn_output, (attn_weights if output_attentions else None), past_key_value

class QuantizedLlamaDecoderLayer(LlamaDecoderLayer, QuantizedModule):
    def __init__(
        self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"
    ):
        super(LlamaDecoderLayer, self).__init__()
        QuantizedModule.__init__(self, backend=backend)
        self.w_qconfig = w_qconfig
        self.a_qconfig = a_qconfig
        self.qinput = qinput
        self.self_attn = QuantizedLlamaAttention(
            org_module.self_attn,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.mlp = QuantizedLlamaMLP(
            org_module.mlp,
            w_qconfig,
            a_qconfig,
            qinput=False,
        )
        self.input_layernorm = org_module.input_layernorm
        self.post_attention_layernorm = org_module.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QuantizedLlamaModel(LlamaModel, QuantizedModule):
    def __init__(
        self, org_module, w_qconfig, a_qconfig, qinput=True, backend="academic"
    ):
        super(LlamaModel, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self.qinput = qinput
        self.padding_idx = org_module.padding_idx
        self.vocab_size = org_module.vocab_size

        self.embed_tokens = org_module.embed_tokens
        self.layers = nn.ModuleList()
        for i in range(self.config.num_hidden_layers):
            self.layers.append(
                QuantizedLlamaDecoderLayer(
                    org_module.layers[i], w_qconfig, a_qconfig, qinput=True
                )
            )
        self.norm = org_module.norm
        self.gradient_checkpointing = False
        self.causal_mask = org_module.causal_mask

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
        observation_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        assert observation_mask is not None
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                observation_mask=observation_mask,
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

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantizedLlamaForCausalLM(LlamaForCausalLM, QuantizedModule):
    def __init__(
        self,
        org_module,
        w_qconfig,
        a_qconfig,
        qinput=True,
        backend="academic",
        is_remove_padding=False,
    ):
        super(LlamaForCausalLM, self).__init__(org_module.config)
        QuantizedModule.__init__(self, backend=backend)
        self._no_split_modules = [
            "QuantizedLlamaDecoderLayer",
            "QuantizedLlamaAttention",
            "QuantizedLlamaMLP",
            "QuantizedLayer",
            "QuantizedModule",
        ]
        self.qinput = qinput
        self.vocab_size = org_module.vocab_size
        self.model = QuantizedLlamaModel(
            org_module.model, w_qconfig, a_qconfig, self.qinput, backend=self.backend
        )
        self.lm_head = org_module.lm_head
        self.is_remove_padding = is_remove_padding

    def is_remove_padding(self, is_remove_padding=False):
        self.is_remove_padding = is_remove_padding

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

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.is_remove_padding and attention_mask is not None:
            observation_mask = attention_mask.clone()
        else:
            observation_mask = None

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
            observation_mask=observation_mask,
        )

        hidden_states = outputs[0]
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