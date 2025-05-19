import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    # LlamaFlashAttention2,
    # LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from typing import Optional
from ..qlinear import QuantLinear
from ..gptq import *
from ..quant import *
from QQQ.utils import find_layers
from transformers.utils import logging

logger = logging.get_logger(__name__)



@torch.no_grad()
def gptq_llama_func(model, dataloader, dev, config, force_to_cpu=False):
    print("Starting GPTQ quantization ...")

    # 기존 설정 보존
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # embed + norm 먼저 디바이스로 이동
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    # calibration을 위한 입력 수집
    inps, attention_mask, position_ids, cache_position = [], [], [], []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            print("[DEBUG] Catcher forward triggered")

            print(f"[Catcher] inp: {inp.shape if isinstance(inp, torch.Tensor) else type(inp)}")
            print(f"[Catcher] kwargs: {kwargs}")
            inps.append(inp)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            cache_position.append(kwargs.get("cache_position", None))
            raise ValueError

    # 첫 레이어에 훅 달아서 한 배치만 수집
    print(f"[DEBUG] Before attaching catcher: {type(layers[0])}")

    layers[0] = Catcher(layers[0])
    print(f"[DEBUG] After attaching catcher: {type(layers[0])}")

    # for batch in dataloader:
    #     try:
    #         model(*[b.to(dev) for b in batch])  # 튜플 전체 언패킹 & to(dev)

    #     except ValueError:
    #         break
    for batch in dataloader:
        print("[DEBUG] Running a batch through the model...")

        try:
            ids, mask, pos_ids = [x.to(dev) for x in batch]
            model(
                ids,
                attention_mask=mask,
                position_ids=pos_ids,
            )
        except ValueError:
            print("[DEBUG] Caught expected ValueError, stopping after 1 batch.")

            break    
            
        
    
    layers[0] = layers[0].module
    print(f"[DEBUG] Collected {len(inps)} input samples")

    if force_to_cpu:
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        torch.cuda.empty_cache()

    # 수집된 inps 복사
    outs = [inp.clone() for inp in inps]
    rope_top = (
        model.model.rotary_emb
        if hasattr(model.model, "rotary_emb")
        else model.model.layers[0].self_attn.rotary_emb
    )
    quantizers = {}
    # 각 레이어별로 GPTQ 실행
    for i, layer in enumerate(layers):
        # 레이어와 데이터를 동일 디바이스로 이동
        layer = layer.to(dev)
        cur_device = layer.input_layernorm.weight.device
        inps = [x.to(cur_device) for x in inps]
        outs = [x.to(cur_device) for x in outs]
        attention_mask = [am.to(cur_device) if am is not None else None for am in attention_mask]
        position_ids = [pid.to(cur_device) for pid in position_ids]
        cache_position = [cp.to(cur_device) if cp is not None else None for cp in cache_position]

        # find_layers 결과 전체를 훅 대상으로 사용
        full = find_layers(layer)
        subset = full
        print(f"[DEBUG] Layer {i}: found layers {list(subset.keys())}")

        # GPTQ 객체 초기화 및 설정
        gptq = {}
        for name, module in subset.items():
            gptq[name] = GPTQ(module)
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                config["wbits"],
                perchannel=True,
                sym=config["sym"],
                mse=config["mse"],
                groupsize=config["groupsize"],
            )

        # forward_hook 달기
        def make_hook(nm):
            def hook_fn(_, inp, out):
                gptq[nm].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = []
        for name, module in subset.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        n_calib = min(config["nsamples"], len(inps))  # ← 실제 배치 수만
        for j in range(n_calib):
        # for j in range(config["nsamples"]):
            # hidden = inps[j]                           # (batch=1, seq_len, hidden_dim)
            # pid    = position_ids[j]                   # (seq_len,)

            # # 1) cos, sin 얻기
            # cos, sin = model.model.rotary_emb(hidden, pid)
            # # 2) 차원 확장
            # cos = cos.unsqueeze(0).unsqueeze(0)        # (1,1,seq_len,head_dim)
            # sin = sin.unsqueeze(0).unsqueeze(0)
            hidden = inps[j]            # (bsz=1, seq_len, hidden_dim)
            pid    = position_ids[j]    # (seq_len,)

            # 1) cos, sin 얻기  ― 새 API는 seq_len만 요구
            # seq_len = hidden.shape[1]
            # cos, sin = model.model.rotary_emb(
            #     seq_len=seq_len,
            #     device=hidden.device,
            #     dtype=hidden.dtype,
            # )                
            # 1) cos, sin 얻기 (top-level rope 없으면 0-번 레이어 rope 사용)
            seq_len = hidden.shape[1]
            cos, sin = rope_top(
                seq_len=seq_len,
                device=hidden.device,
                dtype=hidden.dtype,
            )    
            # 3) 레이어에 넘겨주기 (position_embeddings 인자로 전달)
            outs[j] = layer(
                hidden,
                attention_mask=attention_mask[j],
                position_ids=pid,
                cache_position=cache_position[j],
                # position_embeddings=(cos, sin),
            )[0]            
            

        # 훅 제거
        for h in handles:
            h.remove()

        # quantization 수행
        for name in subset:
            scale, zero, g_idx, scale_extra = gptq[name].fasterquant(
                percdamp=config["percdamp"],
                groupsize=config["groupsize"],
                actorder=config["act_order"],
                static_groups=config["static_groups"],
            )
            quantizers[f"model.layers.{i}.{name}"] = (scale, zero, g_idx, scale_extra)
            print(f"[DEBUG] Quantized {name}: scale shape {scale.shape}, zero shape {zero.shape}")

            gptq[name].free()

        # 다음 레이어용 inps/outs 교환
        inps, outs = outs, inps

        if force_to_cpu:
            layer = layer.cpu()
            torch.cuda.empty_cache()
        layers[i] = layer

    model.config.use_cache = use_cache
    return quantizers

class QuantizedLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: dict,
        layer_idx: Optional[int] = None,
    ):
        super(LlamaAttention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
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

        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.q_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = QuantLinear(
            wbits,
            group_size,
            self.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()



class QuantizedLlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        super(LlamaMLP, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        group_size = quant_config["group_size"]
        wbits = quant_config["wbits"]
        self.gate_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = QuantLinear(
            wbits, group_size, self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = QuantLinear(
            wbits, group_size, self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]


class QuantizedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, quant_config: dict, layer_idx: int):
        super(LlamaDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        # self.self_attn = QUANT_LLAMA_ATTENTION_CLASSES[config._attn_implementation](
        #     config=config, quant_config=quant_config, layer_idx=layer_idx
        # )
        self.self_attn = QuantizedLlamaAttention(
            config=config, quant_config=quant_config, layer_idx=layer_idx
        )
        self.mlp = QuantizedLlamaMLP(config, quant_config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class QuantizedLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, quant_config: dict):
        super(LlamaModel, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # no quant on embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                QuantizedLlamaDecoderLayer(config, quant_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_position_embeddings`.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings),
            fill_value=True,
            dtype=torch.bool,
        )
        self.register_buffer(
            "causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False
        )
        # Initialize weights and apply final processing
        self.post_init()


class QuantizedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, quant_config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = QuantizedLlamaModel(config, quant_config)
        self.vocab_size = config.vocab_size
        # no quant on lm_head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
