import os
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from QQQ.utils import (
    get_loaders,
    get_model_architecture,
    str2torch_device,
    find_layers,
    recurse_setattr,
    free_memory,
)
from .models import get_gptq_model_func
from .qlinear import QuantLinear



@torch.no_grad()
def apply_gptq(model, gptq_args, args, dataloader):
    """
    Run GPTQ calibration/packing with a 외부 dataloader.
    transformers 4.51+ / GPTQ 구버전 랩퍼 모두 호환하도록
    rotary embedding 을 런타임 패치한다.
    """
    
    print(f"[DEBUG] model type: {type(model)}")
    print(f"[DEBUG] model.model type: {type(getattr(model, 'model', None))}")
    # ── 1. 환경 & 설정 ──────────────────────────────────────────────
    device = str2torch_device(args.device)
    gptq_config = dict(
        sym          = gptq_args.sym,
        groupsize    = gptq_args.groupsize,
        mse          = gptq_args.mse,
        act_order    = gptq_args.act_order,
        percdamp     = gptq_args.percdamp,
        wbits        = gptq_args.wbits,
        static_groups= gptq_args.static_groups,
        nsamples     = args.nsamples,
    )

    # ── 2. rotary_emb / rope API 호환 패치 ──────────────────────────
    from types import MethodType
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    import torch

    def _patch_rope(r: LlamaRotaryEmbedding):
        """단일 인스턴스에 새·구 API 모두 지원하도록 forward 교체."""
        if getattr(r, "_patched_for_seq_len", False):
            return r                      # 이미 패치됨
        orig = r.forward
#고치기전전
        # def new_forward(self, *a, **kw):
        #     # ▸ 최신 API : (seq_len, device, dtype)
        #     # if a and isinstance(a[0], int):
        #     #     return orig(*a, **kw)

        #     # ▸ 최신 API : (seq_len, [device], [dtype]) ─ 불필요한 추가 인자는 버린다
        #     if a and isinstance(a[0], int):
        #         seq_len = a[0]
        #         # try:
        #         #     # transformers ≥4.40
        #         #     return orig(seq_len, device=kw.get("device"), dtype=kw.get("dtype"))

        #         # ── ⚠️ device 인자가 torch.device 가 아닐 수도 있다 ──
        #         dev = kw.get("device")
        #         if not isinstance(dev, torch.device):
        #             dev = None            # ← 잘못된 타입이면 전달하지 않음
        #         try:                       # transformers ≥4.40
        #             return orig(seq_len, device=dev, dtype=kw.get("dtype"))
        #         except TypeError:
        #             try:                 # transformers 4.38–4.39
        #                 return orig(seq_len)
        #             except TypeError:    # 아주 옛 버전 (seq_len, position_ids)
        #                 return orig(seq_len, None)

        #     # ▸ 구 API : (hidden_states, position_ids)
        #     if a and isinstance(a[0], torch.Tensor):
        #         hidden = a[0]
        #         seq_len = hidden.shape[-2]

        #         # 1) (seq_len, device, dtype)
        #         try:
        #             return orig(seq_len,
        #                         device=hidden.device,
        #                         dtype=hidden.dtype)
        #         except TypeError:
        #             pass
        #         # 2) (seq_len)
        #         try:
        #             return orig(seq_len)
        #         except TypeError:
        #             pass
        #         # 3) (hidden, position_ids)
        #         bsz = hidden.shape[0]
        #         dummy_pid = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(bsz, -1)
        #         return orig(hidden, dummy_pid)

        #     # ▸ 키워드 전용 호출
        #             # ── 키워드 전용 호출 ─────────────────────────────
        #     if "seq_len" in kw:
        #         try:
        #             return orig(**kw)                 # 최신 impl
        #         except TypeError:
        #             kw.pop("device", None)
        #             kw.pop("dtype", None)
        #             try:
        #                 return orig(**kw)             # 중간 impl
        #             except TypeError:
        #                 # ▸▸▸ 가장 옛 impl (위치 인자) ▸▸▸
        #                 sl = kw["seq_len"]
        #                 try:
        #                     return orig(sl)           # (seq_len)
        #                 except TypeError:
        #                     return orig(sl, None)     # (seq_len, position_ids)


        #     raise TypeError("invalid rotary_emb call pattern")


        # QQQ/gptq/apply_gptq.py  안의  _patch_rope() ----
        # def new_forward(self, *a, **kw):
        #     # ── 모든 버전 공통: 첫 번째 인수가 seq_len (int) 인 경우 ──
        #     if a and isinstance(a[0], int):
        #         return orig(a[0])               # ← seq_len 하나만

        #     # ── 옛 API: (hidden_states, position_ids) ──
        #     if a and isinstance(a[0], torch.Tensor):
        #         seq_len = a[0].shape[-2]
        #         return orig(seq_len)            # ← seq_len 하나만

        #     # ── keyword 호출 : seq_len=... ──
        #     if "seq_len" in kw:
        #         return orig(kw["seq_len"])      # ← seq_len 하나만

        #     raise TypeError("invalid rotary_emb call pattern")
        # -------------------------------------------------
        def new_forward(self, *a, **kw):
            if a and isinstance(a[0], int):
                try:
                    return orig(a[0], device=kw.get("device"), dtype=kw.get("dtype"))
                except TypeError:
                    try:
                        return orig(a[0])
                    except TypeError:
                        dummy_hidden = torch.empty(1, a[0], self.dim, device=kw.get("device"))
                        dummy_pos = torch.arange(a[0], device=kw.get("device")).unsqueeze(0)
                        return orig(dummy_hidden, dummy_pos)
            
            if a and isinstance(a[0], torch.Tensor):
                seq_len = a[0].shape[-2]
                try:
                    return orig(seq_len, device=a[0].device, dtype=a[0].dtype)
                except TypeError:
                    try:
                        return orig(seq_len)
                    except TypeError:
                        dummy_pos = torch.arange(seq_len, device=a[0].device).unsqueeze(0)
                        return orig(a[0], dummy_pos)
            
            if "seq_len" in kw:
                sl = kw["seq_len"]
                try:
                    return orig(sl, device=kw.get("device"), dtype=kw.get("dtype"))
                except TypeError:
                    try:
                        return orig(sl)
                    except TypeError:
                        dummy_hidden = torch.empty(1, sl, self.dim, device=kw.get("device"))
                        dummy_pos = torch.arange(sl, device=kw.get("device")).unsqueeze(0)
                        return orig(dummy_hidden, dummy_pos)
            
            raise TypeError("invalid rotary_emb call pattern")





        r.forward = MethodType(new_forward, r)
        r._patched_for_seq_len = True
        return r

    # top-level
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = _patch_rope(model.model.rotary_emb)
    elif hasattr(model.model, "rope"):
        model.model.rope = _patch_rope(model.model.rope)

    # per-layer
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "rotary_emb"):
            attn.rotary_emb = _patch_rope(attn.rotary_emb)
        elif hasattr(attn, "rope"):
            attn.rope = _patch_rope(attn.rope)
    # ───────────────────────────────────────────────────────────────

    # ── 3. GPTQ 실행 ───────────────────────────────────────────────
    model_type = get_model_architecture(model.config)
    gptq_func  = get_gptq_model_func(model_type)

    quantizers = gptq_func(model, dataloader, device, gptq_config)
    print("디디디디버깅")
    print("디디디디버깅")
    print("디디디디버깅")
    print(f"GPTQ config: {gptq_config}")
    print(f"Quantizers found: {list(quantizers.keys())}")
    # ── 4. 결과 저장 & 모델 패킹 ──────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(quantizers, os.path.join(args.save_path, "quantizers.pth"))

    # from QQQ.gptq.pack import pack_model
    pack_model(
        model,
        quantizers,
        bits       = gptq_config["wbits"],
        group_size = gptq_config["groupsize"],
        force_layer_back_to_cpu=True,   # ← GPU 대신 CPU에서 pack

    )

    free_memory()
    return model



@torch.no_grad()
def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    force_layer_back_to_cpu: bool = False,
):
    CPU = torch.device("cpu")
    if force_layer_back_to_cpu:
        model.to(CPU)

    # logger.info("Packing model...")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
    )
    qlayers = find_layers(model, [QuantLinear])

    pbar = tqdm(qlayers.keys(), leave=True)
    for name in pbar:
        pbar.set_description(f"Packing {name}...", refresh=True)

        scale, zero, g_idx, scale_extra = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx, scale_extra = (
            layers[name].to(CPU),
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU),
            scale_extra.to(CPU) if scale_extra is not None else None,
        )
        qlayers[name].pack(layers[name], scale, scale_extra)
        qlayers[name].to(layer_device)
        del layers[name]
        free_memory()
    print("Model packed.")


def make_quant(
    module,
    names,
    bits,
    group_size,
    trainable: bool = False,
):
    if isinstance(module, QuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            new_layer = QuantLinear(
                bits,
                group_size,
                in_features,
                out_features,
                bias,
                trainable=trainable,
                weight_dtype=submodule.weight.dtype,
            )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))
