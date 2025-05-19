import argparse
import logging
import os
# import torch

from QQQ.utils.model_utils import _llama_bool_causal_mask
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention

# LlamaModel 과 LlamaAttention, QuantizedLlamaAttention 에 모두 _update_causal_mask 붙이기
LlamaModel._update_causal_mask     = _llama_bool_causal_mask
LlamaAttention._update_causal_mask = _llama_bool_causal_mask

# QuantizedLlamaAttention 은 QQQ 코드 안에 클래스 정의가 있으니,
# 메서드만 추가로 붙여줍니다.
from QQQ.smooth.models.llama import QuantizedLlamaAttention
QuantizedLlamaAttention._update_causal_mask = _llama_bool_causal_mask


from types import MethodType
import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

def _safe_rope_forward(self, *a, **kw):
    # ----------------------------- 인자 해석 ----------------------------
    if a and isinstance(a[0], int):                   # (seq_len, …)
        seq_len, hidden = int(a[0]), None
    elif "seq_len" in kw:                             # (seq_len=…, …)
        seq_len, hidden = int(kw["seq_len"]), None
    elif a and isinstance(a[0], torch.Tensor):        # (hidden_states, …)
        hidden, seq_len = a[0], a[0].shape[-2]
    else:
        raise TypeError("invalid rotary_emb call pattern")
    # ------------------------------------------------------------------

    # ① 최신-API  (seq_len, device, dtype)
    try:
        dev = kw.get("device") if isinstance(kw.get("device"), torch.device) else None
        return self.__orig_forward(seq_len, device=dev, dtype=kw.get("dtype"))
    except TypeError:
        pass
    # ② 중간-API  (seq_len)
    try:
        return self.__orig_forward(seq_len)
    except TypeError:
        pass
    # ③ 구-API   (hidden_states, position_ids)
    if hidden is None:                       # 최신 호출 → dummy hidden 생성
        hidden = torch.zeros(1, seq_len, self.inv_freq.numel() * 2, device="cpu")
    dummy_pid = torch.arange(seq_len, dtype=torch.long, device=hidden.device).unsqueeze(0)
    return self.__orig_forward(hidden, dummy_pid)

# 한 번만 패치
if not getattr(LlamaRotaryEmbedding, "_patched_safe", False):
    LlamaRotaryEmbedding.__orig_forward = LlamaRotaryEmbedding.forward
    LlamaRotaryEmbedding.forward = _safe_rope_forward
    LlamaRotaryEmbedding._patched_safe = True


# ------------------------------------------------------------------




from QQQ.rotation import fuse_layer_norms, rotate_model
from QQQ.smooth import smooth, export_smoothed_model, quantize_model
from QQQ.gptq import apply_gptq
from QQQ.utils import (
    setup_seed,
    build_model_and_tokenizer,
    prepare_for_inference,
    free_memory,
    str2bool,
    remove_empty_parameters,
)
# from QQQ.datasets import get_wikitext2    # ← 이 줄을 추가

from datasets import load_dataset
from torch.utils.data import DataLoader
# QQQ/patches/rope_patch.py

def get_wikitext2(tokenizer, block_size, batch_size=8, split="train"):
    """
    GPTQ 보정을 위한 DataLoader 생성.
    - tokenizer: Huggingface tokenizer
    - block_size: 모델 max_position_embeddings 이하 길이
    - batch_size: calibration 배치 크기
    - split: "train" 또는 "validation"
    """
    # 1) wikitext-2 로우 데이터셋 로드
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # 2) 토크나이즈
    def tokenize_fn(examples):
        return tokenizer(examples["text"])
    tok_ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 3) block_size 단위로 시퀀스 분할
    def group_texts(examples):
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // block_size) * block_size
        chunks = [
            all_ids[i : i + block_size]
            for i in range(0, total_len, block_size)
        ]
        return {"input_ids": chunks}
    # lm_ds = tok_ds.map(group_texts, batched=True, remove_columns=["input_ids"])
    lm_ds = tok_ds.map(
        group_texts,
        batched=True,
        remove_columns=["input_ids", "attention_mask"],
    )
    # 4) DataLoader 반환
    # def collate_fn(batch):
    #     ids = torch.tensor(batch[0]["input_ids"], dtype=torch.long)
    #     mask = torch.ones_like(ids)  # 전부 1로 채움 (패딩 없는 경우)
    #     pos_ids = torch.arange(ids.shape[0], dtype=torch.long)  # 0, 1, 2, ...
    #     print(f"ids: {ids}, mask: {mask}, pos_ids: {pos_ids}")
    #     return (ids, mask, pos_ids)

    def collate_fn(batch):
        ids = torch.tensor(batch[0]["input_ids"], dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        mask = torch.ones_like(ids)  # [1, seq_len]
        pos_ids = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)  # [1, seq_len]
        print(f"ids: {ids.shape}, mask: {mask.shape}, pos_ids: {pos_ids.shape}")
        return (ids, mask, pos_ids)


    return DataLoader(
        lm_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )





logger = logging.getLogger("QQQ")


# NOTE(HandH1998): If enable smooth, it is recommended to use the default configuration, no need to change
def parse_a_qconfig(args):
    parser = argparse.ArgumentParser(
        description="Activation Quantization Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--a_quantizer",
        dest="quantizer",
        type=str,
        default="TokenFixedFakeQuantize",
        help="Quantizer for activation",
    )
    parser.add_argument(
        "--a_observer",
        dest="observer",
        type=str,
        default="MinMaxObserver",
        help="Observer for activation",
    )
    parser.add_argument(
        "--a_bit",
        dest="bit",
        type=int,
        default=8,
        help="Bit width for activation quantization",
    )
    parser.add_argument(
        "--a_symmetric",
        dest="symmetric",
        type=str2bool,
        default=True,
        help="Symmetric quantization for activation",
    )
    parser.add_argument(
        "--a_ch_axis",
        dest="ch_axis",
        type=int,
        default=0,
        help="Channel axis for activation quantization",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


# NOTE(HandH1998): If enable smooth, `w_quantizer=FixedQuantize, w_group_size=-1` is for weight per-channel quantizaiton,
# `w_quantizer=GroupFixedQuantize, w_group_size=128` is for weight per-group quantization.
# The other parameters can use the default configuration.
def parse_w_qconfig(args):
    parser = argparse.ArgumentParser(
        description="Weight Quantization Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--w_quantizer",
        dest="quantizer",
        type=str,
        default="FixedQuantize",
        choices=["FixedQuantize", "GroupFixedQuantize"],
        help="Quantizer for weights, (`FixedQuantize` for per-channel, `GroupFixedQuantize` for per-group)",
    )
    parser.add_argument(
        "--w_observer",
        dest="observer",
        type=str,
        default="MinMaxObserver",
        help="Observer for weights",
    )
    parser.add_argument(
        "--w_bit",
        dest="bit",
        type=int,
        default=4,
        help="Bit width for weight quantization",
    )
    parser.add_argument(
        "--w_symmetric",
        dest="symmetric",
        type=str2bool,
        default=True,
        help="Symmetric quantization for weights",
    )
    parser.add_argument(
        "--w_ch_axis",
        dest="ch_axis",
        type=int,
        default=0,
        help="Channel axis for weight quantization (0 for per-channel, -1 for per-layer)",
    )
    parser.add_argument(
        "--w_group_size",
        dest="group_size",
        type=int,
        default=-1,
        choices=[-1, 128],
        help="Group size for weight quantization (-1 for per-channel, 128 for per-group)",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


# NOTE(HandH1998): If enable smooth, the `calibrate_path` should be changed to your own data path. The other parameters can use the default configuration.
def parse_smooth_args(args):
    parser = argparse.ArgumentParser(
        description="Smooth Configuration Parser", add_help=False
    )
    # Calibration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size of smooth calibration inference",
    )

    # Padding removal
    parser.add_argument(
        "--is_remove_padding",
        dest="is_remove_padding",
        type=str2bool,
        default=True,
        help="Remove padding during quantization",
    )

    # Smooth method
    parser.add_argument(
        "--smooth_method", dest="smooth_method", type=str, default="os+"
    )
    smooth_args, remaining_args = parser.parse_known_args(args)
    smooth_args.a_qconfig, remaining_args = parse_a_qconfig(remaining_args)
    smooth_args.w_qconfig, remaining_args = parse_w_qconfig(remaining_args)
    return smooth_args, remaining_args


# NOTE(HandH1998): `gptq_mse=False` is for `Smooth + GPTQ`, `gptq_mse=True` is for `Rotation + GPTQ`.
# `gptq_groupsize=-1` is for per-channel weight quantization, `gptq_groupsize=128` is for per-group weight quantization
def parse_gptq_args(args):
    parser = argparse.ArgumentParser(
        description="GPTQ Configuration Parser", add_help=False
    )
    parser.add_argument(
        "--gptq_sym",
        dest="sym",
        type=str2bool,
        default=True,
        help="Symmetric quantization for GPTQ, only support sym for now",
    )
    parser.add_argument(
        "--gptq_groupsize",
        dest="groupsize",
        type=int,
        default=-1,
        choices=[-1, 128],
        help="Group size for GPTQ (-1 for per-channel, 128 for per-group), it should be same with w_group_size when enable smooth",
    )
    parser.add_argument(
        "--gptq_mse", dest="mse", type=str2bool, default=True, help="Use MSE for GPTQ"
    )
    parser.add_argument(
        "--gptq_act_order",
        dest="act_order",
        type=str2bool,
        default=True,
        help="Activation order for GPTQ",
    )
    parser.add_argument(
        "--gptq_percdamp",
        dest="percdamp",
        type=float,
        default=0.01,
        help="Percentage damping for GPTQ",
    )

    parser.add_argument(
        "--gptq_wbits",
        dest="wbits",
        type=int,
        default=4,
        help="Bit width for weights in GPTQ",
    )
    parser.add_argument(
        "--gptq_static_groups",
        dest="static_groups",
        type=str2bool,
        default=True,
        help="Use static groups for GPTQ",
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


def parse_rotation_args(args):
    parser = argparse.ArgumentParser(
        description="Rotation Configuration Parser", add_help=False
    )

    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )

    args, remaining_args = parser.parse_known_args(args)
    return args, remaining_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--smooth", type=str2bool, default=False)
    parser.add_argument("--rotation", type=str2bool, default=True)
    parser.add_argument("--max_length", dest="max_length", type=int, default=2048)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    # Calibration
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["wikitext2", "pile", "ptb", "new_ptb", "c4", "mix"],
        help="Calibration Dataset. If you want to use your own dataset, this should be the default value",
    )
    parser.add_argument(
        "--custom_dataset",
        type=str,
        default="",
        help="Custom Calibration Dataset. It should be your own dataset path. If you want to use the public dataset, this should be the default value",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration samples",
    )
    args, remaining_args = parser.parse_known_args()
    smooth_args, remaining_args = parse_smooth_args(remaining_args)
    gptq_args, remaining_args = parse_gptq_args(remaining_args)
    rotation_args, remaining_args = parse_rotation_args(remaining_args)
    return args, smooth_args, gptq_args, rotation_args


@torch.no_grad()
def main():
    args, smooth_args, gptq_args, rotation_args = parse_args()
    # set seed
    setup_seed(args.seed)

    # process save_path
    if args.save_path:
        sub_dir_name = args.model_path.split("/")[-1]
        args.save_path = os.path.join(args.save_path, sub_dir_name)
        os.makedirs(args.save_path, exist_ok=True)

    # tokenizer path
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # load model
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype
    )

    # rotate model
    if args.rotation:
        model = fuse_layer_norms(model)
        model, Q = rotate_model(model, rotation_args, args)
        free_memory()

    # NOTE(HandH1998): No smoothing would give better results for now
    if args.smooth:
        # smooth model
        assert smooth_args.w_qconfig.group_size == gptq_args.groupsize
        model = quantize_model(model, smooth_args, args)
        scale_list = smooth(model, tokenizer, smooth_args, args)
        del model
        del tokenizer
        free_memory()

        # load model and apply smooth scales
        model, tokenizer = build_model_and_tokenizer(
            args.model_path, args.tokenizer_path, args.dtype
        )
        if args.rotation:
            # NOTE(HandH1998): smooth scale should work on the rotated model
            model = fuse_layer_norms(model)
            model, _ = rotate_model(model, rotation_args, args, Q)
            free_memory()

        model = export_smoothed_model(model, scale_list)

    # apply gptq
    # model = prepare_for_inference(model, args.device, args.dtype)
    # model = apply_gptq(model, gptq_args, args)

    # # quant_config
    # model.config.quantization_config = {
    #     "group_size": gptq_args.groupsize,
    #     "quant_method": "qqq",
    #     "wbits": gptq_args.wbits,
    # }

    # # save quantized model
    # state_dict = remove_empty_parameters(model)
    # model.save_pretrained(args.save_path, state_dict=state_dict)
    # tokenizer.save_pretrained(args.save_path)
    # logger.info(
    #     "Quant Finished! The quantized model is saved at {}.".format(args.save_path)
    # )
    # apply gptq
    # ─── calibration용 DataLoader: 시퀀스를 max_length 이하로 잘라냅니다 ───
    dataloader = get_wikitext2(
        tokenizer,
        block_size=args.max_length,
        batch_size=args.nsamples
    )

    # 2) inference 준비 & GPTQ 보정 실행 (dataloader 전달)
    model = prepare_for_inference(model, args.device, args.dtype)
    model = apply_gptq(model, gptq_args, args, dataloader)

    # quant_config
    model.config.quantization_config = {
        "group_size": gptq_args.groupsize,
        "quant_method": "qqq",
        "wbits": gptq_args.wbits,
    }

    # save quantized model
    state_dict = remove_empty_parameters(model)
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)
    logger.info(
        "Quant Finished! The quantized model is saved at {}.".format(args.save_path)
    )

if __name__ == "__main__":
    main()
