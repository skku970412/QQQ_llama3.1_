import torch
import argparse
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from QQQ.utils import build_model_and_tokenizer, get_model_architecture


def export_smoothed_llama(model, scale_list):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaDecoderLayer):
            # 1) attention layernorm + q/k/v
            attn_ln = module.input_layernorm
            q, k, v, o = (
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
                module.self_attn.o_proj,
            )

            # layernorm 스케일링
            attn_ln.weight.data /= scale_list[cnt].to(attn_ln.weight.data.device)
            # q, k, v 스케일링
            q.weight.data *= scale_list[cnt].to(q.weight.data.device)
            k.weight.data *= scale_list[cnt].to(k.weight.data.device)
            v.weight.data *= scale_list[cnt].to(v.weight.data.device)
            cnt += 1

            # 2) o_proj / v_proj 스무딩 (조건 분기 처리)
            scale = scale_list[cnt].to(o.weight.data.device)
            o.weight.data *= scale

            # v_proj: 헤드 단위 벡터인지 스칼라인지 분기
            scale_v = scale_list[cnt]
            head_dim = v.weight.data.shape[0]
            if scale_v.numel() == head_dim:
                # per-head 스케일 벡터
                v.weight.data /= scale_v.reshape(-1, 1).to(v.weight.data.device)
            else:
                # 스칼라 혹은 broadcasting 가능한 경우
                v.weight.data /= scale_v.to(v.weight.data.device)
            cnt += 1

            # 3) feed-forward layernorm + MLP
            ffn_ln = module.post_attention_layernorm
            gate   = module.mlp.gate_proj
            up     = module.mlp.up_proj
            down   = module.mlp.down_proj

            # ffn layernorm
            ffn_ln.weight.data /= scale_list[cnt].to(ffn_ln.weight.data.device)
            # gate, up 스케일링
            gate.weight.data *= scale_list[cnt].to(gate.weight.data.device)
            up.weight.data   *= scale_list[cnt].to(up.weight.data.device)
            cnt += 1

            # down, up 후처리
            down.weight.data *= scale_list[cnt].to(down.weight.data.device)
            # up.weight.data   /= scale_list[cnt].to(up.weight.data.device)
            s = scale_list[cnt].to(up.weight.data.device)
            if s.numel() == up.weight.data.shape[0]:
                up.weight.data /= s.view(-1, 1)
            else:
                up.weight.data /= s
            cnt += 1

    return model


def export_smoothed_qwen2(model, scale_list):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2DecoderLayer):
            # 1) attention layernorm + q/k/v
            attn_ln = module.input_layernorm
            q, k, v, o = (
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
                module.self_attn.o_proj,
            )

            attn_ln.weight.data /= scale_list[cnt].to(attn_ln.weight.data.device)
            q.weight.data     *= scale_list[cnt].to(q.weight.data.device)
            k.weight.data     *= scale_list[cnt].to(k.weight.data.device)
            v.weight.data     *= scale_list[cnt].to(v.weight.data.device)
            cnt += 1

            # 2) o_proj / v_proj smoothing (조건 분기 처리)
            scale = scale_list[cnt].to(o.weight.data.device)
            o.weight.data *= scale

            scale_v = scale_list[cnt]
            head_dim = v.weight.data.shape[0]
            if scale_v.numel() == head_dim:
                # per-head 스케일 벡터
                v.weight.data /= scale_v.reshape(-1, 1).to(v.weight.data.device)
                v.bias.data   /= scale_v.to(v.weight.data.device)
            else:
                # 스칼라 혹은 broadcasting 가능한 경우
                v.weight.data /= scale_v.to(v.weight.data.device)
                v.bias.data   /= scale_v.to(v.weight.data.device)
            cnt += 1

            # 3) feed-forward layernorm + MLP
            ffn_ln = module.post_attention_layernorm
            gate   = module.mlp.gate_proj
            up     = module.mlp.up_proj
            down   = module.mlp.down_proj

            ffn_ln.weight.data /= scale_list[cnt].to(ffn_ln.weight.data.device)
            gate.weight.data   *= scale_list[cnt].to(gate.weight.data.device)
            up.weight.data     *= scale_list[cnt].to(up.weight.data.device)
            cnt += 1

            down.weight.data *= scale_list[cnt].to(down.weight.data.device)
            up.weight.data   /= scale_list[cnt].to(up.weight.data.device)
            cnt += 1

    return model


def export_smoothed_model(model, scale_list):
    model_type = get_model_architecture(model.config)
    if model_type == "llama":
        model = export_smoothed_llama(model, scale_list)
    elif model_type == "qwen2":
        model = export_smoothed_qwen2(model, scale_list)
    else:
        raise NotImplementedError
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--scale_list", required=True)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    model, tokenizer = build_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.dtype, args.device
    )
    scale_list = torch.load(args.scale_list)
    model = export_smoothed_model(model, scale_list)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
