from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':
    from thop import profile, clever_format
    # from fvcore.nn import FlopCountAnalysis
    # from ptflops import get_model_complexity_info

    # tqmsa = TQMSA(dim=384,
    #         num_heads=6,
    #         qkv_bias=False,
    #         qk_norm=False,
    #         attn_drop=0.,
    #         proj_drop=0.,
    #         norm_layer=nn.LayerNorm,
    #         tq_type='ttq',
    #         tq_level = [3,3,3,3],
    #         dic_n=1000, dic_dim=4, index=0,tq_Tmax = 10, tq_Tinit=-1)
    # msa = MSA(dim=384,
    #         num_heads=6,
    #         qkv_bias=False,
    #         qk_norm=False,
    #         attn_drop=0.,
    #         proj_drop=0.,
    #         )
    # model=tqmsa
    # model=msa
    # model.reparameterize()

    # input_shape_3=(1,197,384)




    model = nn.LayerNorm(384)
    # model = nn.Linear(384,4)
    # # model.reparameterize()
    model = model.cuda().eval()
    param_count = sum([m.numel() for m in model.parameters()])
    # param_count = format_param_count(param_count)
    input=torch.randn(1,197,384).cuda()
    # # output=model(input)
    # # if isinstance(output, tuple):
    # #     output = output[0]
    flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops} ,  params: {param_count}")

    # softmaxqkv = cal_qkvMatDot_FLOPs(batch=1,head_num = 6, seq_len = 145,dim = 384,block_num=12,istq=False)
    # print(f"qkv FLOPs: {softmaxqkv} ")

    # input = torch.randn(input_shape_3).cuda()  # 输入张量形状需匹配模型
    # flops = FlopCountAnalysis(model, input)
    # print(f"FlopCountAnalysis FLOPs: {flops.total()} ")


    # macs, params = get_model_complexity_info(model, (197,384*3), as_strings=True)
    # print(f"ptflops FLOPs: {macs}")

    # from calflops import calculate_flops
    # flops, macs, _ = calculate_flops(
    # model=model,
    # input_shape=input_shape_3,
    # output_as_string=True
    # )
    # print(f"calflops FLOPs: {flops}")

