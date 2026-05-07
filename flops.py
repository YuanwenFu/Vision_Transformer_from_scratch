import torch
from fvcore.nn import FlopCountAnalysis

from vit_model import Attention


def main():
    """
    此函数通过实验验证Self-Attention和Multi-Head Attention的计算量
    """

    # Self-Attention
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = torch.nn.Identity()  # remove Wo

    # Multi-Head Attention
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    flops1 = FlopCountAnalysis(a1, t)
    print("Self-Attention FLOPs:", flops1.total())

    flops2 = FlopCountAnalysis(a2, t)
    print("Multi-Head Attention FLOPs:", flops2.total())

    #多头注意力设计不在于减少计算量，而在于让不同的头并行学习不同的注意力模式（局部/全局，语义/位置等），提升表达能力。


if __name__ == '__main__':
    main()

