import torch
from tab_transformer_pytorch import TSTransformer

def test_forward_without_errors():

    model = TSTransformer(
        categories = (10, 5, 6, 5, 8),      # tuple containing the number of unique values within each category
        num_continuous = 10,                # number of continuous values
        dim = 6,                           # dimension, paper set at 32
        dim_out = 1,                        # binary prediction, but could be anything
        depth = 6,                          # depth, paper recommended 6
        heads = 8,                          # heads, paper recommends 8
        attn_dropout = 0.1,                 # post-attention dropout
        ff_dropout = 0.1                    # feed forward dropout
    )

    x_categ = torch.randint(0, 5, (1, 10, 5))     # category values, from 0 - max number of categories, in the order as passed into the constructor above
    x_numer = torch.randn(1, 10, 10)              # numerical value

    pred = model(x_categ, x_numer)
    
    assert pred.shape == torch.Size([1, 1])