import math


def extend_embeddings(src):
    return src[: src.shape[0], :, :].repeat(math.ceil(256 / src.shape[0]), 1, 1)[
        :256, :, :
    ]


import math


def extend_attention_masks(src):
    return src[: src.shape[0], :].repeat(math.ceil(256 / src.shape[0]), 1)[:256, :]
