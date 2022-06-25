import copy

import torch.nn as nn

from ai.transformer import layers as L


def make_model(
    src_vocab,
    target_vocab,
    N=6,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    d_model=512,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = L.MultiHeadedAttention(n_heads, d_model)
    ff = L.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = L.PositionalEncoding(d_model, dropout)

    encoder_layer = L.EncoderLayer(d_model, c(attn), c(ff), dropout)
    encoder = L.Encoder(encoder_layer, N)

    decoder_layer = L.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    decoder = L.Decoder(decoder_layer, N)

    model = L.EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        src_embed=nn.Sequential(L.Embeddings(d_model, src_vocab), c(position)),
        target_embed=nn.Sequential(L.Embeddings(d_model, target_vocab), c(position)),
        generator=L.Generator(d_model, target_vocab),
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    print("Number of model parameters:", model.number_of_parameters)
    return model
