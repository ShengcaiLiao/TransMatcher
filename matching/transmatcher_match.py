import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Module
from torch.nn.modules.container import ModuleList
from torch import einsum


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len, d_model=512, dim_feedforward=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        score_embed = torch.randn(seq_len, seq_len)
        score_embed = score_embed + score_embed.t()
        self.score_embed = nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        self.fc1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
        self.bn2 = nn.BatchNorm1d(dim_feedforward)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(dim_feedforward, 1)
        self.bn3 = nn.BatchNorm1d(1)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        q, h, w, d = tgt.size()
        assert(h * w == self.seq_len and d == self.d_model)
        k, h, w, d = memory.size()
        assert(h * w == self.seq_len and d == self.d_model)

        tgt = tgt.view(q, -1, d)
        memory = memory.view(k, -1, d)
        query = self.fc1(tgt)
        key = self.fc1(memory)
        score = einsum('q t d, k s d -> q k s t', query, key) * self.score_embed.sigmoid()
        score = score.reshape(q * k, self.seq_len, self.seq_len)
        
        prob_score, index_in_gal = score.max(dim=1)
        gal_score, index_in_prob = score.max(dim=2)
        score = torch.cat((prob_score, gal_score), dim=-1)

        # score = torch.cat((score.max(dim=1)[0], score.max(dim=2)[0]), dim=-1)
        score = score.view(-1, 1, self.seq_len)
        score = self.bn1(score).view(-1, self.seq_len)

        score = self.fc2(score)
        score = self.bn2(score)
        score = self.relu(score)
        score = self.fc3(score)
        score = score.view(-1, 2).sum(dim=-1, keepdim=True)
        score = self.bn3(score)
        score = score.view(q, k)

        prob_score = prob_score.view(q, k, h, w)
        index_in_gal = index_in_gal.view(q, k, h, w)
        gal_score = gal_score.view(q, k, h, w)
        index_in_prob = index_in_prob.view(q, k, h, w)
        match_index = torch.stack((prob_score, index_in_gal, gal_score, index_in_prob), dim=2)

        return score, match_index


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        tgt = tgt.chunk(self.num_layers, dim=-1)
        memory = memory.chunk(self.num_layers, dim=-1)
        match_index = []
        for i, mod in enumerate(self.layers):
            if i == 0:
                score, index = mod(tgt[i], memory[i])
            else:
                score2, index = mod(tgt[i], memory[i])
                score = score + score2
            match_index.append(index)

        match_index = torch.stack(match_index, dim=2)

        if self.norm is not None:
            q, k = score.size()
            score = score.view(-1, 1)
            score = self.norm(score)
            score = score.view(q, k)

        return score, match_index


class TransMatcher(nn.Module):

    def __init__(self, seq_len, d_model=512, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.decoder_layer = TransformerDecoderLayer(seq_len, d_model, dim_feedforward)
        decoder_norm = nn.BatchNorm1d(1)
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers, decoder_norm)
        self.memory = None
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def make_kernel(self, features):
        self.memory = features

    def forward(self, features):
        score, match_index = self.decoder(self.memory, features)
        return score, match_index


if __name__ == "__main__":
    import time
    model = TransMatcher(24*8, 512, 3).eval()
    gallery = torch.rand((32, 24, 8, 512*3))
    probe = torch.rand((16, 24, 8, 512*3))

    start = time.time()
    model.make_kernel(gallery)
    out, match_index = model(probe)
    print(out.size())
    end = time.time()
    print('Time: %.3f seconds.' % (end - start))

    start = time.time()
    model.make_kernel(probe)
    out2, match_index2 = model(gallery)
    print(out2.size())
    end = time.time()
    print('Time: %.3f seconds.' % (end - start))
    out2 = out2.t()
    print((out2 == out).all())
    print((out2 - out).abs().mean())
    print(out[:4, :4])
    print(out2[:4, :4])
