"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

d_model = 96
d_word = 300
d_char = 64


# Added char level modeling inside BiDAF
class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, drop_prob_char=0.05):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.wc_emb = layers.CharWordEmbedding(word_vectors=word_vectors,
                                               char_vectors=char_vectors,
                                               hidden_size=hidden_size,
                                               drop_prob=drop_prob,
                                               drop_prob_char=drop_prob_char)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        print("c_len:{}".format(c_len))
        print("q_len:{}".format(q_len))
        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


# Added by Reetika - hard-coded max limit of context and question, make it an argument later
class QANet(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, drop_prob_char=0.05):
        super(QANet, self).__init__()

        self.wc_emb = layers.CharWordEmbedding(word_vectors=word_vectors,
                                               char_vectors=char_vectors,
                                               hidden_size=hidden_size,
                                               drop_prob=drop_prob,
                                               drop_prob_char=0.05)

        self.c_conv = layers.DepthwiseSeparableConv(in_ch=d_word + d_char,
                                                    out_ch=d_model,
                                                    k=5)
        self.q_conv = layers.DepthwiseSeparableConv(in_ch=d_word + d_char,
                                                    out_ch=d_model,
                                                    k=5)
        self.c_emb_enc = layers.EncoderBlock(conv_num=4,
                                             ch_num=d_model,
                                             k=7,
                                             length=400)

        self.q_emb_enc = layers.EncoderBlock(conv_num=4,
                                             ch_num=d_model,
                                             k=7,
                                             length=50)

        self.att = layers.QANetAttention(d_model=d_model,
                                         drop_prob=drop_prob)

        self.CQ_conv = layers.DepthwiseSeparableConv(in_ch=4 * d_model,
                                                     out_ch=d_model,
                                                     k=5)

        self.mod = layers.EncoderBlock(conv_num=2,
                                       ch_num=d_model,
                                       k=5,
                                       length=400)

        self.mod_enc = nn.ModuleList([self.mod] * 7)

        self.out = layers.QANetOutput(d_model=d_model)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        # print("cw_idxs shape:{}".format(cw_idxs.shape))
        # print("cc_idxs shape:{}".format(cc_idxs.shape))
        # print("qw_idxs shape:{}".format(qw_idxs.shape))
        # print("qc_idxs shape:{}".format(qc_idxs.shape))
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)


        # Added by Reetika
        c = self.wc_emb(cw_idxs, cc_idxs)  # Create embeddings for context
        q = self.wc_emb(qw_idxs, qc_idxs)  # Create embeddings for question
        # print("c shape:{}".format(c.shape))
        c = self.c_conv(c)  # Depthwise conv changes from d_word+d_char to d_model
        q = self.q_conv(q)  # Depthwise conv changes from d_word+d_char to d_model
        # print("c after conv:{}".format(c.shape)) # (batch_size, d_model, clen)
        # print("q after conv:{}".format(q.shape))
        c_enc = self.c_emb_enc(c, c_mask)
        q_enc = self.q_emb_enc(q, q_mask)
        # End of changes
        # print("c after encoder:{}".format(c_enc.shape))

        att = self.att(c_enc, q_enc, c_mask, q_mask)  # (batch_size, 4 * d_model, clen)
        # print("att shape:{}".format(att.shape))
        att = self.CQ_conv(att)  # Compress to (batch_size, d_model, clen)
        # print("att shape after conv:{}".format(att.shape))

        x = att
        for i in range(3):
            for mod in self.mod_enc:
                x = mod(x, c_mask)
            if i == 0:
                x1 = x
            if i == 1:
                x2 = x
        x3 = x
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        out = self.out(torch.cat([x1, x2], dim=1), torch.cat([x1, x3], dim=1), c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
