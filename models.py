"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CharWordEmbedding, Encoder, QANetOutput, Initialized_Conv1d, QANetAttention
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

        # self.wc_emb = layers.CharWordEmbedding(word_vectors=word_vectors,
        #                                          char_vectors=char_vectors,
        #                                        hidden_size=hidden_size,
        #                                          dropout_w=drop_prob)

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
        # print("c_mask:{}".format(c_mask))
        # print("c_len:{}".format(c_len))
        # print("q_len:{}".format(q_len))

        # c_emb = self.wc_emb(cw_idxs, cc_idxs)  # Create embeddings for context
        # q_emb = self.wc_emb(qw_idxs, qc_idxs)

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
    def __init__(self, word_vec, char_vec, c_max_len, q_max_len, d_model, drop_prob=0.1, num_head=8):  # !!! notice: set it to be a config parameter later.
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(char_vec)
        self.word_emb = nn.Embedding.from_pretrained(word_vec)
        print(word_vec.shape[1])
        print(char_vec.shape[1])
        self.emb = CharWordEmbedding(d_word=word_vec.shape[1], d_char=char_vec.shape[1], channels=d_model)
        self.num_head = num_head
        self.emb_enc = Encoder(conv_num=4, d_model=d_model, num_head=num_head, k=7, drop_prob=drop_prob)
        self.cq_att = QANetAttention(d_model=d_model, drop_prob=drop_prob)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([Encoder(conv_num=2, d_model=d_model, num_head=num_head, k=5, drop_prob=drop_prob) for _ in range(7)])
        self.out = QANetOutput(d_model)
        self.Lc = c_max_len
        self.Lq = q_max_len
        self.dropout = drop_prob

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        # print("maskC:{}".format(maskC))
        # print("maskC1:{}".format(maskC1))
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        C, Q = self.emb(Cw, Cc), self.emb(Qw, Qc) #Switch it around if using old function
        Ce = self.emb_enc(C, maskC, 1, 1) # (batch_size, seq_len, hidden_size)
        Qe = self.emb_enc(Q, maskQ, 1, 1) # (batch_size, seq_len, hidden_size)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=self.dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, 7)
        M3 = M0
        out = self.out(M1, M2, M3, maskC)
        return out

