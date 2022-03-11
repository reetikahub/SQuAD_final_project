"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CharWordEmbedding, CharEmbedding, WordEmbeddingFeatures, Encoder, QANetOutput, Initialized_Conv1d, QANetAttention


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

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, pos_size=51, pos_dim=16, ner_size=27, ner_dim=8):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.char_emb = nn.Embedding.from_pretrained(char_vectors)
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        
        self.ch_emb = CharEmbedding(d_char=char_vectors.shape[1], channels=hidden_size, drop_prob_char=0.05)
        self.q_word_emb = WordEmbeddingFeatures(d_word=word_vectors.shape[1], d_pos=pos_dim, d_ner=ner_dim, channels=hidden_size, drop_prob = 0.1, add_feat = False)
        self.c_word_emb = WordEmbeddingFeatures(d_word=word_vectors.shape[1], d_pos=pos_dim, d_ner=ner_dim, channels=hidden_size, drop_prob = 0.1, add_feat = True)
        self.pos_emb = nn.Embedding(pos_size, pos_dim)
        self.ner_emb = nn.Embedding(ner_size, ner_dim)
	#self.wc_emb = layers.CharWordEmbedding(d_word=word_vectors.shape[1], d_char=char_vectors.shape[1], channels=hidden_size)
        #self.c_proj = Initialized_Conv1d(word_vectors.shape[1]+hidden_size, hidden_size, bias=False)  # (d_char+d_word)
        #self.q_proj = Initialized_Conv1d(word_vectors.shape[1]+hidden_size, hidden_size, bias=False)
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

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs, c_pos, c_ner, c_freq, c_em):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        Qw, Qc = self.word_emb(qw_idxs), self.char_emb(qc_idxs)
        Cw, Cc = self.word_emb(cw_idxs), self.char_emb(cc_idxs)

        Cp = self.pos_emb(c_pos)
        Cn = self.ner_emb(c_ner)
        #print(Cw.shape)
        #print(Cp.shape)
        Cc = self.ch_emb(Cc)
        c_emb = self.c_word_emb(Cw, Cc, Cp, Cn, c_freq, c_em)
        Qc = self.ch_emb(Qc)
        b, len, _ = Qc.shape
        #print(b,len)
        Cp_dummy = torch.zeros(b, len, 16)
        Cn_dummy = torch.zeros(b, len, 8)
        Cf_dummy = torch.zeros(b, len)
        q_emb = self.c_word_emb(Qw, Qc, Cp_dummy.cuda(), Cn_dummy.cuda(), Cf_dummy.cuda(), Cf_dummy.cuda()) #Switch it around if using old function
        
        #q_emb = self.q_proj(q_emb)
        #c_emb = self.c_proj(c_emb)
        #c_emb = self.wc_emb(Cw, Cc)  # Create embeddings for context
        #q_emb = self.wc_emb(Qw, Qc)

        # c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        # q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb.transpose(1, 2), c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb.transpose(1, 2), q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    def __init__(self, word_vec, char_vec, d_model, drop_prob=0.1, num_head=8):  # Everything come from train.py.
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(char_vec)
        self.word_emb = nn.Embedding.from_pretrained(word_vec)
        #print(word_vec.shape[1])
        #print(char_vec.shape[1])
        self.emb = CharWordEmbedding(d_word=word_vec.shape[1], d_char=char_vec.shape[1], channels=d_model)
        self.num_head = num_head
        self.emb_enc = Encoder(conv_num=4, d_model=d_model, num_head=num_head, k=7, drop_prob=drop_prob)
        self.cq_att = QANetAttention(d_model=d_model, drop_prob=drop_prob)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([Encoder(conv_num=2, d_model=d_model, num_head=num_head, k=5, drop_prob=drop_prob) for _ in range(5)])
        self.out = QANetOutput(d_model)
        self.dropout = drop_prob

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        #print("maskC:{}".format(maskC))
        # print("maskC1:{}".format(maskC1))
        Cw = self.word_emb(Cwid)
        Cc = self.char_emb(Ccid)
        #print(Cw.shape)
        #print(Cc.shape)
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

# Added by Reetika - hard-coded max limit of context and question, make it an argument later
class QANet_extra(nn.Module):
    def __init__(self, word_vec, char_vec, d_model, drop_prob=0.1, num_head=8, pos_size=51, pos_dim=16, ner_size=27, ner_dim=8):  # Everything come from train.py.
        super().__init__()
        self.char_emb = nn.Embedding.from_pretrained(char_vec)
        self.word_emb = nn.Embedding.from_pretrained(word_vec)
        self.pos_emb = nn.Embedding(pos_size, pos_dim)
        self.ner_emb = nn.Embedding(ner_size, ner_dim)
        #print(word_vec.shape[1])
        #print(char_vec.shape[1])
        self.ch_emb = CharEmbedding(d_char=char_vec.shape[1], channels=d_model, drop_prob_char=0.05)
        #self.q_word_emb = WordEmbeddingFeatures(d_word=word_vec.shape[1], d_pos=pos_dim, d_ner=ner_dim, channels=d_model, drop_prob = 0.1, add_feat = True)
        self.c_word_emb = WordEmbeddingFeatures(d_word=word_vec.shape[1], d_pos=pos_dim, d_ner=ner_dim, channels=d_model, drop_prob = 0.1, add_feat = True)

        self.num_head = num_head
        self.emb_enc = Encoder(conv_num=4, d_model=d_model, num_head=num_head, k=7, drop_prob=drop_prob)
        self.cq_att = QANetAttention(d_model=d_model, drop_prob=drop_prob)
        self.cq_resizer = Initialized_Conv1d(d_model * 4, d_model)
        self.model_enc_blks = nn.ModuleList([Encoder(conv_num=2, d_model=d_model, num_head=num_head, k=5, drop_prob=drop_prob) for _ in range(5)])
        self.out = QANetOutput(d_model)
        self.dropout = drop_prob

    def forward(self, Cwid, Ccid, Qwid, Qcid, c_pos, c_ner, c_freq, c_em):
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        #print("maskC:{}".format(maskC))
        # print("maskC1:{}".format(maskC1))
        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Cp = self.pos_emb(c_pos)
        Cn = self.ner_emb(c_ner)
        #print(Cw.shape)
        #print(Cp.shape)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)
        
        Cc = self.ch_emb(Cc)
        C = self.c_word_emb(Cw, Cc, Cp, Cn, c_freq, c_em)
        Cp_dummy = torch.zeros(b, len, 16)
        Cn_dummy = torch.zeros(b, len, 8)
        Cf_dummy = torch.zeros(b, len)
        Qc = self.ch_emb(Qc)
        Q = self.c_word_emb(Qw, Qc, Cp_dummy, Cn_dummy, Cf_dummy, Cf_dummy) #Switch it around if using old function
        
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

