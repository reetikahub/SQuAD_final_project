"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import math


def mask_logits(target, mask):
    return target * (mask) + (1 - mask) * (-1e30)


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        return self.out(x)


class Initialized_Conv1d_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')

    def forward(self, x):
        return F.relu(self.out(x))


def PosEncoder(x):
    channels = x.size()[1]  # (batch_size, channels, length)
    length = x.size()[2]
    freq = torch.Tensor([math.log(1e4) * (-i / ((channels // 2) - 1)) for i in range(channels // 2)])
    freq = torch.exp(freq).unsqueeze(dim=1)  # (channels/2, 1)
    # print(freqs)
    # print(freqs.shape)
    position = torch.arange(length).repeat(channels // 2, 1).to(torch.float)  # (channels/2, length)
    # print(position.shape)
    position = torch.mul(position, freq)  # (channels/2, length)
    # print(position)
    signal = torch.cat([torch.sin(position), torch.cos(position)], dim=0)  # (channels, length)
    signal = signal.view(1, channels, length)
    return x + signal.cuda()  # (batch_size, channels, length)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.key = Initialized_Conv1d(in_channels=d_model, out_channels=d_model , kernel_size=1, bias=False)
        self.query = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=False)
        self.value = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask, bias=False): # (batch_size, channels, length)
        B, C, T = x.size()
        K = self.key(x).transpose(1,2) #(B, T, C)
        # print(K.shape)
        K = K.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        # print(K.shape)
        Q = self.query(x).transpose(1,2)
        Q = Q.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        V = self.value(x).transpose(1,2)
        V = V.view(B, T, self.num_head, C // self.num_head).transpose(1, 2) # (B, nh, T, hs)
        # print(Q.shape)
        # print(K.shape)
        # print(V.shape)
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
        if bias:
            att += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(att.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            att = mask_logits(att, mask)
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ V
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return y.transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, conv_num, d_model, num_head, k, drop_prob=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(d_model, d_model, k) for _ in range(conv_num)])
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(conv_num)])
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, num_head, drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)
        # self.fc1 = nn.Linear(d_model, d_model)
        self.fc_1 = Initialized_Conv1d_Relu(d_model, d_model, bias=True)
        self.fc_2 = Initialized_Conv1d(d_model, d_model, bias=True)
        self.drop_prob = drop_prob
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        x = PosEncoder(x)
        res = x
        for i, conv in enumerate(self.convs):
            x = self.layer_norms1[i](x.transpose(1, 2)).transpose(1,
                                                                  2)  # (batch_size, seq_len, channels) We're taking layer norm along the channels dimension
            if i % 2 == 0:
                x = F.dropout(x, p=self.drop_prob, training=self.training)
            x = conv(x)
            x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
            l += 1
            res = x
        x = self.layer_norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.attention(x, mask)
        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        l += 1
        res = x
        x = self.layer_norm3(x.transpose(1, 2)).transpose(1, 2)
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        return x

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
                print(pred)
                print(dropout)
                print("___________________")
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CharWordEmbedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, d_word, d_char, channels, drop_prob=0.1, drop_prob_char=0.05):
        super().__init__()
        self.drop_prob = drop_prob
        self.drop_prob_char = drop_prob_char
        self.conv2d = nn.Conv2d(d_char, channels, kernel_size=(1, 5), bias=True)  # (d_char, d_char), earlier I did k=5
        self.conv1d = Initialized_Conv1d(d_word + channels, channels, bias=False)  # (d_char+d_word)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        # self.proj = nn.Linear(word_vectors.size(1)+d_char, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, channels)

    def forward(self, wd_emb, ch_emb):
        # print(ch_emb.shape)
        wd_emb = F.dropout(wd_emb, self.drop_prob, self.training)  # (batch_size, seq_len, embed_size)
        ch_emb = ch_emb.permute(0, 3, 1, 2)  # (batch_size, embed_size, seq_len, words)
        ch_emb = F.dropout(ch_emb, self.drop_prob_char, self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.transpose(1, 2)  # (batch_size, seq_len, embed_size)
        emb = torch.cat([ch_emb, wd_emb], dim=2)  # (batch_size, seq_len, embed_size_word+channels)
        # emb = self.proj(emb)  # (batch_size, seq_len, embed_size_word+embed_size_char)
        # Modification
        emb_c = self.conv1d(emb.transpose(1,
                                          2))  # (batch_size, channels, len) because next is conv for which #ch should be 2nd argument
        # print(emb_c.shape)
        emb_c = self.hwy(emb_c.transpose(1, 2))  # (batch_size, len, channels)
        return emb_c.transpose(1, 2)  # (batch_size, channels, len)


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        # print(emb.shape)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)
        return emb  # (batch_size, seq_len, hidden_size)


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x
        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class QANetAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, d_model, drop_prob=0.1):
        super(QANetAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(d_model, 1))
        self.q_weight = nn.Parameter(torch.zeros(d_model, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, d_model))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        c = c.transpose(1, 2)
        q = q.transpose(1, 2)
        # print("c shape:{}".format(c.shape))
        # print("q shape:{}".format(q.shape))

        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        # print("c_length:{}".format(c_len))
        # print("q_length:{}".format(q_len))
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        # print("s:{}".format(s.shape))
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        # print("output of attention layer{}".format(x.shape))
        return x.transpose(1, 2)

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        # print("c shape:{}".format(c.shape))
        # print("q shape:{}".format(q.shape))

        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        # print("c_length:{}".format(c_len))
        # print("q_length:{}".format(q_len))
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        # print("s:{}".format(s.shape))
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)
        # print(logits_1.shape)
        # # print(logits_1.squeeze().shape)
        # print((logits_1.squeeze())[1, 45:60])
        # print((logits_1.squeeze())[1, -10:-1])

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)
        # print("log of prob")
        # print(log_p1[1, 45:60])
        # print(log_p2[1, -10:-1])
        return log_p1, log_p2


class QANetOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, d_model):
        super().__init__()
        self.w1 = Initialized_Conv1d(d_model * 2, 1)
        self.w2 = Initialized_Conv1d(d_model * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        y1 = self.w1(X1)
        y2 = self.w2(X2)
        # print(logits_1.squeeze()[1, 45:60])
        # print(logits_1.squeeze()[1, -10:-1])
        log_p1 = masked_softmax(y1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(y2.squeeze(), mask, log_softmax=True)
        # print(log_p1[1, 45:60])
        # print(log_p2[1, -10:-1])
        return log_p1, log_p2


