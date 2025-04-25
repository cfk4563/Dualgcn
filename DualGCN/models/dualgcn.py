'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import head_to_tree, tree_to_adj
from math import sqrt


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []

        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DualGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        outputs1, outputs2, adj_ag, adj_dep = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        logits = self.classifier(final_outputs)

        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag@adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()

        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2
        
        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
            penal = self.opt.beta * penal
        
        return logits, penal


class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        self.opt = opt
        self.embedding_matrix = embedding_matrix
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None        # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim, padding_idx=0) if opt.post_dim > 0 else None    # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj = inputs           # unpack inputs
        maxlen = max(l.data)
        mask = mask[:, :maxlen]
        if self.opt.parseadj:
            adj_dep = adj[:, :maxlen, :maxlen].float()
        else:
            def inputs_to_tree_reps(head, words, l):
                trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
                adj = [tree_to_adj(maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
                adj = np.concatenate(adj, axis=0)
                adj = torch.from_numpy(adj)
                return adj.cuda()
            adj_dep = inputs_to_tree_reps(head.data, tok.data, l.data)

        h1, h2, adj_ag = self.gcn(adj_dep, inputs)
        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                        # aspect words num
        mask = mask.unsqueeze(-1).repeat(1,1,self.opt.hidden_dim)     # mask for h
        outputs1 = (h1*mask).sum(dim=1) / asp_wn
        outputs2 = (h2*mask).sum(dim=1) / asp_wn
        
        return outputs1, outputs2, adj_ag, adj_dep


class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim+opt.post_dim+opt.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True, dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden

        # transformer ecoder
        transformer_hidden = opt.rnn_hidden * 2
        self.projection = nn.Linear(input_size, transformer_hidden)
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=opt.rnn_dropout,
                                      output_attention=False), transformer_hidden, opt.attention_heads),
                    transformer_hidden,
                    transformer_hidden * 4,
                    dropout=opt.rnn_dropout,
                    activation='gelu'
                ) for l in range(opt.rnn_layers)
            ],
            norm_layer=nn.LayerNorm(self.in_dim)
        )

        # drop out
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim*2)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.in_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def encode_with_transformer(self, transformer_inputs, seq_lens):
        transformer_inputs = nn.utils.rnn.pack_padded_sequence(transformer_inputs, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        transformer_inputs, _ = nn.utils.rnn.pad_packed_sequence(transformer_inputs, batch_first=True)
        transformer_inputs = self.projection(transformer_inputs)
        transformer_outputs, attn = self.transformer_encoder(transformer_inputs)
        return transformer_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs           # unpack inputs
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = max(l.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        self.rnn.flatten_parameters()
        # gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))

        gcn_inputs = self.rnn_drop(self.encode_with_transformer(embs, l))

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        outputs_dep = None
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = mask_ * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn