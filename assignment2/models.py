import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_vocab_size, embedding_dim, hidden_dim, dropout=0.3):
        super(BiLSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tag_vocab_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True).to(device)
        self.tag_projection_layer = nn.Linear(hidden_dim, self.tagset_size).to(device)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def compute_lstm_emission_features(self, sentence):
        hidden = self.init_hidden()
        embeds = self.dropout(self.word_embeds(sentence))
        bilstm_out, hidden = self.bilstm(embeds, hidden)
        bilstm_out = self.dropout(bilstm_out)
        bilstm_out = bilstm_out
        bilstm_feats = self.tag_projection_layer(bilstm_out)
        return bilstm_feats

    def forward(self, sentence):
        bilstm_feats = self.compute_lstm_emission_features(sentence)
        return bilstm_feats.max(-1)[0].sum(), bilstm_feats.argmax(-1)

    def loss(self, sentence, tags):
        bilstm_feats = self.compute_lstm_emission_features(sentence)
        # transform predictions to (n_examples, n_classes) and ground truth to (n_examples)
        return F.cross_entropy(
            bilstm_feats.view(-1, self.tagset_size),
            tags.view(-1),
            reduction='sum'
        )


class BiLSTMCRFTagger(nn.Module):
    def __init__(self, vocab_size, tag_vocab_size, embedding_dim, hidden_dim, dropout=0.3, bigram=False):
        super(BiLSTMCRFTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tag_vocab_size + 1  # add one sp tag: start
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True).to(device)

        self.tag_projection_layer = nn.Linear(hidden_dim, self.tagset_size).to(device)

        self.bigram = bigram
        if bigram:
            self.transition = nn.Linear(hidden_dim, self.tagset_size * self.tagset_size)
        else:
            self.transition = nn.Parameter(torch.zeros(size=(self.tagset_size, self.tagset_size))).to(device)

        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def compute_lstm_emission_features(self, sentence):
        hidden = self.init_hidden()
        embeds = self.dropout(self.word_embeds(sentence))
        bilstm_out, hidden = self.bilstm(embeds, hidden)
        bilstm_out = self.dropout(bilstm_out)
        bilstm_out = bilstm_out
        bilstm_feats = self.tag_projection_layer(bilstm_out)
        return bilstm_feats, bilstm_out

    def get_energy(self, proj_feats, raw_feats):
        if not self.bigram:
            energy = self.transition.unsqueeze(0) + proj_feats.unsqueeze(1)
            # length x tagset_size(i-1) x tagset_size(i)
        else:
            energy = self.transition(raw_feats).view(-1, self.tagset_size, self.tagset_size) + proj_feats.unsqueeze(1)
            # length x tagset_size(i-1) x tagset_size(i)
        return energy

    def forward(self, sentence):
        # length x feature dim
        bilstm_feats, bilstm_out = self.compute_lstm_emission_features(sentence)
        bilstm_feats = bilstm_feats.squeeze(0)
        length, _ = bilstm_feats.size()

        energy = self.get_energy(bilstm_feats, bilstm_out)

        energy = energy[:, :-1, :-1]
        label_num = self.tagset_size - 1

        score_matrix = sentence.new_zeros(size=(length, label_num), dtype=torch.float)
        pointer = sentence.new_zeros(size=(length, label_num), dtype=torch.long)

        for t in range(length):
            cur_energy = energy[t]

            if t == 0:
                score_matrix[t] = cur_energy[-1:]
                pointer[t] = -1
            else:
                score_matrix[t], pointer[t] = torch.max(cur_energy + score_matrix[t-1].unsqueeze(1), dim=0)

        last_score, last_pointer = torch.max(score_matrix[-1], dim=0)
        pred_label = sentence.new_zeros(size=(length,), dtype=torch.long)
        pred_label[-1] = last_pointer

        for t in reversed(range(1, length)):
            last_pointer = pointer[t, last_pointer]
            pred_label[t - 1] = last_pointer

        return last_score.unsqueeze(0), pred_label.unsqueeze(0)

    def loss(self, sentence, tags):
        tags = tags.squeeze(0)

        bilstm_feats, bilstm_out = self.compute_lstm_emission_features(sentence)
        bilstm_feats = bilstm_feats.squeeze(0)
        length, _ = bilstm_feats.size()

        energy = self.get_energy(bilstm_feats, bilstm_out)

        # energy = self.transition.unsqueeze(0) + bilstm_feats.unsqueeze(1)  # length x tagset_size(i-1) x tagset_size(i)

        target_energy = sentence.new_zeros(size=(1,), dtype=torch.float)
        summary = None

        for t in range(length):
            cur_energy = energy[t]

            if t == 0:
                summary = cur_energy[-1, :]  # from start to the first token
                target_energy += cur_energy[-1, tags[t]]
            else:
                summary = torch.logsumexp(summary.unsqueeze(1) + cur_energy, dim=0)
                target_energy += cur_energy[tags[t-1], tags[t]]

        return torch.logsumexp(summary, dim=0) - target_energy
