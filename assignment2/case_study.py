import conlleval
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import *
from utils import read_conll_sentence, prepare_dataset, train, evaluate
from models import BiLSTMTagger, BiLSTMCRFTagger


if __name__ == '__main__':
    # load a list of sentences, where each word in the list is a tuple containing the word and the label
    train_data = list(read_conll_sentence(TRAIN_DATA))
    train_word_counter = Counter([word for sent in train_data for word in sent[0]])
    train_label_counter = Counter([label for sent in train_data for label in sent[1]])
    word_vocab = Vocab(train_word_counter, specials=(UNK, PAD), min_freq=2)
    label_vocab = Vocab(train_label_counter, specials=(), min_freq=1)
    train_data = prepare_dataset(train_data, word_vocab, label_vocab)
    print('Train word vocab:', len(word_vocab), 'symbols.')
    print('Train label vocab:', len(label_vocab), f'symbols: {list(label_vocab.stoi.keys())}')
    valid_data = list(read_conll_sentence(VALID_DATA))
    valid_data = prepare_dataset(valid_data, word_vocab, label_vocab)
    print('Train data:', len(train_data), 'sentences.')
    print('Valid data:', len(valid_data))

    print(' '.join([word_vocab.itos[i.item()] for i in train_data[0][0]]))
    print(' '.join([label_vocab.itos[i.item()] for i in train_data[0][1]]))

    print(' '.join([word_vocab.itos[i.item()] for i in valid_data[1][0]]))
    print(' '.join([label_vocab.itos[i.item()] for i in valid_data[1][1]]))

    rnn_tagger = BiLSTMTagger(len(word_vocab), len(label_vocab), 128, 256)\
        .to(device)
    rnn_tagger.load_state_dict(torch.load('./BiLSTM.pt'))
    crf_tagger = BiLSTMCRFTagger(len(word_vocab), len(label_vocab), 128, 256)\
        .to(device)
    crf_tagger.load_state_dict(torch.load('./BiLSTM+CRF.pt'))

    results = {}

    print('BiLSTM'.center(20, '='))
    sents, true_tags, pred_tags, _ = evaluate(rnn_tagger, valid_data, word_vocab, label_vocab)
    results['BiLSTM'] = {
        'sentence': sents,
        'true': true_tags,
        'pred': pred_tags,
    }
    print('BiLSTM+CRF'.center(20, '='))
    sents, true_tags, pred_tags, _ = evaluate(crf_tagger, valid_data, word_vocab, label_vocab)
    results['BiLSTM+CRF'] = {
        'sentence': sents,
        'true': true_tags,
        'pred': pred_tags,
    }
    for sent, rnn_pred, crf_pred, true in zip(results['BiLSTM']['sentence'], results['BiLSTM']['pred'], results['BiLSTM+CRF']['pred'], results['BiLSTM']['true']):
        if rnn_pred != true and crf_pred == true:
            print(sent)
            print('RNN:', rnn_pred)
            print('CRF:', crf_pred)
            print('='*20)



