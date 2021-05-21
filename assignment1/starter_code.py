import argparse
import random
import time

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from queue import PriorityQueue

import tqdm
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        # need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        hidden_batch_size = hidden.shape[0]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def translate_sentence_with_beam_search(sentence, src_field, trg_field, model, device, beam_size, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    beam_wids = torch.zeros(size=(max_len, beam_size), dtype=torch.int).to(device)
    beam_logits = torch.zeros(size=(beam_size, 1), dtype=torch.float).to(device)
    beam_attentions = torch.zeros(size=(max_len, beam_size, len(src_indexes))).to(device)
    beam_back_pointer = torch.zeros(size=(max_len, beam_size), dtype=torch.int).to(device)

    beam_hidden = hidden.repeat(beam_size, 1)
    beam_wids[0][0] = trg_field.vocab.stoi[trg_field.init_token]
    beam_back_pointer[0][0] = 0

    encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)
    mask = mask.repeat(beam_size, 1)

    def batch_top_k(batch, k):
        bsz, dim = batch.shape
        flatten = batch.flatten()
        top_values, top_ids = torch.topk(flatten, k=k)

        batch_ids = torch.divide(top_ids, dim, rounding_mode="trunc")
        dim_ids = torch.remainder(top_ids, dim)

        return top_values, batch_ids, dim_ids

    for t in range(1, max_len):
        decoder_input = beam_wids[t - 1]
        decoder_hidden = beam_hidden
        with torch.no_grad():
            output, decoder_hidden, attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs, mask)
            beam_attentions[t] = attention
            logits = F.log_softmax(output, dim=-1)

        # logits: beam_size x vocab
        # decoder_hidden: beam_size x hidden size
        # attention: beam_size x src len

        logits += beam_logits

        top_logits, selected_beams, top_wids = batch_top_k(logits, k=beam_size)

        # top_logits: beam_size
        # selected_beams: beam_size
        # top_wids: beam_size

        beam_back_pointer[t] = selected_beams

        beam_hidden = decoder_hidden[selected_beams]
        beam_wids[t] = top_wids
        beam_logits = top_logits.unsqueeze(1)

        if torch.all(top_wids == trg_field.vocab.stoi[trg_field.eos_token]):
            break

    predicted_wids = []
    predicted_attentions = []

    beam_pointer = torch.argmax(beam_logits).item()

    for t in range(max_len - 1, 0, -1):
        predicted_wids.append(beam_wids[t][beam_pointer].item())
        predicted_attentions.append(beam_attentions[t][beam_pointer].unsqueeze(0))

        beam_pointer = beam_back_pointer[t][beam_pointer]

    predicted_wids = predicted_wids[::-1]
    predicted_attentions = predicted_attentions[::-1]

    predicted_sequence = []
    for i in predicted_wids:
        predicted_sequence.append(trg_field.vocab.itos[i])
        if predicted_sequence[-1] == trg_field.eos_token:
            break
    predicted_attentions = predicted_attentions[:len(predicted_sequence)]
    predicted_attentions = torch.cat(predicted_attentions, dim=0)

    return predicted_sequence, predicted_attentions


def translate_sentence_with_nucleus_sampling(sentence, src_field, trg_field, model, device, sample_prob, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention

        logits = F.softmax(output, dim=-1).squeeze()

        sorted_logits, sorted_indexes = torch.sort(logits, descending=True)

        accum_sorted_logits = torch.cumsum(sorted_logits, dim=-1)

        sample_mask = (accum_sorted_logits < sample_prob).int()

        sample_mask = torch.cat([sample_mask.new_ones((1,)), sample_mask], dim=-1)[:-1]

        masked_sorted_logits = sample_mask * sorted_logits
        masked_sorted_logits /= masked_sorted_logits.sum()

        sample = torch.multinomial(masked_sorted_logits, num_samples=1).item()

        pred_token = sorted_indexes[sample].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]


def display_attention(sentence, translation, attention, name='_'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    attention = attention.squeeze(1).cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)

    x_ticks = [''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>']
    y_ticks = [''] + translation

    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_yticklabels(y_ticks)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    plt.savefig(f'./outputs/{name}.png')
    plt.close()


def calculate_bleu(data, src_field, trg_field, model, device, translator, max_len=50):
    trgs = []
    pred_trgs = []

    for datum in tqdm.tqdm(data):
        src = vars(datum)['src']
        trg = vars(datum)['trg']

        kwargs = {'sentence': src,
                  'src_field': src_field,
                  'trg_field': trg_field,
                  'model': model,
                  'device': device,
                  'max_len': max_len}
        if args.method == 'beam':
            kwargs['beam_size'] = args.beam_size
        elif args.method == 'nucleus':
            kwargs['sample_prob'] = args.sample_prob

        pred_trg, _ = translator(**kwargs)

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--method', type=str, choices=['greedy', 'beam', 'nucleus'])
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--sample_prob', type=float, default=0.9)
    args = parser.parse_args()

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')


    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in spacy_de.tokenizer(text)]


    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]


    SRC = Field(
        #   tokenize = tokenize_de,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True,
        include_lengths=True
    )

    TRG = Field(
        # tokenize = tokenize_en,
        init_token='<sos>',
        eos_token='<eos>',
        lower=True
    )

    # NOTE: this line takes a long time to run on Colab so
    # instead we'll load the already tokenized json dataset
    # from Github
    #
    # train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
    #                                                     fields = (SRC, TRG))

    # fetch from Github repo
    # !wget
    # https://raw.githubusercontent.com/tberg12/cse291spr21/main/assignment1/train.json
    # !wget
    # https://raw.githubusercontent.com/tberg12/cse291spr21/main/assignment1/valid.json
    # !wget
    # https://raw.githubusercontent.com/tberg12/cse291spr21/main/assignment1/test.json

    # and load to same variables
    fields = {'src': ('src', SRC), 'trg': ('trg', TRG)}
    train_data, valid_data, test_data = TabularDataset.splits(
        path='.',
        train='train.json',
        validation='valid.json',
        test='test.json',
        format='json',
        fields=fields
    )

    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device)

    ## Training the Seq2Seq Model

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    if args.do_train:

        def init_weights(m):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)


        model.apply(init_weights)


        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)


        print(f'The model has {count_parameters(model):,} trainable parameters')

        optimizer = optim.Adam(model.parameters())

        N_EPOCHS = 15
        CLIP = 1

        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    if args.do_test:

        assert os.path.exists('tut4-model.pt')

        model.load_state_dict(torch.load('tut4-model.pt'))

        test_loss = evaluate(model, test_iterator, criterion)

        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        if args.method == 'greedy':
            title = 'greedy'
            translator = translate_sentence
            kwargs = {'sentence': None,
                      'src_field': SRC,
                      'trg_field': TRG,
                      'model': model,
                      'device': device}
        elif args.method == 'beam':
            title = f'beam{args.beam_size}'
            translator = translate_sentence_with_beam_search
            kwargs = {'sentence': None,
                      'src_field': SRC,
                      'trg_field': TRG,
                      'model': model,
                      'device': device,
                      'beam_size': args.beam_size}
        else:
            title = f'nucleus{args.sample_prob}'
            translator = translate_sentence_with_nucleus_sampling
            kwargs = {'sentence': None,
                      'src_field': SRC,
                      'trg_field': TRG,
                      'model': model,
                      'device': device,
                      'sample_prob': args.sample_prob}

        ## Inference

        example_idx = 12

        src = vars(train_data.examples[example_idx])['src']
        trg = vars(train_data.examples[example_idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        kwargs['sentence'] = src
        translation, attention = translator(**kwargs)

        print(f'predicted trg = {translation}')

        display_attention(src, translation, attention, name=f'{title}-test{example_idx}')

        example_idx = 14

        src = vars(valid_data.examples[example_idx])['src']
        trg = vars(valid_data.examples[example_idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        kwargs['sentence'] = src
        translation, attention = translator(**kwargs)

        print(f'predicted trg = {translation}')

        display_attention(src, translation, attention, name=f'{title}-test{example_idx}')

        example_idx = 18

        src = vars(test_data.examples[example_idx])['src']
        trg = vars(test_data.examples[example_idx])['trg']

        print(f'src = {src}')
        print(f'trg = {trg}')

        kwargs['sentence'] = src
        translation, attention = translator(**kwargs)

        print(f'predicted trg = {translation}')

        display_attention(src, translation, attention, name=f'{title}-test{example_idx}')

        ## BLEU

        bleu_score = calculate_bleu(test_data, SRC, TRG, model, device, translator=translator)

        print(f'BLEU score = {bleu_score * 100:.2f}')
