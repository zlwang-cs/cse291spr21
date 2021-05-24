import torch

torch.manual_seed(291)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA = 'train.data.quad'
VALID_DATA = 'dev.data.quad'
UNK = '<unk>'
PAD = '<pad>'
START_TAG = "<start>"  # you can add this explicitly or use it implicitly in your CRF layer
STOP_TAG = "<stop>"    # you can add this explicitly or use it implicitly in your CRF layer

MODEL_TYPE = 'BiLSTM+CRF'
assert MODEL_TYPE in ['BiLSTM', 'BiLSTM+CRF']

