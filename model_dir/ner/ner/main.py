import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import os
import argparse
import logging

from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import trange, tqdm
from collections import OrderedDict
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)

class Net(nn.Module):
    def __init__(self, config, bert_state_dict, hp):
        super().__init__()
        
        self.tagset_size = len(hp.VOCAB)
        self.tag_to_ix = hp.tag2idx
        
        self.START_TAG = hp.START_TAG
        self.STOP_TAG = hp.STOP_TAG
        
        self.bert = BertModel(config)
        if bert_state_dict is not None:
            self.bert.load_state_dict(bert_state_dict)
            
        self.bert.eval()
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, self.tagset_size)
        self.device = hp.device

    def forward(self, x, y, seqlens):
        '''
        x: (N, T). int64
        y: (N, T). int64
        seqlens: (N,)
        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        maxlen = max(seqlens)
        m = [
            [1] * length + [0] * (maxlen - length)
            for length in seqlens
        ]
        m = torch.tensor(m, dtype=torch.long)
        m = m.to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.bert(x, attention_mask=m)
            enc = encoded_layers[-1]

        enc = pack_padded_sequence(enc, seqlens, batch_first=True, enforce_sorted=False)
        enc, _ = self.rnn(enc)
        enc, _ = pad_packed_sequence(enc, batch_first=True)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

class HParams:
    def __init__(self, model_dir, vocab_type, batch_size, device):
        
        self.VOCAB_FILE = os.path.join(model_dir, 'vocab.txt')
        self.BERT_CONFIG_FILE = os.path.join(model_dir, 'bert_config.json')
        self.BERT_WEIGHTS = os.path.join(model_dir, 'pytorch_weight')
        
        self.START_TAG = '[CLS]'
        self.STOP_TAG = '[SEP]'
        
        self.VOCAB_DICT = {
            'gvdc': (
                '<PAD>', '[CLS]', '[SEP]',
                'O',
                'B-gene', 'I-gene',
                'B-mutation', 'I-mutation',
                'B-disease', 'I-disease',
                'B-drug', 'I-drug',
            ),
        }
        self.VOCAB = self.VOCAB_DICT[vocab_type]
        self.tag2idx = {v:k for k,v in enumerate(self.VOCAB)}
        self.idx2tag = {k:v for k,v in enumerate(self.VOCAB)}

        self.batch_size = int(batch_size)
        # self.lr = 0.0001
        # self.n_epochs = 30 

        self.tokenizer = BertTokenizer(vocab_file=self.VOCAB_FILE, do_lower_case=False)
        self.device = device if torch.cuda.is_available() else 'cpu'

class NerDataset(data.Dataset):
    def __init__(self, path, hp):
        self.hp = hp
        instances = open(path).read().strip().split('\n\n')
        sents = []
        tags_li = []
        for entry in instances:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)


    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.hp.tokenizer.convert_tokens_to_ids(tokens)
            if not xx:
                xx = self.hp.tokenizer.convert_tokens_to_ids(["[PAD]"])

            is_head = [1] + [0]*(len(xx) - 1)

            t = [t] + ["<PAD>"] * (len(xx) - 1)  # <PAD>: no decision
            yy = [self.hp.tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen

def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens

def main(args):
    os.makedirs(args.target_dir, exist_ok=True)
    
    hp = HParams(args.model_dir, 'gvdc', args.batch_size, args.device)
    test_dataset = NerDataset(args.source_file, hp)
    test_iter = data.DataLoader(dataset=test_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)
    config = BertConfig(vocab_size_or_config_json_file=hp.BERT_CONFIG_FILE)

    tmp_d = torch.load(hp.BERT_WEIGHTS, map_location='cpu')
    state_dict = OrderedDict()
    for i in list(tmp_d.keys())[:199]:
        x = i
        if i.find('bert') > -1:
            x = '.'.join(i.split('.')[1:])
        state_dict[x] = tmp_d[i]

    model = Net(config = config, bert_state_dict = state_dict, hp = hp)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.pt'), map_location=torch.device(hp.device)))

    model.to(hp.device)

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    for datum in tqdm(test_iter, desc='test..'):
        words, x, is_heads, tags, y, seqlens = datum
        x = x.to(hp.device)

        _, _, y_hat = model(x, y, seqlens)  # y_hat: (N, T)

        Words.extend(words)
        Is_heads.extend(is_heads)
        Tags.extend(tags)
        Y.extend(y.numpy().tolist())
        Y_hat.extend(y_hat.cpu().numpy().tolist())

    with open(os.path.join(args.target_dir, 'target.tsv'), 'w') as f:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [hp.idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                f.write(f"{w}\t{t}\t{p}\n")
            f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="./source.tsv")
    parser.add_argument("--target_dir", type=str, default="./target_dir")
    parser.add_argument("--model_dir", type=str, default="./model_dir")
    parser.add_argument("--device", type=str, default="cuda:0")    
    parser.add_argument("--batch_size", type=int, default=128)    
    args = parser.parse_args()

    main(args)
