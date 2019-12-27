from torch.nn import Module, Embedding, LSTM, Sequential, Linear, ReLU,Tanh, LogSoftmax
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch


class SequenceTagger(Module):
    def __init__(self, vocab_size, embed_dim, lstm_hidden, mlp_hidden, output_dim, device=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, lstm_hidden, batch_first=True)
        self.mlp = Sequential(Linear(lstm_hidden, output_dim), LogSoftmax(dim=1))
        #self.mlp = Sequential(Linear(lstm_hidden, mlp_hidden), Tanh(), Linear(mlp_hidden, output_dim), LogSoftmax(dim=1))
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, batch):
        lens = list(map(len, batch))
        padded = pad_sequence(batch, batch_first=True).to(self.device)
        embeded = self.embedding(padded)
        packed = pack_padded_sequence(embeded, lens, batch_first=True, enforce_sorted=False)
        output, (ht, ct) = self.lstm(packed)
        output, out_lens = pad_packed_sequence(output, batch_first=True)
        last_seq = output[torch.arange(output.shape[0]), out_lens - 1]
        return self.mlp(last_seq)

