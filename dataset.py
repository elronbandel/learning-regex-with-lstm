from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch

def load_data(file):
    def prepare(line):
        string, tag = line.split()
        return [ord(char) for char in string], int(tag)
    return [prepare(line) for line in open(file)]


class RegexDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return list(map(torch.tensor, self.data[index]))

    def __len__(self):
        return len(self.data)

    @classmethod
    def from_file(cls, file):
        return RegexDataset(load_data(file))

def identity(x):
    return x

def collate_sequences(batch):
    return [[item[0] for item in batch], torch.stack([item[1] for item in batch])]

def loader(file, batch_size, workers=6):
    return DataLoader(RegexDataset.from_file(file), batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_sequences)


def test():
    train = loader('data/test', 30)
    x, y = next(iter(train))
    print(x)
    print(y)

if __name__ == "__main__":
    test()