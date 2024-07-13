from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size, num_layers, type_rnn):
        super().__init__()
        self.type_rnn = type_rnn
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx = 0)
        if type_rnn.lower() == 'rnn':
            self.rnn = nn.RNN(input_size = embed_dim, hidden_size = hidden_size, num_layers = num_layers)
        elif type_rnn.lower() == 'gru':
            self.rnn = nn.GRU(input_size = embed_dim, hidden_size = hidden_size, num_layers = num_layers)
        elif type_rnn.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, num_layers = num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted = False, batch_first = True)

        if self.type_rnn != 'lstm':
            _, hidden = self.rnn(out)
            out = hidden[-1]
        else:
            _, (hidden, cell) = self.rnn(out)
            out = hidden[-1, :, :]

        out = self.fc(out)
        return out

class TextDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, index: int):
        return self.data[index], self.targets[index]



