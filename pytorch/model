import torch
import torch.nn as nn
from torch.nn import LSTM, Sequential, LeakyReLU, Dropout, Linear


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(input_size, hidden_size, num_layers, dropout=0.4, batch_first=True)

        
        self.fc = Sequential(
            Linear(hidden_size, 128),
            nn.ReLU(),
            Linear(128, output_size)
        )
        
        

    def forward(self, x):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
