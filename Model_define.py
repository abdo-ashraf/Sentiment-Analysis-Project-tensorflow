import torch
from torch import nn

class Sentiment_LSTM(nn.Module):
    def __init__(self, embedding_matrix:torch.Tensor, lstm_hidden_size, lstm_layers, linear_hidden_size):
        super(Sentiment_LSTM, self).__init__()

        self.input_size = embedding_matrix.size(-1)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.linear_hidden_size = linear_hidden_size

        self.embedding_matrix = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size,
                          num_layers=self.lstm_layers, batch_first=True)

        # We will use only the last hidden state of the last layer for the prediction
        self.fc = nn.Sequential(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size),
                                nn.ReLU(), nn.Dropout(0.3))

        self.classifier = nn.Linear(self.linear_hidden_size, 1)

    def forward(self, x, device):
        h_0 = torch.zeros((self.lstm_layers, x.size(0), self.lstm_hidden_size)).to(device)
        c_0 = torch.zeros((self.lstm_layers, x.size(0), self.lstm_hidden_size)).to(device)

        embds = self.embedding_matrix(x)
        all_outputs, (h_final, c_final) = self.lstm(embds, (h_0, c_0))
        h_final_final_layer = h_final[-1,:,:]

        fc_out = self.fc(h_final_final_layer)

        output = self.classifier(h_final_final_layer)
        return nn.functional.sigmoid(output)