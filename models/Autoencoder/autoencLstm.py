import torch
import torch.nn as nn
import torchaudio
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # Encoding
        _, (hidden, _) = self.encoder(x)

        # Repeating the last hidden state to create a sequence to decode
        seq_len = x.size(1)
        decode_input = hidden.repeat(1, seq_len, 1)

        # Decoding
        decoded, _ = self.decoder(decode_input)

        return decoded

input_size = 160  # Might be equal to the sample rate, assuming 1 sec of audio.
hidden_size = 128
num_layers = 1
learning_rate = 0.001
num_epochs = 100
batch_size = 64


# Assume `audio_data` is a pre-processed and normalized tensor
# audio_data: torch.Tensor [num_samples, sequence_length, num_features]

train_loader = DataLoader(TensorDataset(audio_data), batch_size=64, shuffle=True)


# Initialize model, loss, and optimizer
model = LSTMAutoencoder(input_size=160, hidden_size=128, num_layers=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch[0]

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Logging / Validation Logic
    print(f'Epoch {epoch}, Loss: {loss.item()}')


