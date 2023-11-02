import torch
import torch.nn as nn
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
import librosa

MU = 255

def mu_law_encode(y, mu=MU):
    """
    Encode floating values ranging from -1.0 ~ 1.0 to integer 0 ~ mu.
    """
    mu_law_compressed = librosa.mu_compress(y, mu=mu, quantize=True) + ((mu+1) // 2)
    mu_law_compressed = mu_law_compressed.astype('int16')
    return mu_law_compressed


def mu_law_decode(y, mu=MU):
    """
    Decode integer values between 0 ~ mu to values bewteen -1.0 ~ 1.0.
    """
    mu_law_expanded = librosa.mu_expand(y - ((mu+1) // 2), mu=mu, quantize=True)
    return mu_law_expanded

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



class SingerDataset(Dataset):
    def __init__(self, data_path,sample_rate=16000):
        self.sample_rate = sample_rate
        # data_path is a directory that in a nested structure of it there is .wav files. create a list of the paths of the .wav files
        self.data_path_list = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    duration = int(librosa.get_duration(filename=audio_path))
                    for i in range(duration):
                        self.data_path_list.append([audio_path,i])
        print(len(self.data_path_list))




    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        audiopath, start_time = self.data_path_list[idx]
        audio, sr = librosa.load(audiopath, sr=self.sample_rate, offset=start_time, duration=1)
        audioMU = mu_law_encode(audio)
        return audioMU, sr




input_size = 16000  # Might be equal to the sample rate, assuming 1 sec of audio.
hidden_size = 1280
num_layers = 1
learning_rate = 0.001
num_epochs = 1
batch_size = 64


# set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Assume `audio_data` is a pre-processed and normalized tensor
# audio_data: torch.Tensor [num_samples, sequence_length, num_features]

# TODO : These wave file are not preprocessed and normalized. We need to do that first.

data_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer\female1'
train_loader = DataLoader(SingerDataset(data_path), batch_size=16, shuffle=True)
print('Data loaded')

# Initialize model, loss, and optimizer
model = LSTMAutoencoder(input_size=16, hidden_size=12, num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
total_loss = []
# Training loop
print('Training started')
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch[0].to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        print(f'Batch {batch}, Loss: {total_loss[-1]}')

    # Logging / Validation Logic
    print(f'Epoch {epoch}, Loss: {total_loss[-1]}')

# TODO: Save the model and graph the loss

# Assume `new_audio_data` is your new data to be encoded/decoded
# new_audio_data: torch.Tensor [num_samples, sequence_length, num_features]

# Ensure model is in evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    reconstructed_data = model(new_audio_data)
