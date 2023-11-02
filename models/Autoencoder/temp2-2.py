import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchaudio
import os
import time
import librosa
from torch.utils.tensorboard import SummaryWriter
import psutil
from datetime import datetime


class SingerDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000):
        self.sample_rate = sample_rate
        # data_path is a directory that in a nested structure of it there is .wav files. create a list of the paths of the .wav files
        self.data_path_list = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    a = torchaudio.info(audio_path).num_frames
                    b = librosa.get_duration(filename=audio_path)
                    duration = torchaudio.info(audio_path).num_frames // self.sample_rate
                    duration = int(librosa.get_duration(filename=audio_path))
                    for i in range(duration):
                        self.data_path_list.append([audio_path, i])
        print(len(self.data_path_list))

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        audiopath, start_time = self.data_path_list[idx]
        # audio, sr = librosa.load(audiopath, sr=self.sample_rate, offset=start_time, duration=1)
        audio, sr = torchaudio.load(audiopath, frame_offset := start_time * self.sample_rate,
                                    num_frames=1 * self.sample_rate)
        return audio.reshape(1, -1)


# Define the model architecture
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.lstm = nn.LSTM(32, hidden_size, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Adjust to (batch_size, seq_len, num_features)
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 4, output_size)

    def forward(self, x, encoder_outputs):
        x, (hidden, cell) = self.lstm(x)
        context, attention_weights = self.attention(encoder_outputs)
        context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat((x, context), dim=2)
        x = self.fc(x)
        return x, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x, target_length):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_input = torch.zeros((x.size(0), target_length, hidden.size(-1) * 2)).to(x.device)
        decoder_outputs, attention_weights = self.decoder(decoder_input, encoder_outputs)
        return decoder_outputs, attention_weights


# Define custom Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores = self.fc(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # return in MB


def save_checkpoint(epoch, model, optimizer, filename='checkpoint.pth'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']


def train(model_path=None):
    # Set hyperparameters
    input_size = 1
    hidden_size = 64
    output_size = 1
    batch_size = 1
    seq_len = 16000
    output_seq_len = 16000
    num_epochs = 15000
    checkpoint_saver = 100
    summary_param_saver = 100
    # result_path = '/content/drive/MyDrive/autoencoder/'
    result_path = 'runs/'
    # data_path = '/content/drive/MyDrive/VocalSet/female1'
    data_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer\female1'

    formatted_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    writer = SummaryWriter(result_path + 'experiment_name')

    # Create example data and DataLoader
    train_dataloader = DataLoader(SingerDataset(data_path), batch_size=batch_size, shuffle=True)
    print('Data loaded')

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = Seq2Seq(input_size, hidden_size, output_size).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    loss_values = []
    start_epoch = 0
    if model_path:
        start_epoch = load_checkpoint(model_path, model, optimizer)
        start_epoch += 1  # We want to start from the next epoch

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        for iter, inputs in enumerate(train_dataloader):
            inputs = inputs.to(device)

            # Forward pass
            outputs, attention_weights = model(inputs, target_length=output_seq_len)
            # Calculate loss
            loss = criterion(outputs, inputs)
            epoch_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                # print(f"iter {iter + 1}, Loss: {loss.item():.8f}")
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + iter)
            if iter % 700 == 0:
                writer.add_audio('Sample Output', outputs.view(-1), epoch, sample_rate=16000)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
        if (epoch - start_epoch) % summary_param_saver == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'{name}.grad', param.grad.clone().cpu().data.numpy(), epoch)
            # Assuming attention_weights is of shape (batch_size, seq_len)
            for i, weights in enumerate(attention_weights):
                writer.add_histogram(f'Attention Weights/{i}', weights.clone().cpu().data.numpy(), epoch)
        if (epoch - start_epoch) % checkpoint_saver == 0:
            save_checkpoint(epoch, model, optimizer, result_path + f'checkpoint_epoch_{epoch}.pth')

        average_epoch_loss = epoch_loss / len(train_dataloader)
        loss_values.append(average_epoch_loss)

    # Save at the end of training
    save_checkpoint(epoch, model, optimizer,
                    result_path + f'/content/drive/MyDrive/autoencoder/checkpoint_epoch_{epoch}.pth')

    # Model evaluation (you can use a separate validation set here)
    model.eval()
    with torch.no_grad():
        sample_input = torch.randn(1, input_size, seq_len).to(device)
        output, attention_weights = model(sample_input, target_length=output_seq_len)


def time_memory_usage(funtion_name):
    # Before processing
    memory_before = get_memory_usage()
    # Start the clock
    start_time = time.time()

    funtion_name()

    memory_after = get_memory_usage()
    print(f"Memory used: {memory_after - memory_before} MB")
    # End the clock
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


time_memory_usage(train)

