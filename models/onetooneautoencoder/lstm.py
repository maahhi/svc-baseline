import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import psutil
import os
import time
import gc


class VoiceTransformNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(VoiceTransformNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Encoder
        outputs, (hn, cn) = self.encoder(x)

        # Decoder
        decoder_outputs, _ = self.decoder(outputs, (hn, cn))
        out = self.fc(decoder_outputs)
        return out


def main():
    writer = SummaryWriter('runs/experiment_name')
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2  # return in MB
    # Before processing
    memory_before = get_memory_usage()
    # Start the clock
    start_time = time.time()

    input_dim = 1027
    hidden_dim = 256
    output_dim = 1027
    n_layers = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VoiceTransformNet(input_dim, hidden_dim, output_dim, n_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 200


    # data loading
    data_run = 'pairwise_run2'
    Xtrain = load('./../../data-preprocess/'+data_run+'/Xtrain.joblib')
    Ytrain = load('./../../data-preprocess/'+data_run+'/Ytrain.joblib')
    song_id = load('./../../data-preprocess/'+data_run+'/new_song_id.joblib')


    X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).permute(0, 2, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1).to(device)


    dataset = TensorDataset(X_train, Y_train)
    batch_size = 32  # set to a smaller value if needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    loss_values = []
    # Assume X_train is your data for singer 1 and Y_train is for singer 2
    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
        loss_values.append(loss.item())
    outputs = model(X_test)
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss_values': loss_values  # save the list of loss values here
    }, 'checkpoint.pth')

    # delet all cuda and model related variables
    del model
    del optimizer
    #del criterion
    #del outputs
    del X_train
    del Y_train
    del X_test
    #del Y_test
    del dataset
    del train_loader
    torch.cuda.empty_cache()
    gc.collect()

    val_loss = criterion(outputs, Y_test)
    print(f"Validation Loss: {val_loss.item()}")

    memory_after = get_memory_usage()

    print(f"Memory used: {memory_after - memory_before} MB")
    # End the clock
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

if "__name__" == "__main__" :
    main()