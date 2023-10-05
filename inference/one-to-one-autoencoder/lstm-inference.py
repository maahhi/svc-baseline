import torch
from joblib import dump, load
from sklearn.model_selection import train_test_split
from models.onetooneautoencoder.lstm import VoiceTransformNet
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data loading
data_run = 'pairwise_run2'
Xtrain = load('./../../data-preprocess/'+data_run+'/Xtrain.joblib')
Ytrain = load('./../../data-preprocess/'+data_run+'/Ytrain.joblib')
song_id = load('./../../data-preprocess/'+data_run+'/new_song_id.joblib')

X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=42)
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1).to(device)

input_dim = 1027
hidden_dim = 256
output_dim = 1027
n_layers = 1
model = VoiceTransformNet(input_dim, hidden_dim, output_dim, n_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Load
loading = True
if loading :
    checkpoint = torch.load('../../models/onetooneautoencoder/checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    prev_loss = checkpoint['loss_values']
model = model.to(device)
outputs = model(X_test)
val_loss = criterion(outputs, Y_test)
print(f"Validation Loss: {val_loss.item()}")

outputs = outputs.permute(0, 2, 1).cpu().detach().numpy()
Y_test = Y_test.permute(0, 2, 1).cpu().detach().numpy()
X_test = X_test.permute(0, 2, 1).cpu().detach().numpy()
new_run_dir = 'inf1'
dump(X_test, './'+new_run_dir+'/X_test.joblib')
dump(outputs, './'+new_run_dir+'/outputs.joblib')
dump(Y_test, './'+new_run_dir+'/Y_test.joblib')