
import numpy as np
from joblib import dump, load
import os


def make_new_dir():
    path = os.getcwd()
    print(path)
    dirs = os.listdir(path)
    run_dirs = []
    for dir in dirs:
        if dir.startswith("pairwise_run"):
            run_dirs.append(dir)
    run_dirs.sort()
    new_dir = "pairwise_run" + str(int(run_dirs[-1][3:]) + 1)
    os.mkdir(new_dir)
    return new_dir

# prepare data for Auto encoder
data_run = 'run8'
X = load('./'+data_run+'/data.joblib')
Y = load('./'+data_run+'/labels.joblib')
songs = load('./'+data_run+'/songs.joblib')
# Find indices where Y is 0 or 1
filtered_indices = np.where((Y == 0) | (Y == 9))[0]

# Use these indices to filter X and Y
X = X[filtered_indices]
Y = Y[filtered_indices]
songs = songs[filtered_indices]
#gc.collect()

song_pice = np.zeros_like(songs)  # Create an ndarray of the same shape but filled with zeros
song_id = np.zeros_like(songs)

# Iterate over the data array
for i in range(1, len(songs)):
    if songs[i] == songs[i-1]:
        song_pice[i] = song_pice[i-1] + 1
    else:
        song_pice[i] = 0


songs = np.column_stack((songs, song_pice))
print(songs)

# Group indices of identical rows in songs
unique_rows, return_inverse = np.unique(songs, axis=0, return_inverse=True)
buckets = [[] for _ in range(unique_rows.shape[0])]
for i, group_id in enumerate(return_inverse):
    buckets[group_id].append(i)

# Filter out groups with only one index
duplicate_groups = [group for group in buckets if len(group) > 1]

print(duplicate_groups)

for j in range(len(duplicate_groups)):
    for i in duplicate_groups[j]:
        song_id[i] = j


remove_zeros_song_id = np.where(song_id != 0)[0]
X = X[remove_zeros_song_id]
Y = Y[remove_zeros_song_id]
song_id = song_id[remove_zeros_song_id]
#

# Get indices where Y has value 0
indices_singer0 = np.where(Y == 0)[0]

# Extract rows from X where Y == 0
Xtrain = X[indices_singer0]

# For each song_id from the filtered Xtrain, find the row in X with the same song_id but Y == 1
new_song_id = []
Ytrain_list = []
for idx in indices_singer0:
    song_id_val = song_id[idx]
    new_song_id.append(song_id_val)
    paired_song_idx = np.where((song_id == song_id_val) & (Y == 9))[0]
    if paired_song_idx.size:
        Ytrain_list.append(X[paired_song_idx[0]])

Ytrain = np.array(Ytrain_list)
#new_run_dir = make_new_dir()
new_run_dir = 'pairwise_run2'
dump(Xtrain, './'+new_run_dir+'/Xtrain.joblib')
dump(Ytrain, './'+new_run_dir+'/Ytrain.joblib')
dump(new_song_id,'./'+new_run_dir+'/new_song_id.joblib')


