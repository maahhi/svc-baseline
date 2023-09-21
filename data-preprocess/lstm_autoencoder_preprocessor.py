
import numpy as np
from joblib import dump, load
import os

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
#new_run_dir = make_new_dir()
new_run_dir = 'pairwise_run2'
dump(X, './'+new_run_dir+'/data.joblib')
dump(Y, './'+new_run_dir+'/labels.joblib')
dump(song_id,'./'+new_run_dir+'/song_id.joblib')


