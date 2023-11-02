import os
import time
import psutil
from CNN_eval_preprocessor import extract_features, get_memory_usage, make_new_dir
from joblib import dump, load

def main():
    memory_before = get_memory_usage()  # Before processing
    start_time = time.time()  # Start the clock
    log = [time.ctime()]  # add runtime data and time to the log file

    """main code starts here"""

    # Parameters
    sample_rate = 16000
    log.append(f'{sample_rate=}')
    dataset_path = r'C:\Users\maahh\Downloads\Compressed\VocalSet1-2\data_by_singer'
    log.append(f'{dataset_path=}')
    limit_of_lables = 1
    log.append(f'{limit_of_lables=}')
    lable_list = ['male1']
    log.append(f'{lable_list=}')
    new_run_dir = make_new_dir()
    log.append(f'{new_run_dir=}')

    labels = sorted(os.listdir(dataset_path))
    for label_id, label in enumerate(labels):
        if len(lable_list) > 0 :
            if label not in lable_list:
                continue
        elif label_id>limit_of_lables-1:
            break
        print(label_id,label)
        class_folder = os.path.join(dataset_path, label)
        for root, _, files in os.walk(class_folder):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    mel_db = extract_features(audio_path,sample_rate)
                    audio_sub_path = audio_path[len(dataset_path)+1:-4]
                    output_data_path= os.path.join(new_run_dir, audio_sub_path)
                    output_data_dir = os.path.dirname(output_data_path)
                    os.makedirs(os.path.dirname(output_data_dir+'\\'), exist_ok=True)
                    dump(mel_db,  output_data_path+ '.joblib')



    """main code ends here"""

    memory_after = get_memory_usage()
    log.append(f"Memory used: {memory_after - memory_before} MB")
    # End the clock
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    log.append(f"Time taken: {elapsed_time:.2f} seconds")
    # Save the log file
    with open('./'+new_run_dir+'/log.txt', 'w') as f:
        for item in log:
            f.write("%s\n" % item)
if __name__ == "__main__" :
    print('an')
    main()