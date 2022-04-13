import json
import os
import re
import librosa
from numpy import mat
import math

from torch import sign

DATASET_PATH = 'Data/genres_original/'
JSON_PATH = 'data.json'
SAMPLE_RATE = 22050
DURATION = 30
SAMPLE_PER_TRACK = SAMPLE_RATE * DURATION 


def save_mfcc(dataset_path , json_path , n_mfcc = 13, n_fft = 2048, hop_length=512 , num_segment=5 ):
    

    #dictonary to save data ://
    data = {
        "mapping" : [],
        "mfcc" : [],
        "label" : []
    }

    num_samples_per_segment = int(SAMPLE_PER_TRACK / num_segment)
    expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for i , (dirpath , dirnames , filenames )  in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path : 
            
            dirpath_components = dirpath.split('\\')
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)

            print("\nProcessing {}".format(semantic_label) )


            for f in filenames:
                extra_songs = f.split(".")
                if int(extra_songs[1]) < 10 :
                    
                    print(extra_songs , "\n\n")
                    file_path = os.path.join(dirpath , f)
                    signal , sr = librosa.load(file_path , sr=SAMPLE_RATE)
                    # print("File path {} , fileName : {} \n".format(file_path , f))
                    for s in range(num_segment):
                        start_sample = num_samples_per_segment * s 
                        finish_sample = start_sample + num_samples_per_segment 

                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample] , 
                         sr=sr,
                        n_fft = n_fft,
                        n_mfcc = n_mfcc , 
                        hop_length = hop_length)

                        mfcc = mfcc.T #need to transpose 

                        if len(mfcc) == expected_num_mfcc_vector_per_segment :
                            data["mfcc"].append(mfcc.tolist())
                            data["label"].append(i-1)
                            print("{} segment:{}".format(file_path , s))
            print("\n")
    with open(json_path , "w") as fp:
        json.dump(data , fp , indent=4)


if __name__ == "__main__" :
    save_mfcc(dataset_path=DATASET_PATH ,json_path=JSON_PATH )

