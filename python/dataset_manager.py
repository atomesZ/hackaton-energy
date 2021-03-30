import numpy as np
import os

from random import shuffle

from read_pics import get_pics_from_file


def get_dataset():
    """
    returns the dataset in a dict with key : name of the key
    value: list of spikes
    """
    train_dataset = {}
    test_dataset = {}
    os.chdir('../data/')

    for filename in os.listdir():
        list_of_spikes, info = get_pics_from_file(filename)

        #We get the name of the key by removing "pics_" at the beginning and ".bin" at the end
        key = filename[5:-4]

        train_dataset[key] = list_of_spikes[:6000]
        test_dataset[key]= list_of_spikes[6000:]

    return train_dataset, test_dataset


def get_X_Y(dataset: dict):
    """
    Get X and Y in right format for ML algorithms
    """
    X = []
    Y = []
    for k in dataset:
        X += dataset[k]
        Y += [k] * len(dataset[k])


    assert len(X) == len(Y)        
    return X, Y


def shuffle_X_Y(X: list, Y: list):
    length_data = len(X)
    assert length_data == len(Y)
    
    
    i_data = list(range(length_data))

    shuffle(i_data)
    
    
    X_shuffled = X[i_data]
    Y_shuffled = Y[i_data]
    
    return X_shuffled, Y_shuffled
