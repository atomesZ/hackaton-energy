import numpy as np
import os

from copy import deepcopy
from random import shuffle, randrange

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
        if key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            key = key.lower()

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

def get_X_Y_vectorized_int(dataset: dict):
    """
    Get X and Y in right format for ML algorithms
    """
    X = []
    Y = []

    d_list = list(dataset)

    for k in dataset:
        X += dataset[k]

        temp = [0] * len(d_list)

        index_in_d_list = d_list.index(k)
    
        temp[index_in_d_list] = 1

        for i in range(len(dataset[k])):
            Y += [temp]

    assert len(X) == len(Y)
    return X, Y



def shuffle_X_Y(X: list, Y: list):
    length_data = len(X)
    assert length_data == len(Y)
    X = np.array(X)
    Y = np.array(Y)

    i_data = list(range(length_data))

    shuffle(i_data)
    
    
    X_shuffled = X[i_data]
    Y_shuffled = Y[i_data]
    
    return X_shuffled, Y_shuffled

def shift_letter(train_dataset, test_dataset):
    """Add SHIFT + LETTER to each dataset
    Call with : shift_letter(train_dataset, test_dataset)
    """
    dataset = train_dataset
    for key, val in test_dataset.items():
        dataset[key] += val
    
    arr_shift = dataset.get("SHIFT")
    for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        arr_key = []
        arr_letter = dataset.get(key.lower())
        for i in range(0, min(len(arr_shift), len(arr_letter))):
            arr_key.append(np.amax([arr_shift[i], arr_letter[i]], axis=0))
        train_dataset[key] = arr_key[:6000]
        test_dataset[key] = arr_key[6000:]


def ctrl_alt_suppr(train_dataset, test_dataset):
    """Add ALT, CTRL+ALT, CTRL+ALT+SHIFT to each dataset
    Call with : shift_letter(train_dataset, test_dataset)
    """
    dataset = train_dataset
    for key, val in test_dataset.items():
        dataset[key] += val
    
    arr_nokey = dataset.get("NOKEY")
    arr_suppr = dataset.get("SUPPR")
    arr_ctrl = dataset.get("CTRL")
    arr_alt, arr_ctrl_alt, arr_ctrl_alt_suppr = [], [], []
    min_len = min(len(arr_nokey), len(arr_suppr), len(arr_ctrl))
    for i in range(0, min_len):
        arr_alt_i = np.copy(arr_nokey[i])
        #randrange(3, 4)
        arr_alt_i[3] = randrange(150, 225) / 100
        arr_alt.append(arr_alt_i)
    for i in range(0, min_len):
        arr_ctrl_alt.append(np.amax([arr_alt[randrange(len(arr_alt))], arr_ctrl[i]], axis=0))
        arr_ctrl_alt_suppr.append(np.amax([arr_alt[randrange(len(arr_alt))], arr_ctrl[randrange(len(arr_ctrl))], arr_suppr[i]], axis=0))
    
    train_dataset["ALT"] = arr_alt[:6000]
    test_dataset["ALT"] = arr_alt[6000:]
    train_dataset["CTRL+ALT"] = arr_ctrl_alt[:6000]
    test_dataset["CTRL+ALT"] = arr_ctrl_alt[6000:]
    train_dataset["CTRL+ALT+SUPPR"] = arr_ctrl_alt_suppr[:6000]
    test_dataset["CTRL+ALT+SUPPR"] = arr_ctrl_alt_suppr[6000:]

def filtre_subtract(i, nokey, key, nbr_trunc):
    a = np.subtract(i, nokey)
    if not np.isnan(nbr_trunc):
        for j in range(len(a)):
            if a[j] <= nbr_trunc:
                a[j] = 0
    return a

def filtre(train_dataset, test_dataset, nbr_trunc):
    """Soustrait aux trames le NOKEY et applique un set à 0 aux valeurs en dessous de nbr_trunc"""
    
    d_list = list(test_dataset)
    train_nokey = np.array(train_dataset.get('NOKEY')).mean(0)
    test_nokey = np.array(test_dataset.get('NOKEY')).mean(0)

    train_data2, test_data2 = {}, {}
    for key in d_list:
        #Ne modifie pas les trames de type NOKEY
        if (key == "NOKEY"):
            train_data2.update({key : train_dataset.get(key)})
            test_data2.update({key : test_dataset.get(key)})
            continue
            
        #Soustrait la moyenne des NOKEY au dataset
        arr = []
        for i in train_dataset.get(key):
            arr.append(filtre_subtract(i, train_nokey, key, nbr_trunc))
        train_data2.update({key : arr})
        arr = []
        for i in test_dataset.get(key):
            arr.append(filtre_subtract(i, test_nokey, key, nbr_trunc))
        test_data2.update({key : arr})
    return train_data2, test_data2

def trame_distance(t1, t2):
    return np.linalg.norm(t1 - t2)

def deduplicate(X_loginmdp, delta=0.77777):
    X_loginmdp_dedup = []
    X_loginmdp_len = len(X_loginmdp)
    
    range_start = None

    for index, elm in enumerate(X_loginmdp):       
        if index + 1 == X_loginmdp_len:
            trames = X_loginmdp[range_start:index + 1]
            pics = []
            for i in range(17):
                mean = np.mean([e[i] for e in trames])
                pics.append(mean)

            X_loginmdp_dedup.append(pics)

            range_start = None
        else:
            dist = trame_distance(elm, X_loginmdp[index + 1])
            if dist < delta and range_start is None:
                range_start = index
            elif dist >= delta and range_start is not None:
                trames = X_loginmdp[range_start:index + 1]
                pics = []
                for i in range(17):
                    mean = np.mean([e[i] for e in trames])
                    pics.append(mean)

                X_loginmdp_dedup.append(pics)

                range_start = None
            elif dist >= delta and range_start is None:
                X_loginmdp_dedup.append(elm)

    return X_loginmdp_dedup