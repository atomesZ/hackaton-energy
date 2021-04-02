import os
import time
import threading
import numpy
import multiprocessing

from copy import deepcopy
from tqdm import tqdm

from read_pics import *

NBTHREADS = multiprocessing.cpu_count() + 1

def get_dataset():
    dataset = {}
    os.chdir('../data/')

    for filename in os.listdir():
        list_of_spikes, info = get_pics_from_file(filename)

        #We get the name of the key by removing "pics_" at the beginning and ".bin" at the end
        key = filename[5:-4]

        dataset[key] = list_of_spikes
    return dataset

def get_pics_from_loginmdp():
    return get_pics_from_file("../tohack/pics_LOGINMDP.bin")[0]

def trame_distance(t1, t2):
    return numpy.linalg.norm(t1 - t2)

class ComputeThread(threading.Thread):
    decoded_keys = []

    def __init__(self, threadID, dataset, loginmdp_part, pbar):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.dataset = dataset
        self.loginmdp_part = loginmdp_part
        self.pbar = pbar
        self.decoded_keys = []

    def run(self):
        print(f"Thread {self.threadID} running | {len(self.loginmdp_part)} trames")
        pbar.update(0)

        for mdp_trame in self.loginmdp_part:
            min_key = (None, float('inf'))
            for key, trames in self.dataset.items():
                min_d = float('inf')
                for trame in trames:
                    d = trame_distance(mdp_trame, trame)
                    if min_d > d:
                        min_d = d
                if min_key[1] > min_d:
                    min_key = (key, min_d)
            pbar.update(1)
            self.decoded_keys.append(min_key[0])

        print(f"Thread {self.threadID} done")

dataset = get_dataset()
loginmdp = get_pics_from_loginmdp()
loginmdp_len = len(loginmdp)

decoded_keys = []

with tqdm(total=loginmdp_len) as pbar:
    #for mdp_trame in loginmdp:
    #    min_key = (None, float('inf'))
    #    for key, trames in dataset.items():
    #        min_d = float('inf')
    #        for trame in trames:
    #            d = trame_distance(mdp_trame, trame)
    #            if min_d > d:
    #                min_d = d
    #        if min_key[1] > min_d:
    #            min_key = (key, min_d)
    #    pbar.update(1)
    #    decoded_keys.append(min_key[1])

    threads = []
    trames_split = numpy.array_split(loginmdp, NBTHREADS)
    for i in range(NBTHREADS):
        threads.append(ComputeThread(i, deepcopy(dataset), deepcopy(trames_split[i]), pbar))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for t in threads:
        decoded_keys += t.decoded_keys

print(decoded_keys)

with open('result_hackaton.txt', 'w+') as f:
    f.write('\n'.join(decoded_keys))
