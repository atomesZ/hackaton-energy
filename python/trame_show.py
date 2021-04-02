import matplotlib.pyplot as plt
from dataset_manager import *


train_dataset, test_dataset = get_dataset()
d_list = list(test_dataset)

def trame_show(X, key, figure_nbr):
    """Afficher les 100eres trames"""
    plt.figure(figure_nbr)
    for i in X[:100]:
        plt.plot(range(1, len(i) + 1), i)
    
    plt.xlim(1, 17)
    plt.ylim(0, 2.5)
    plt.xlabel("freq")
    plt.ylabel("Intensité")
    plt.title("{0}, 100 trames".format(key))
    plt.grid()
    plt.show()

def trame_show_AllKey():
    """Afficher les 100eres trames de toutes les keys"""
    for i in range(len(d_list)):
        trame_show(train_dataset.get(d_list[i]), d_list[i], i)

def trame_show_key(key):
    """Afficher les 100eres trames d'une key particulière
    key : str
    """
    for i in range(len(d_list)):
        if (d_list[i] == key):
            trame_show(train_dataset.get(d_list[i]), d_list[i], 0)

