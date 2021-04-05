import matplotlib.pyplot as plt

from dataset_manager import *
from termcolor import colored

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

def trame_show_AllKey(dataset):
    """Afficher les 100eres trames de toutes les keys"""
    d_list = list(dataset)
    for i in range(len(d_list)):
        trame_show(dataset.get(d_list[i]), d_list[i], i)

def trame_show_key(dataset, key):
    """Afficher les 100eres trames d'une key particulière
    key : str
    """
    d_list = list(dataset)
    for i in range(len(d_list)):
        if (d_list[i] == key):
            trame_show(dataset.get(d_list[i]), d_list[i], 0)
            

def trame_show_only_result_sup_accuracy(res, accuracy, accur_lim):
    """Trouve les trames de caractères dont l'accuracy est supérieure à 'accur_lim' et qui sont présents dans le résultat"""
    #out = ""
    X_login_mdp, _ = get_pics_from_file("../tohack/pics_LOGINMDP.bin")
    trames80 = {}
    for index, key in enumerate(res):
        if key not in []:
            #out += colored(key, 'red' if accuracy[key] >= accur_lim else 'grey') + " "
            if accuracy[key] >= accur_lim:
                if key not in trames80:
                    trames80[key] = []
                trames80[key].append(X_login_mdp[index])
    return trames80
            
    
def trame_show_result(dataset, accuracy, res, accur_lim):
    """Affiche les trames de caractères dont l'accuracy est supérieure à 'accur_lim' et qui sont présents dans le résultat"""
    trames = trame_show_only_result_sup_accuracy(res, accuracy, accur_lim)
    print(list(trames.keys()))
    d_list = list(dataset)

    for key in trames:
        trame_show_key(dataset, 'NOKEY')
        trame_show_key(dataset, key)
        trame_show(trames[key], f"{key} - LOGINMDP", 0)
        print('===========================')