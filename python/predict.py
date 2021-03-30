import numpy as np

from read_pics import get_pics_from_file


def get_prediction_list(model):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    """

    x_pred, info = get_pics_from_file("../tohack/pics_LOGINMDP.bin")

    res = []

    poor_progress_bar = 0
    for i in range(len(x_pred) // 400 + 1):
        print('#', end="")
    print("")
    for trame_pred in x_pred:
        if poor_progress_bar % 400 == 0:
            print('#', end="")
        poor_progress_bar += 1
        t = [np.array(trame_pred)]
        prediction = model.predict(t)[0]
        if prediction != 'NOKEY':
            res.append(prediction)

    print("")
    return res


def get_creds(prediction_list):
    for i_k in range(len(prediction_list)):
        if prediction_list[i_k] == 'CTRL' and i_k + 2 < len(prediction_list) and prediction_list[i_k + 2] == 'SUPPR':
            print('#####################')
            print(prediction_list[i_k:i_k + 42])
