import numpy as np

from read_pics import get_pics_from_file


def get_prediction_list(model):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    """

    x_pred, info = get_pics_from_file("../tohack/pics_LOGINMDP.bin")

    res = []

    predictions = model.predict(x_pred)
    
    for prediction in predictions:
        if prediction != 'NOKEY':
            res.append(prediction)
            
    return res


def get_creds(prediction_list):
    for i_k in range(len(prediction_list)):
        if prediction_list[i_k] == 'CTRL' and i_k + 2 < len(prediction_list) and prediction_list[i_k + 2] == 'SUPPR':
            print('#####################')
            print(prediction_list[i_k:i_k + 42])

def get_prediction_list_keras(model, d_list, pic="tohack/pics_LOGINMDP"):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    """

    x_pred, info = get_pics_from_file(f"../{pic}.bin")
    
    res = []

    preds_brut = model.predict(np.array(x_pred))
    for pred in preds_brut:
        itemindex = np.where(pred==max(pred))[0][0]
        prediction = d_list[itemindex]
        #if "data" in pic or prediction != 'NOKEY':
        res.append(prediction)
    return res

def get_prediction_list2_keras(model, d_list, pic="tohack/pics_LOGINMDP"):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    """

    x_pred, info = get_pics_from_file(f"../{pic}.bin")
    
    res = []

    preds_brut = model.predict(np.array(x_pred))
    for pred in preds_brut:
        itemindex1 = np.where(pred==max(pred))[0][0]
        pred[itemindex1] = 0
        itemindex2 = np.where(pred==max(pred))[0][0]
        prediction = (d_list[itemindex1], d_list[itemindex2])
        if "data" in pic or prediction != 'NOKEY':
            res.append(prediction)
    return res
            
def compute_accuracy_keras(model, d_list, X_test, Y_test):
    accuracy, accuracy_count = {}, {}
    for e in d_list:
        accuracy[e] = 0
        accuracy_count[e] = 0
 
    preds_brut = model.predict(np.array(X_test))
    
    for pred, y in zip(preds_brut, Y_test):
        realitemindex = np.where(y==1)[0][0]
        key = d_list[realitemindex]
        itemindex = np.where(pred==max(pred))[0][0]
        accuracy[key] += realitemindex == itemindex
        accuracy_count[key] += 1


    for key in accuracy:
        accuracy[key] /= max(accuracy_count[key], 1)

    return accuracy
