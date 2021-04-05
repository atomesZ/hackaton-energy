import numpy as np

from read_pics import get_pics_from_file


def get_prediction_list(model):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    Prévoit/Estime les caractères du résultat grâce au modèle et supprime tous les 'NOKEY'.
    """

    x_pred, info = get_pics_from_file("../tohack/pics_LOGINMDP.bin")

    res = []

    predictions = model.predict(x_pred)
    
    for prediction in predictions:
        if prediction != 'NOKEY':
            res.append(prediction)
            
    return res


def get_creds(prediction_list):
    """Récupère et affiche tous les caractères du résultat grâce à prediction_list"""
    for i_k in range(len(prediction_list)):
        if prediction_list[i_k] == 'CTRL' and i_k + 2 < len(prediction_list) and prediction_list[i_k + 2] == 'SUPPR':
            print('#####################')
            print(prediction_list[i_k:i_k + 42])

def get_prediction_list_keras(model, d_list, pic="tohack/pics_LOGINMDP"):
    """
    The model is like "model = LinearRegression()" (or other algorithms)
    Prévoit/Estime les caractères du résultat grâce au modèle
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
    """Calcule la réelle précision de chaque caractère du modèle"""
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


        
def create_blocks_of_trames_prediction(trames_prediction: list, range_blk_lookup: int = 40):
    """
    This function gets the main key by blocks of key pressed to only have essential informations
    """
    
    def amplify_ctrl_alt_suppr(l: list, amplify_ratio: int = 2):
        """
        Because CTRL+ALT+SUPPR is detected few times but rightly, we amplify it, in order to affiliate a block to it
        """
        amplify_ratio -= 1
        for i in range(len(l)):
            if l[i] == "CTRL+ALT+SUPPR":
                for j in range(amplify_ratio):
                    l.append("CTRL+ALT+SUPPR")
    
    res = []
    i = 0
    length = len(trames_prediction)
    
    while i < length:
        
        caract = trames_prediction[i]
        
        # Get block limit default
        
        i_blk_end = i + range_blk_lookup
        
        # to avoid list out of bound exception
        if i_blk_end >= length:
            break
        
        sub_list = trames_prediction[i:i_blk_end]
        amplify_ctrl_alt_suppr(sub_list)
        
        # Get block value
        block_value = max(sub_list, key=sub_list.count)
        
        
        # Extend block limit to the edge
        
        next_sub_list = trames_prediction[i_blk_end:i_blk_end + range_blk_lookup]
        next_block_value = max(next_sub_list, key=next_sub_list.count)
        
        while next_block_value == block_value:
            
            i_blk_end += range_blk_lookup
            
            
            # to avoid list out of bound exception
            
            if i_blk_end + range_blk_lookup >= length:
                break
                
            next_sub_list = trames_prediction[i_blk_end:i_blk_end + range_blk_lookup]
            amplify_ctrl_alt_suppr(next_sub_list)
            
            next_block_value = max(next_sub_list, key=next_sub_list.count)
        
        
        
        # shrink to last occurence
        while i_blk_end >= 0 and trames_prediction[i_blk_end] != block_value:
            i_blk_end -= 1
        
        res.append((block_value, i, i_blk_end))
        
        
        i = i_blk_end
    
    
    return res

def clean_up_blocks_of_trames_prediction(trames_prediction: list, blocks_of_trames_prediction: list):
    def concatenate_alphanums(trames_prediction: list, blocks_of_trames_prediction: list):
        """
        This function cleans up by concatenating following alphanums characters
        """
        def is_alphanum(c: str):
            # We use a little trick here
            return len(c) == 1

        res = []

        length = len(blocks_of_trames_prediction)

        i = 0

        while i < length:

            real_key, key_start, key_end = blocks_of_trames_prediction[i]


            real_key_end = key_end


            i_has_been_incremented = False

            while i < length and is_alphanum(blocks_of_trames_prediction[i][0]):
                real_key_end = blocks_of_trames_prediction[i][2]
                i += 1
                i_has_been_incremented = True

            if key_end != real_key_end:
                sub_list = trames_prediction[key_start:real_key_end]

                # Get real block value
                real_key = max(sub_list, key=sub_list.count)

            res.append(real_key)
            if not i_has_been_incremented:
                i += 1

        return res

    def remove_following_key_duplicates(prediction: list):
        res = []

        last_key = ""

        for key in prediction:
            if key != last_key:
                res.append(key)
            last_key = key

        return res
    
    res = concatenate_alphanums(trames_prediction, blocks_of_trames_prediction)
    res = remove_following_key_duplicates(res)
    
    return res


def get_credentials(key_pressed: list):
    """
    key_pressed is the result of clean_up_blocks_of_trames_prediction
    
    """
    def isolate_credentials_sequence(key_pressed: list):
        """
        return the list of the sequence of the password, None if not found
        """
        i = 0
        length = len(key_pressed)
        # Look for "CTRL+ALT+SUPPR"
        while i < length and key_pressed[i] != "CTRL+ALT+SUPPR":
            i += 1
        
        # if "CTRL+ALT+SUPPR" not found
        if i == length:
            return None
        
        sequence_start = i
        
        # Look for "ENTER"
        while i < length and key_pressed[i] != "ENTER":
            i += 1
        
        # if "CTRL+ALT+SUPPR" not found
        if i == length:
            return None
        
        sequence_end = i
        
        return key_pressed[sequence_start:sequence_end]
        
    
    credentials = ""
    
    sequence = isolate_credentials_sequence(key_pressed)
    
    if sequence is None:
        print("Could not find the sequence of the password ('CTRL+ALT+SUPPR' and 'ENTER')")
        return ""
    
    on_a_shift = False
    
    for k in sequence:
        if len(k) == 1:
            if not on_a_shift:
                credentials += k.lower()
            else:
                credentials += k
        
    return credentials