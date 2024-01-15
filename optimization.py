import numpy as np

def simplex_policy(preds):
    return np.argmax(preds)