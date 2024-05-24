import dill
import numpy as np
import sys
import os

import dataGeneration
import modelSetup

sys.path.append('../../src')
import policies 
import bbDebiasing
import maxEnsembleDebias

def generateOutOfSample(bbModel, wbModel, experimentName, test_x, test_y):
    # Generate predictions, predicted payoff, and realized payoff

    bbPreds, bbTrans = bbModel.predict(test_x)
    wbPreds, wbTrans = wbModel.predict(test_x)

    wbPredPayoff = wbModel.getPredPayoff(wbTrans)
    wbRealPayoff = wbModel.getRealPayoff(wbTrans, test_y)

    bbPredPayoff = bbModel.getPredPayoff(bbTrans)
    bbRealPayoff = bbModel.getRealPayoff(bbTrans, test_y)


    # Store

    folder = 'out-of-sample'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    if not os.path.exists(f"{folder}/{experimentName}"):
        os.makedirs(f"{folder}/{experimentName}")

    np.save(f"{folder}/{experimentName}/bbPreds.npy", bbPreds)
    np.save(f"{folder}/{experimentName}/wbPreds.npy", wbPreds)
    np.save(f"{folder}/{experimentName}/wbPredPayoff.npy", wbPredPayoff)
    np.save(f"{folder}/{experimentName}/wbRealPayoff.npy", wbRealPayoff)
    np.save(f"{folder}/{experimentName}/bbPredPayoff.npy", bbPredPayoff)
    np.save(f"{folder}/{experimentName}/bbRealPayoff.npy", bbRealPayoff)

    with open(f"{folder}/{experimentName}/bbTranscript.pkl", 'wb') as file:
        dill.dump(bbTrans, file)
    with open(f"{folder}/{experimentName}/wbTranscript.pkl", 'wb') as file:
        dill.dump(wbTrans, file)


def main():

    pathA = ['linear-label_gb_coord_variance_5000_subsample400_BBModel.pkl',
        'linear-label_gb_coord_variance_5000_subsample400_MaxEnsemble.pkl']
    pathB = ['linear-label_gb_group_variance_5000_subsample400_BBModel.pkl',
        'linear-label_gb_group_variance_5000_subsample400_MaxEnsemble.pkl']
    pathC = ['linear-label_gb_coord_linear-constraint_5000_subsample400_BBModel.pkl',
        'linear-label_gb_coord_linear-constraint_5000_subsample400_MaxEnsemble.pkl']
    pathD = ['linear-label_gb_group_linear-constraint_8000_subsample400_BBModel.pkl',
        'linear-label_gb_group_linear-constraint_8000_subsample400_MaxEnsemble.pkl']

    # pathA = ['linear-label_gb_coord_variance_10_subsample5_BBModel.pkl',
    # 'linear-label_gb_coord_variance_10_subsample5_MaxEnsemble.pkl']
    # pathB = ['linear-label_gb_group_variance_10_subsample5_BBModel.pkl',
    # 'linear-label_gb_group_variance_10_subsample5_MaxEnsemble.pkl']
    # pathC = ['linear-label_gb_coord_linear-constraint_10_subsample5_BBModel.pkl',
    # 'linear-label_gb_coord_linear-constraint_10_subsample5_MaxEnsemble.pkl']
    # pathD = ['linear-label_gb_group_linear-constraint_10_subsample5_BBModel.pkl',
    # 'linear-label_gb_group_linear-constraint_10_subsample5_MaxEnsemble.pkl']

    pathsets = [pathA, pathB, pathC, pathD]
    experimentNames = ["A","B","C","D"]

    data_path = '../../data/synthetic'
    test_x = np.loadtxt(f"{data_path}/features_test.csv", delimiter=',')
    test_y = np.loadtxt(f"{data_path}/linear-labels_test.csv", delimiter=',')
    for i in range(len(pathsets)):
        bbPath = f'debiased-models/{pathsets[i][0]}'
        wbPath = f'debiased-models/{pathsets[i][1]}'
        with open(bbPath, 'rb') as file:
            bbModel = dill.load(file)
        with open(wbPath, 'rb') as file:
            wbModel = dill.load(file)
        
        generateOutOfSample(bbModel, wbModel, experimentNames[i], test_x, test_y)

if __name__ == '__main__':
    main()