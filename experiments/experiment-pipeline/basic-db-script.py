"""
Basic reference code for running debiasing and storing models for arbitrary data
"""

import sys 
import os
import dill
import numpy as np

sys.path.append('../../src')
import policies 
import bbDebiasing
import maxEnsembleDebias

"""
Helper script to run all the debiasing on models.
"""

def main():
    datapath, modelpath, policy_name, experiment_name = sys.argv[1:]

    xs = np.loadtxt(f"{datapath}/features.csv", delimiter=',')
    ys = np.loadtxt(f"{datapath}/labels.csv.csv", delimiter=",")

    model_files = os.listdir(modelpath)
    
    models = []
    for filename in model_files:
        with open(f"{modelpath}/{filename}", 'rb') as file:
            models.append(dill.load(file))
    
    pred_dim = ys.shape[1]

    db_model_path = f"./debiased-models/{experiment_name}"

    if policy_name=='simplex':
        # bit silly syntax, bc wrote code in way that we could theoretically combine different policies,
        # but in practice we never do. 
        pols = [policies.Simplex(pred_dim, models[0])]*len(models) 
    elif policy_name=='variance':
        gran = 0.1
        var_limit = 0.5
        pols = [policies.VarianceConstrained(pred_dim, models[0], gran, var_limit, ys)]*len(models)
    elif policy_name=='linear-constraint':
        gran = 0.1
        pols = [policies.LinearMin(pred_dim, models[0], gran)]*len(models)
    else:
        print("Need to properly specify policy")
        return -1

    max_depth = 1
    tolerance = 0.01

    def init_model(xs):
        return np.tile(np.mean(ys, axis=0), (len(xs),1))
    

    bbModel = bbDebiasing.bbDebias(init_model, pols[0], xs, ys, max_depth, tolerance)
    bbModel.debias(models, pols)

    model_file = f"{db_model_path}_BBModel.pkl"
    with open(model_file, 'wb') as file:
        dill.dump(bbModel, file)
    
    maxModel = maxEnsembleDebias.EnsembledModel(models, pols, xs, ys, max_depth, tolerance)
    maxModel.debias()

    model_file = f"{db_model_path}_MaxEnsemble.pkl"
    with open(model_file, 'wb') as file:
        dill.dump(bbModel, file)



if __name__ == '__main__':
    main()