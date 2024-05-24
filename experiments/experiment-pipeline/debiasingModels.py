import sys 
import os
import dill
import numpy as np
import time

sys.path.append('../../src')
import policies 
import bbDebiasing
import maxEnsembleDebias

"""
Helper script to run all the debiasing on models.
"""

rng = np.random.default_rng(seed=42) #setting random number generator w set seed to use throughout

def main():
    label_version, model_type, specialization, policy_name, max_depth, subsample_size = sys.argv[1:]
    max_depth=int(max_depth)
    subsample_size=int(subsample_size)

    datapath = f"../../data/synthetic"
    xs = np.loadtxt(f"{datapath}/features.csv", delimiter=',')
    if label_version=="linear-label":
        ys = np.loadtxt(f"{datapath}/linear-labels.csv", delimiter=',')
    elif label_version=="poly-label":
        ys = np.loadtxt(f"{datapath}/poly-labels.csv", delimiter=',')
    else:
        print("Need to specify either linear or polynomal as label version in first command-line argument.")
        return -1
    
    # subsampling the datapoints for debiasing for computation's sake
    ind = rng.choice(np.arange(len(xs)), size=subsample_size)
    xs = xs[ind]
    ys = ys[ind]
    
    path = f"init-models/{label_version}/{model_type}"
    all_model_files = os.listdir(path)

    if specialization=='coord':
        model_files = [model_file for model_file in all_model_files if ('coord' in model_file) ]
    elif specialization=='group':
        model_files = [model_file for model_file in all_model_files if ('group' in model_file) ]
    elif specialization=='all':
        model_files = all_model_files
    
    models = []
    for filename in model_files:
        with open(f"{path}/{filename}", 'rb') as file:
            models.append(dill.load(file))
    
    pred_dim = ys.shape[1]

    if policy_name=='simplex':
        # bit silly syntax, bc wrote code in way that we could theoretically combine different policies,
        # but in practice we never do. 
        pols = [policies.Simplex(pred_dim, models[0])]*len(models) 
    elif policy_name=='variance':
        gran = 0.1
        # have to choose variance limit that is somewhat on same scale as the covariance matrix so that the problem is feasible
        if label_version=="linear-label":
            var_limit = 300 
        elif label_version=="poly-label":
            var_limit = 2e13
        pols = [policies.VarianceConstrained(pred_dim, models[0], gran, var_limit, ys)]*len(models)
    elif policy_name=='linear-constraint':
        gran = 0.1
        linear_constraint = np.array([[1,1,0,0], [0,1,1,0]])
        max_val = np.array([0.5,0.6])
        gran = 0.1
        pols = [policies.Linear(pred_dim, models[0], gran, linear_constraint, max_val)]*len(models)
    else:
        print("Need to properly specify policy")
        return -1
    
    db_model_path = f"./debiased-models/{label_version}_{model_type}_{specialization}_{policy_name}_{max_depth}_subsample{subsample_size}"

    if not os.path.exists('debiased-models'):
        os.makedirs('debiased-models')
    
    tolerance = 0.01

    def init_model(xs):
        return np.tile(np.mean(ys, axis=0), (len(xs),1))
    
    start_time = time.time()
    print("Running BB")
    bbModel = bbDebiasing.bbDebias(init_model, pols[0], xs, ys, max_depth, tolerance)
    bbModel.debias(models, pols)
    bb_time = time.time() - start_time

    with open('time.txt', "a") as file:
        file.write(f'BB time: {bb_time} \n')

    model_file = f"{db_model_path}_BBModel.pkl"
    with open(model_file, 'wb') as file:
        dill.dump(bbModel, file)
    
    start_time = time.time()
    print("Running Ensembling")
    maxModel = maxEnsembleDebias.EnsembledModel(models, pols, xs, ys, max_depth, tolerance)
    maxModel.debias()
    max_time = time.time() - start_time

    with open('time.txt', "a") as file:
        file.write(f'WB time: {max_time}')

    model_file = f"{db_model_path}_MaxEnsemble.pkl"
    with open(model_file, 'wb') as file:
        dill.dump(maxModel, file)



if __name__ == '__main__':
    main()