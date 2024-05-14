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
    label_version, model_type, specialization, policy_name = sys.argv[1:]

    datapath = f"../../data/synthetic"
    xs = np.loadtxt(f"{datapath}/features.csv", delimiter=',')
    if label_version=="linear":
        ys = np.loadtxt(f"{datapath}/linear-labels.csv", delimiter=',')
    elif label_version=="polynomial":
        ys = np.loadtxt(f"{datapath}/poly-labels.csv")
    else:
        print("Need to specify either linear or polynomal as label version in first command-line argument.")
    
    path = f"init-models/{label_version}/{model_type}"
    all_model_files = os.listdir(path)

    if specialization=='coord':
        model_files = [model_file for model_file in all_model_files if ('coord' in model_file) ]
    elif specialization=='group':
        model_files = [model_file for model_file in all_model_files if ('group' in model_file) ]
    else:
        model_files = all_model_files
    
    models = []
    for filename in model_files:
        with open(f"{path}/{filename}", 'rb') as file:
            models.append(dill.load(file))
    
    pred_dim = ys.shape[1]

    if policy_name=='simplex':
        # bit silly syntax, bc wrote code in way that we could theoretically combine different policies,
        # but in practice we never do. 
        policies = [policies.Simplex(pred_dim, models[0])]*len(models) 
    elif policy_name=='variance':
        gran = 0.1
        var_limit = 0.5
        policies = [policies.VarianceConstrained(pred_dim, models[0], gran, var_limit, ys)]*len(models)


if __name__ == '__main__':
    main()