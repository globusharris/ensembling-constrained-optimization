import dill
import numpy as np
import sys

import dataGeneration
import modelSetup

sys.path.append('../../src')
import policies 
import bbDebiasing
import maxEnsembleDebias

rng = np.random.default_rng(seed=42)

# Feature parameters
n = 11000
n_features = 20
cov_min = -3
cov_max = 3
mean_min = -5
mean_max = 5
num_categories = 5

xs = dataGeneration.feature_gen(n, n_features, cov_min, cov_max, mean_min, mean_max, num_categories)

# Label parameters
label_dim = 4
n_terms = 5
term_size = 2
coeff_min = -1
coeff_max = 1
max_exponent = 2
noise = 0.01

ys = dataGeneration.linear_label_gen(xs, label_dim, noise)

train_xs = xs[0:10000]
train_ys = ys[0:10000]
holdout_xs = xs[10000:10010]
holdout_ys = ys[10000:10010]

pathA = ['linear-label_gb_coord_variance_5000_subsample400_BBModel.pkl',
'linear-label_gb_coord_variance_5000_subsample400_MaxEnsemble.pkl']
pathB = ['linear-label_gb_group_variance_5000_subsample400_BBModel.pkl',
'linear-label_gb_group_variance_5000_subsample400_MaxEnsemble.pkl']
pathC = ['linear-label_gb_coord_linear-constraint_5000_subsample400_BBModel.pkl',
'linear-label_gb_coord_linear-constraint_5000_subsample400_MaxEnsemble.pkl']
pathD = ['linear-label_gb_group_linear-constraint_8000_subsample400_BBModel.pkl',
'linear-label_gb_group_linear-constraint_8000_subsample400_MaxEnsemble.pkl']

pathsets = [pathA, pathB, pathC, pathD]
experimentNames = ["A","B","C","D"]

for i in range(len(pathsets)):

    bbPath = f'debiased-models/{pathsets[i][0]}'
    wbPath = f'debiased-models/{pathsets[i][1]}'
    with open(bbPath, 'rb') as file:
        bbModel = dill.load(file)
    with open(wbPath, 'rb') as file:
        wbModel = dill.load(file)
    
    # regenerating models to deal with a silly bug
    if i==1 or i==3:
        groups = np.unique(xs[:,-1])
        models = []
        gb_model_params = {'learning_rate': 0.1, 'max_depth':6, 'random_state':42}
        for group in groups:
            model = model.meta_model_by_group(xs, ys, group, 'gradient-boost', gb_model_params)
            models.append(model)
        bbModel.models = models
        wbModel.init_models = models

    bbPreds = bbModel.predict(holdout_xs)[0]
    np.save(f'out-of-sample/experiment-{experimentNames[i]}-BBpreds.npy', bbPreds)

    wbPreds, transcript = wbModel.predict(holdout_xs)
    np.save(f'out-of-sample/experiment-{experimentNames[i]}-WBpreds.npy', wbPreds)
    np.save(f'out-of-sample/experiment-{experimentNames[i]}-WBtrans.npy', np.array(transcript.policies_by_models_by_round))

