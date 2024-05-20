import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import dill
import os
import itertools
import sys

# below is to deal w unpickling and loading objects
sys.path.append('../../src')
import policies 
import bbDebiasing
import maxEnsembleDebias

plt.style.use('ggplot')

# Number of colors you want
n_colors = 10
# Choose a colormap
colormap = plt.get_cmap('viridis')
# Generate colors from the colormap
colors = [colormap(i) for i in np.linspace(0, 1, n_colors)]
# Update the default color cycle
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def experimentSubtitle(params):
    (label, model, spec, policy, n) = params
    if label=='linear-label':
        labelstr = 'Linear'
    elif label=='poly-label':
        labelstr = 'Polynomial'
    if model=='gb':
        modelstr = 'Gradient Boosting Regressors'
    elif model=='lin':
        modelstr = 'Linear Regression'
    if spec=='group':
        specstr = 'trained on identifiable subgroups of dataset'
    elif spec=='coord':
        specstr = 'trained on individual coordinates of label'
    elif spec=='all':
        specstr = 'all specializations (group/coord)'
    if policy=='variance':
        policystr = 'Variance-constrained'
    return f"Label generation: {labelstr}, Initial Models: {modelstr} {specstr},\n Optimization: {policystr}, n: {n}"

def BBMSE(bbModel, params, figPath=None, experimentName=None):
    mses = np.array([mse(bbModel.train_y, pred) for pred in bbModel.predictions_by_round])
    plt.clf()
    # plt.plot(0,mses[0],'.', label="Initial Model")
    # for i in range(1, bbModel.n_models+1):
    #     indices = list(range(len(bbModel.predictions_by_round)))[i::(bbModel.n_models + 1)]
    #     plt.plot(indices, mses[i::(bbModel.n_models + 1)], '.', label=f"Policy {i}")
    # indices = list(range(len(bbModel.predictions_by_round)))[(bbModel.n_models+1)::(bbModel.n_models + 1)]
    # plt.plot(indices, mses[(bbModel.n_models+1)::(bbModel.n_models + 1)], '.', label="Self-Consistency")
    plt.plot(np.arange(len(mses)), mses)
    plt.suptitle("Black-box Algorithm: MSE over Rounds of Debiasing")
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('MSE')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/bbMSE_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleMSE(maxModel, params, figPath=None, experimentName=None):

    preds = np.array(maxModel.predictions_by_round)
    preds_by_model = [preds[:,i] for i in range(maxModel.n_models)]
    mses_by_model = np.zeros((len(preds_by_model), len(preds)))
    for i in range(maxModel.n_models):
        for j in range(len(preds_by_model[i])):
            mses_by_model[i][j] = mse(maxModel.train_y, preds_by_model[i][j])
    
    plt.clf()
    for i in range(maxModel.n_models):
        plt.plot(np.arange(len(maxModel.predictions_by_round)), mses_by_model[i], label=f"Model {i}")
    plt.legend()
    plt.suptitle("White-Box Algorithm: MSE of Models over Rounds")
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('MSE')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleMSE_{experimentName}.png")
    else:
        plt.show()

def BBpredRev(bbModel, params, figPath=None, experimentName=None):
    # predicted revenue of each point per round
    pred_rev = np.mean(np.sum(np.multiply(bbModel.predictions_by_round, np.array(bbModel.policy_by_round)), axis=2), axis=1)
    # realized revenue
    true_rev = np.mean(np.sum(np.multiply(np.tile(bbModel.train_y, (len(bbModel.policy_by_round),1,1)), np.array(bbModel.policy_by_round)), axis=2), axis=1)

    plt.clf()
    plt.plot(range(len(bbModel.predictions_by_round)), pred_rev, '--', label="Predicted Payoff", color=colors[0])
    plt.plot(range(len(bbModel.predictions_by_round)), true_rev, label="Realized Payoff", color=colors[0])
    plt.hlines(np.mean(np.einsum('ij, ij->i', bbModel.train_y, bbModel.policy.run_given_preds(bbModel.train_y))), 0, bbModel.curr_depth, label="Optimal Payoff (Labels Known)", color=colors[-1])
    plt.legend()
    plt.suptitle("Black-Box Algorithm: Predicted and Realized Payoff over Rounds")
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('Payoff')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/bbRev_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleRev(maxModel, params, figPath=None, experimentName=None):
    pred_rev = np.mean(np.array(maxModel.self_assessed_revs_by_round), axis=2)
    real_rev = np.mean(np.array(maxModel.realized_revs_by_round), axis=2)

    n = len(maxModel.self_assessed_revs_by_round)
    plt.clf()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   

    # for i in range(maxModel.n_models):
    #     plt.plot(np.arange(n), pred_rev[:,i], '--', label=f"Predicted Revenue of Policy {i}")
    for i in range(maxModel.n_models):
        plt.plot(np.arange(n), real_rev[:,i], label=f"Realized Payoff: Policy {i}", color=colors[i])

    # generate the average revenue of the initial policies

    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, n-1, label="Optimal Payoff (Labels Known)", color=colors[-1])
    plt.legend(prop={'size': 6})
    plt.suptitle("White-Box Algorithm: Average Realized Payoff of Component Policies")
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('Payoff')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleRev_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleMetaRev(maxModel, params, figPath=None, experimentName=None):

    real_rev = np.mean(np.array(maxModel.realized_revs_by_round), axis=2)

    meta_real_rev = []
    meta_pred_rev = []
    for i in range(maxModel.curr_depth):
        # there is almost certainly a slicker way to do this indexing that doesn't involve a for loop
        meta_real_rev.append(np.mean(maxModel.realized_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
        meta_pred_rev.append(np.mean(maxModel.self_assessed_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
    
    plt.clf()

    # for i in range(maxModel.n_models):
    #     if i == 0:
    #         plt.hlines(real_rev[0,i], 0, maxModel.curr_depth-1, label=f"Realized Payoffs of Initial Policies", color=colors[0])
    #     else:
    #         plt.hlines(real_rev[0,i], 0, maxModel.curr_depth-1, color=colors[0])

    max_init_model = max([real_rev[0,i] for i in range(maxModel.n_models)])
    plt.hlines(max_init_model, 0, maxModel.curr_depth-1, label=f"Realized Payoff of Best Initial Policy", color=colors[0])

    plt.plot(np.arange(maxModel.curr_depth), meta_real_rev, label=f"Realized Payoff of Meta Policy", color=colors[5])
    plt.plot(np.arange(maxModel.curr_depth), meta_pred_rev, '--', label=f"Predicted Payoff of Meta Policy", color=colors[5])
    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, maxModel.curr_depth-1, label="Optimal Payoff (Labels Known)", color=colors[-1])
    plt.legend(prop={'size': 6})
    plt.suptitle("White-Box Meta Algorithm: Predicted and Realized Payoff", fontsize=12)
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('Payoff')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleMetaRev_{experimentName}.png")
    else:
        plt.show()

def BBvsMaxEnsemble(bbModel, maxModel, params, figPath=None, experimentName=None):
    BBpred_rev = np.mean(np.sum(np.multiply(bbModel.predictions_by_round, np.array(bbModel.policy_by_round)), axis=2), axis=1)
    BBreal_rev = np.mean(np.sum(np.multiply(np.tile(bbModel.train_y, (len(bbModel.policy_by_round),1,1)), np.array(bbModel.policy_by_round)), axis=2), axis=1)
    meta_real_rev = []
    meta_pred_rev = []
    for i in range(maxModel.curr_depth):
        meta_real_rev.append(np.mean(maxModel.realized_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
        meta_pred_rev.append(np.mean(maxModel.self_assessed_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
    
    plt.clf()
    plt.plot(np.arange(maxModel.curr_depth), meta_pred_rev, '--', label=f"Predicted Payoff of White-Box Alg", color=colors[0])
    plt.plot(np.arange(len(BBpred_rev[:-1])), BBpred_rev[:-1], '--', label="Predicted Payoff of Black-Box Alg", color=colors[5])
    plt.plot(np.arange(maxModel.curr_depth), meta_real_rev, label=f"Realized Payoff of White-Box Alg", color=colors[0])
    plt.plot(np.arange(len(BBreal_rev[:-1])), BBreal_rev[:-1], label="Realized Payoff of Black-Box Policy", color=colors[5])
    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, maxModel.curr_depth-1, label="Optimal Payoff (Labels Known)", color=colors[-1])
    plt.legend(prop={'size': 6})
    plt.suptitle("Comparison of Payoffs of White-Box and Black Box Algorithms", fontsize=12)
    plt.title(experimentSubtitle(params), fontsize=7)
    plt.ylabel('Payoff')
    plt.xlabel('Round of Debiasing')
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsemblevsBBRev_{experimentName}.png")
    else:
        plt.show()

def allPlots(bbPath, maxEnsemblePath, figPath, experimentName, params):
    with open(bbPath, 'rb') as file:
        bbModel = dill.load(file)
    with open(maxEnsemblePath, 'rb') as file:
        maxModel = dill.load(file)
    
    if not os.path.exists(figPath):
        if not os.path.exists('./fig'):
            os.makedirs('fig')
        os.makedirs(figPath)
    
    BBMSE(bbModel, params, figPath=figPath, experimentName=experimentName)
    MaxEnsembleMSE(maxModel, params, figPath=figPath, experimentName=experimentName)
    BBpredRev(bbModel, params, figPath=figPath, experimentName=experimentName)
    MaxEnsembleRev(maxModel, params, figPath=figPath, experimentName=experimentName)
    MaxEnsembleMetaRev(maxModel, params, figPath=figPath, experimentName=experimentName)
    BBvsMaxEnsemble(bbModel, maxModel, params, figPath=figPath, experimentName=experimentName)

def main():
    label_types = ['poly-label', 'linear-label']
    model_types = ['lin', 'gb']
    spec_types = ['group', 'coord', 'all']
    policy_types = ['variance']

    for (label, model, spec, policy) in itertools.product(label_types, model_types, spec_types, policy_types):
        if (spec=='group') and (label=='linear-label') and (model=='gb'):
            nrounds=5000
            n=400
        else:
            nrounds=1000
            n=250
        
        bbPath = f"debiased-models/{label}_{model}_{spec}_{policy}_{nrounds}_subsample{n}_BBModel.pkl"
        maxEnsemblePath = f"debiased-models/{label}_{model}_{spec}_{policy}_{nrounds}_subsample{n}_MaxEnsemble.pkl"
        figPath = f"figs/{label}_{model}_{spec}_{policy}"
        experimentName=f"{label}_{model}_{spec}_{policy}"
        
        allPlots(bbPath, maxEnsemblePath, figPath=figPath, experimentName=experimentName, params = (label, model, spec, policy, n))

if __name__ == "__main__":
    main()