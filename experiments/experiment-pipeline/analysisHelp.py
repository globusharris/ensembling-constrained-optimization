import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import dill

def BBMSE(bbModel, figPath=None, experimentName=None):
    mses = np.array([mse(bbModel.train_y, pred) for pred in bbModel.predictions_by_round])
    plt.clf()
    plt.plot(0,mses[0],'.', label="Initial Model")
    for i in range(1, bbModel.n_models+1):
        indices = list(range(len(bbModel.predictions_by_round)))[i::(bbModel.n_models + 1)]
        plt.plot(indices, mses[i::(bbModel.n_models + 1)], '.', label=f"Policy {i}")
    indices = list(range(len(bbModel.predictions_by_round)))[(bbModel.n_models+1)::(bbModel.n_models + 1)]
    plt.plot(indices, mses[(bbModel.n_models+1)::(bbModel.n_models + 1)], '.', label="Self-Consistency")
    plt.legend()
    plt.title("BB Alg: MSE over Rounds, by Policy")
    if figPath:
        plt.savefig(f"{figPath}/bbMSE_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleMSE(maxModel, figPath=None, experimentName=None):
    
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
    plt.title("Max Ensemble Alg: MSE of Models over Rounds")
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleMSE_{experimentName}.png")
    else:
        plt.show()

def BBpredRev(bbModel, figPath=None, experimentName=None):
    # predicted revenue of each point per round
    pred_rev = np.mean(np.sum(np.multiply(bbModel.predictions_by_round, np.array(bbModel.policy_by_round)), axis=2), axis=1)
    # realized revenue
    true_rev = np.mean(np.sum(np.multiply(np.tile(bbModel.train_y, (len(bbModel.policy_by_round),1,1)), np.array(bbModel.policy_by_round)), axis=2), axis=1)

    plt.clf()
    plt.plot(range(len(bbModel.predictions_by_round)), pred_rev, 'o', label="predicted mean rev of policy")
    plt.plot(range(len(bbModel.predictions_by_round)), true_rev, 'o', label="realized mean rev of policy")
    plt.hlines(np.mean(np.einsum('ij, ij->i', bbModel.train_y, bbModel.policy.run_given_preds(bbModel.train_y))), 0, bbModel.curr_depth, label="Best Possible Revenue", color='pink')
    plt.legend()
    plt.title("BB Alg: Predicted and Realized Revenue over Rounds")
    if figPath:
        plt.savefig(f"{figPath}/bbRev_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleRev(maxModel, figPath=None, experimentName=None):
    pred_rev = np.mean(np.array(maxModel.self_assessed_revs_by_round), axis=2)
    real_rev = np.mean(np.array(maxModel.realized_revs_by_round), axis=2)

    n = len(maxModel.self_assessed_revs_by_round)
    plt.clf()
    for i in range(maxModel.n_models):
        plt.plot(np.arange(n), pred_rev[:,i], '--', label=f"Predicted Revenue of Policy {i}")
    for i in range(maxModel.n_models):
        plt.plot(np.arange(n), real_rev[:,i], label=f"Realized Revenue of Policy {i}")

    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, n-1, label="Best Possible Revenue", color='pink')
    plt.legend(prop={'size': 6})
    plt.title("Max Ensemble Alg: Average Predicted and Realized Revenue of Policies")
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleRev_{experimentName}.png")
    else:
        plt.show()

def MaxEnsembleMetaRev(maxModel, figPath=None, experimentName=None):
    meta_real_rev = []
    meta_pred_rev = []
    for i in range(maxModel.curr_depth):
        # there is almost certainly a slicker way to do this indexing that doesn't involve a for loop
        meta_real_rev.append(np.mean(maxModel.realized_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
        meta_pred_rev.append(np.mean(maxModel.self_assessed_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
    
    plt.clf()
    plt.plot(np.arange(maxModel.curr_depth), meta_real_rev, label=f"Realized Revenue of Meta Policy")
    plt.plot(np.arange(maxModel.curr_depth), meta_pred_rev, label=f"Predicted Revenue of Meta Policy")
    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, maxModel.curr_depth-1, label="Best Possible Revenue", color='pink')
    plt.legend(prop={'size': 6})
    plt.title("Max Ensemble Alg: Average Predicted and Realized Revenue of Meta Ensemble Policy")
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsembleMetaRev_{experimentName}.png")
    else:
        plt.show()

def BBvsMaxEnsemble(bbModel, maxModel, figPath=None, experimentName=None):
    BBpred_rev = np.mean(np.sum(np.multiply(bbModel.predictions_by_round, np.array(bbModel.policy_by_round)), axis=2), axis=1)
    BBreal_rev = np.mean(np.sum(np.multiply(np.tile(bbModel.train_y, (len(bbModel.policy_by_round),1,1)), np.array(bbModel.policy_by_round)), axis=2), axis=1)
    meta_real_rev = []
    meta_pred_rev = []
    for i in range(maxModel.curr_depth):
        meta_real_rev.append(np.mean(maxModel.realized_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
        meta_pred_rev.append(np.mean(maxModel.self_assessed_revs_by_round[i][maxModel.max_policy_index_by_round[i], np.arange(maxModel.n)]))
    
    plt.clf()
    plt.plot(np.arange(maxModel.curr_depth), meta_pred_rev, label=f"Predicted Revenue of Policy")
    plt.plot(np.arange(bbModel.curr_depth), BBpred_rev[:-1], label="Predicted Revenue of BB Policy")
    plt.plot(np.arange(maxModel.curr_depth), meta_real_rev, label=f"Realized Revenue of Policy")
    plt.plot(np.arange(bbModel.curr_depth), BBreal_rev[:-1], label="Realized Revenue of BB Policy")
    plt.hlines(np.mean(np.einsum('ij, ij->i', maxModel.train_y, maxModel.policies[0].run_given_preds(maxModel.train_y))), 0, maxModel.curr_depth-1, label="Best Possible Revenue", color='pink')
    plt.legend(prop={'size': 6})
    plt.title("Predicted Revenue of Ensemble Policies")
    if figPath:
        plt.savefig(f"{figPath}/MaxEnsemblevsBBRev_{experimentName}.png")
    else:
        plt.show()

def allPlots(bbPath, maxEnsemblePath, figPath, experimentName):
    with open(bbPath, 'rb') as file:
        bbModel = dill.load(file)
    with open(maxEnsemblePath, 'rb') as file:
        maxModel = dill.load(file)
    
    BBMSE(bbModel, figPath=figPath, experimentName=experimentName)
    MaxEnsembleMSE(maxModel, figPath=figPath, experimentName=experimentName)
    BBpredRev(bbModel, figPath=figPath, experimentName=experimentName)
    MaxEnsembleRev(maxModel, figPath=figPath, experimentName=experimentName)
    MaxEnsembleMetaRev(maxModel, figPath=figPath, experimentName=experimentName)
    BBvsMaxEnsemble(bbModel, maxModel, figPath=figPath, experimentName=experimentName)