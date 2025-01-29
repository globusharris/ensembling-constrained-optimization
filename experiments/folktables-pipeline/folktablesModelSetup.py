import numpy as np
import pandas as pd
import acsData

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class State:
    def __init__(self, name, acs_year):
        self.name = name
        self.features, self.targets = acsData.get_data(acs_year, [self.name])
        
        target_transform=lambda x: x > 50000
        self.bool_targets = target_transform(self.targets)

        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(
            self.features, self.bool_targets, test_size=0.3, random_state=42)
    
        self.model = make_pipeline(StandardScaler(), LogisticRegression())
        self.model_preds_train = None
        self.model_preds_test = None
        
        self.meta_model = make_pipeline(StandardScaler(), LogisticRegression())
        self.meta_model_targets_train = None
        self.meta_model_targets_test = None
        self.meta_model_preds_train = None
        self.meta_model_preds_test = None
    
    def train_state_model(self):
        self.model.fit(self.features_train, self.target_train) 
        self.model_preds_train = pd.Series(self.model.predict(self.features_train), index=self.features_train.index)
        self.model_preds_test = pd.Series(self.model.predict(self.features_test), index=self.features_test.index)

    def train_state_meta_model(self):
        """
        Train meta-model which tries to predict when the original model makes mistakes.
        """
        self.meta_model_targets_train = (self.model_preds_train==self.target_train)
        self.meta_model_targets_test = (self.model_preds_test==self.target_test)
        self.meta_model.fit(self.features_train, self.meta_model_targets_train)
        self.meta_model_preds_train = pd.Series(self.meta_model.predict_proba(self.features_train)[:,1], index=self.features_train.index)
        self.meta_model_preds_test = pd.Series(self.meta_model.predict_proba(self.features_test)[:,1], index=self.features_test.index)

class Ensemble:
    def __init__(self, submodels):
        self.n_predictors = len(submodels) #k; number of models being ensembled
        self.submodels = submodels

        # TODO: might make sense to have the data used for debiasing be separate from the training data.

        # TODO: should probably combine datasets rather than just using single state's data
        self.features_train = self.submodels[0].features_train
        self.features_test = self.submodels[0].features_test
        self.target_train = self.submodels[0].target_train
        self.target_test = self.submodels[0].target_test
        
        self.preds_train = self.predict_component_models(self.features_train) # shape n x k; vals in {0,1}
        self.meta_preds_train = self.meta_predictor(self.features_train)    # shape n x k; vals in [0,1]
        self.meta_targets_train = (self.preds_train == np.expand_dims(self.target_train, axis=-1)) #n x k; vals in {0,1}
        
    def meta_predictor(self, features):
        """
        Shape n x k where coordinate i is the meta predictions of model i
        """
        n_samples = len(features)
        meta_preds = np.zeros((n_samples, self.n_predictors))
        for i,submodel in enumerate(self.submodels):
            meta_preds[np.arange(n_samples), i] = submodel.meta_model.predict_proba(features)[:,1]
        return meta_preds
    
    def predict_component_models(self, features):
        """
        Shape n x k where coordinate i is the prediction of model i 
        """
        n_samples = len(features)
        preds = np.zeros((n_samples, self.n_predictors))
        for i, submodel in enumerate(self.submodels):
            preds[np.arange(n_samples), i] = submodel.model.predict(features)
        return preds
    
    def predict(self, features):
        """
        The final prediction of the ensemble of all the models. 
        Takes predictions of the component models' confidence and picks the prediction of the model 
        with highest confidence. 
        """
        n_samples = len(features)
        preds = np.zeros((n_samples))
        
        meta_preds = self.meta_predictor(features) # get confidences of each model; shape n x k
        max_model_indices = np.argmax(meta_preds, axis=1) # determine which model was maximal
        for i, submodel in enumerate(self.submodels):
            preds[max_model_indices==i] = submodel.model.predict(features[max_model_indices==i])
        return preds

def runSetup(acs_year, state_names):
    state_list=[]
    for state_name in state_names:
        new_state = State(state_name, acs_year)
        new_state.train_state_model()
        new_state.train_state_meta_model()
        state_list.append(new_state)
    return state_list