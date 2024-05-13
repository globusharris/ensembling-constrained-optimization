import numpy as np

class bbDebias:
    """
    "Bias Bounty" style method of iteratively debiasing a predictor with respect to a series of policies
    """
    
    def __init__(self, init_model, policy, train_x, train_y, max_depth, tolerance):
        """
        init_models: Single initial model, which debiasing will be done with respect to. 
        policies: A single policy induced by the initial model. Should be a policy object. 
        train_x: training data features of length n
        train_y: training data labels of length n
        max_depth: maximal depth of the debiasing process. 
        tolerance: tolerance parameter for stopping debiasing

        Notes:
        - Assuming that each of the policies takes the same possible values as the initializing policy, 
          and that they are the same for every coordinate. 
        """

        self.init_model = init_model
        self.policy = policy
        self.train_x = train_x 
        self.train_y = train_y
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.pred_dim = len(train_y[0]) #prediction dimension

        self.models = [] # list of all k models used for debiasing
        self.policies = [] # list of all k policies induced by models used for debiasing

        self.n_models = 0
        self.n_values = self.policy.n_vals
        self.policy_vals = self.policy.coordinate_values
        self.gran = self.policy.gran
        self.n_conditions = 0

        self.curr_depth = 0
        self.debias_conditions = [] # shape curr_depth * 3; row i is (model/policy index, coordinate, value) used for that round of debiasing
        self.bias_array = [] # shape curr_depth * pred dim; row i is bias vector for the conditioning event described in row i of self.debias_conditions
        self.curr_preds = self.init_model(train_x) # current predictions of debiased model
        self.predictions_by_round = [np.copy(self.curr_preds)] # length self.curr_depth, entry i is predictions of model on ith round of debiasing on training data
        self.policy_by_round = [self.policy.run_given_preds(self.curr_preds)] # self.curr_depth * len(train_x) * len(train_y)
        self.policy_preds = [] # shape k*n*pred_dim; list of induced policies of all k models on the training data
        self.probabilities = [] # length curr_depth; entry i is empirical weight of the conditioning event of round i of debiasing 
        self.halting_cond = 0
    
    def debias(self, models, policies):
        """
        Debias model according to collection of models and policies. Additionally, enforce self-consistency of final model. 

        models: list of k models to debias with respect to.
        policies: list of k policies associated with each of the k models.
        """

        # Add models and policies to global list, and update number of conditioning events for debiasing. 
        self.models.extend(models)
        self.policies.extend(policies)
        self.n_models = len(self.models) 
        self.n_conditions = (self.n_models+1) * self.n_values * self.pred_dim #+1 to deal w 

        # Evaluate models and policies induced by models
        preds = [model(self.train_x) for model in models]
        pols = [policies[i].run_given_preds(preds[i]) for i in range(len(models))]
        self.policy_preds.extend(pols)

        for i in range(self.max_depth):
            # Get the conditions to debias with respect to in this round, and store.
            model_index = self.curr_depth % (self.n_models + 1)
            coord = (self.curr_depth//(self.n_values*(self.n_models+1))) % self.pred_dim 
            val = self.policy_vals[(self.curr_depth // (self.n_models+1)) % self.n_values]
            self.debias_conditions.append([model_index, coord, val])

            # Get the policy used for debiasing. 
            if model_index >= self.n_models: # if running self-consistency check
                curr_policy = self.policy.run_given_preds(self.curr_preds)
            else: # otherwise
                curr_policy = self.policy_preds[model_index]
            
            # Calculate bias for this event
            flag = (curr_policy[:,coord] >= val) & (curr_policy[:,coord] < (val + self.gran))
            self.probabilities.append(sum(flag)/len(flag))
            if sum(flag)!=0:
                bias = np.mean(self.curr_preds[flag] - self.train_y[flag], axis=0)
                self.bias_array.append(bias)
                self.curr_preds[flag] -= bias
            else:
                self.bias_array.append(np.zeros(self.pred_dim))   
            
            self.predictions_by_round.append(np.copy(self.curr_preds))
            self.policy_by_round.append(self.policy.run_given_preds(self.curr_preds))
            
            if self._halt():
                print("Hit tolerance; halting debiasing.")
                break

            self.curr_depth +=1
        
        if self.curr_depth == self.max_depth:
            print("Maximum depth reached.")
        
        return None
    
    def _halt(self):
        """
        Stopping condition for debiasing. Verifies if the last n_conditions rounds had sufficiently small bias to halt early.
        """
        violation = self.probabilities[self.curr_depth] * max(self.bias_array[self.curr_depth])
        if violation < self.tolerance:
            self.halting_cond += 1
        else: 
            self.halting_cond = 0

        if self.halting_cond == self.n_conditions:
            return True
        
        return False 

    def predict(self, xs):
        """ 
        Run final debiased predictor on xs.
        """
        
        curr_preds = self.init_model(xs)
        model_preds = [model(xs) for model in self.models] #predictions of the models we're debiasing with respect to
        policy_preds = [self.policies[i].run_given_preds(model_preds[i]) for i in range(self.n_models)]
        
        transcript = []
        for i in range(self.curr_depth):
            model_index, coord, val = self.debias_conditions[i]

            # Get the policy used for debiasing. 
            if model_index >= self.n_models: # if running self-consistency check
                curr_policy = self.policy.run_given_preds(curr_preds)
            else: # otherwise
                curr_policy = policy_preds[model_index]
            
            flag = (curr_policy[:,coord] >= val) & (curr_policy[:,coord] < val + self.gran)
            curr_preds[flag] -= self.bias_array[i]
            transcript.append(np.copy(curr_preds))

        return curr_preds, transcript