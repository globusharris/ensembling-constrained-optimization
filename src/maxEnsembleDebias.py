import numpy as np

class EnsembledModel:

    """
    "Max ensemble" version of debiasing. 
    
    Given a series of k initial models and their associated policies, it iteratively debiases all of them,
    conditioning on which policy induced by the models is maximal and a coord x val pair. 
    """

    def __init__(self, init_models, policies, train_x, train_y, max_depth, tolerance):
        """
        init_models: An array of k different models for the prediction task. 
        policies: An array of k policies which are induced by those models. These should be policy objects.
        train_x: training data features
        train_y: training data labels
        max_depth: maximal depth of the debiasing process. 
        tolerance: tolerance parameter for stopping debiasing

        Notes:
        - Running under the assumption that each of the policies takes the same possible values, and that they are the same
          for every coordinate. 
        """

        self.init_models = init_models 
        self.policies = policies 
        self.train_x = train_x
        self.train_y = train_y
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.pred_dim = len(train_y[0]) # prediction dimension
        self.n = len(train_x)

        
        self.n_models = len(self.init_models)   # number of models
        self.n_values = self.policies[0].n_vals # number of values policy can take
        self.policy_vals = self.policies[0].coordinate_values # list of possible values any coordinate of policy can take
        self.gran = self.policies[0].gran # assuming all policies have same granularity
        self.n_conditions = self.n_models * self.n_values * self.pred_dim # total number of conditioning events 
        
        self.curr_depth = 0
        self.debias_conditions = [] # Shape self.curr_depth*3. Row i is (model_index, coordinate, value) tuple for conditining events, where model_index is index of model with which bias is measured wrt in that round.
        self.curr_preds = [model(self.train_x) for model in self.init_models] # length k; entry i are most up-to-date (most debiased) predictions of model i on the training data
        self.predictions_by_round = [np.copy(self.curr_preds)] # shape self.curr_depth * k + 1; entry (i,j) is predictions on training data of model j after i rounds of debiasing
        self.bias_array = [] # shape self.curr_depth * k; entry (i,j) is bias of target model (described in row i of debias_conditions) when policy j is maximal and policy i has target val at coord
        self.probabilities = [] # shape self.curr_depth * k; entry (i,j) is the empirical weight of the conditioning event at round i of debiasing when model j is maximal and the conditions described in row i of debias conditions are met. 
        self.halting_cond = 0

        self.policies_by_round = []
        self.self_assessed_revs_by_round = []
        self.realized_revs_by_round = []
        self.max_policy_index_by_round = []
        self.meta_policy_choice_by_round = []
        self.meta_model_pred_by_round = []

    def debias(self):
        """
        Debiases all k of the initial models.

        At each round of the debiasing, a model to debias from the k models is picked, called model_index,
        along with a coordinate and value to predicate the conditioning. Then, for *each* of the k models, 
        call  that model j, the bias of the model_index model is determined on the subset of points for which 
        the model_index model has value "val" at coordinate "coord" *and* such that policy induced by model j 
        is the policy with highest self-assessed revenue.  

        Currently, this terminates when the maximum depth is reached, and there is no additional stopping criterion
        (e.g. not checking for approximate unbiasedness.)

        """
        for i in range(self.max_depth):
            
            # Get the conditions to debias with respect to in this round, and store.
            model_index = self.curr_depth % self.n_models 
            coord = (self.curr_depth//(self.n_values* self.n_models)) % self.pred_dim 
            val = self.policy_vals[(self.curr_depth // self.n_models) % self.n_values]
            self.debias_conditions.append([model_index, coord, val])

            # Evaluate models, determine the policy associated w each and that policy's self-assessed revenue
            policies_by_models = np.array([self.policies[i].run_given_preds(self.curr_preds[i]) for i in range(self.n_models)])     
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', self.curr_preds[i], policies_by_models[i]) for i in range(self.n_models)]) # array of length k, where each entry is of shape n, and is dot product of pred and policy vector
            realized_revs = np.array([np.einsum('ij,ij->i', self.train_y, policies_by_models[i]) for i in range(self.n_models)])
            maximal_policy = np.argmax(self_assessed_revs, axis=0) # length n; returns index of the maximal policy
            curr_policy = policies_by_models[model_index]

            # Track what the meta policy would choose at this round and what all policies have been doing
            self.policies_by_round.append(policies_by_models)
            self.self_assessed_revs_by_round.append(self_assessed_revs)
            self.realized_revs_by_round.append(realized_revs)
            self.max_policy_index_by_round.append(maximal_policy)
            self.meta_policy_choice_by_round.append(policies_by_models[maximal_policy, np.arange(policies_by_models.shape[1]),:])
            self.meta_model_pred_by_round.append(np.array(self.curr_preds)[maximal_policy, np.arange(self.n),:])
           
            # Debias model
            bias_this_round= [] # will be k-dimensional, jth entry is bias when jth policy is maximal
            probs = [] # k-dimensional, jth entry is empirical probability of event where jth policy is maximal
            for j in range(self.n_models):
                flag = (curr_policy[:,coord] >= val) & (curr_policy[:,coord] < val + self.gran) & (maximal_policy == j)
                probs.append(sum(flag)/len(flag)) # empirical probability of being in the target conditioning event
                if sum(flag)!=0:
                    bias = np.mean(self.curr_preds[model_index][flag] - self.train_y[flag], axis=0)
                    bias_this_round.append(bias)
                    self.curr_preds[model_index][flag] -= bias
                else:
                    # just store 0 if bucket is empty
                    bias_this_round.append(np.zeros(self.pred_dim))

            self.bias_array.append(np.array(bias_this_round))
            self.predictions_by_round.append(np.copy(self.curr_preds))
            self.probabilities.append(probs)

            if self._halt():
                print("Hit tolerance; halting debiasing.")
                break

            self.curr_depth += 1 
        
        if self.curr_depth == self.max_depth:
            print("Maximum depth reached.")

        return None 
    
    def _halt(self):
        """
        Stopping condition for debiasing. Verifies if the bias of each of the k models is within tolerance over all
        of the possible coordinate * value pairs. 
        """   
        
        max_bias = [max(self.bias_array[self.curr_depth][i]) for i in range(self.n_models)] # max bias term for each of the k debiasing steps
        violations = np.array([self.probabilities[self.curr_depth][i] * max_bias[i] for i in range(self.n_models)]) # max bias terms weighted by empirical probability of event
        
        if np.all(violations < self.tolerance):  # if all violations are less than the tolerance condition
            self.halting_cond += 1          # add to counter for halting 
        else:
            self.halting_cond = 0           # otherwise, reset counter to 0
        
        if self.halting_cond == self.n_conditions:      # if last n_conditions rounds didn't have sufficiently large violation, stop.
            return True
        
        return False
    
    def predict(self, xs):
        """
        Use debiased predictors to get predictions on x. 

        Outputs:
        curr_preds: A k-dimensional vector, where entry i is all of the predictions on the input xs of debiased model i.
        transcript: self.curr_depth * k dimensional matrix, where entry (i,j) is the predictions of model j after i rounds of debiasing. 
        """
        
        n = len(xs)

        # evaluate all k initial models on the data
        curr_preds = [model(xs) for model in self.init_models]

        class Transcript:
            def __init__(self):
                self.preds = []
                self.policies_by_models_by_round = []
                self.max_policy_index_by_round = []
                self.meta_policy_choice_by_round = []
                self.meta_model_pred_by_round = []
        
        transcript = Transcript()
        transcript.preds.append(np.copy(curr_preds))

        for i in range(self.curr_depth):
            # determine what the conditioning event for this round was
            model_index, coord, val = self.debias_conditions[i]
            # determine policy and policy's self assessed revenue for each model
            policies_by_models = np.array([self.policies[i].run_given_preds(curr_preds[i]) for i in range(self.n_models)])      
            self_assessed_revs =  np.array([np.einsum('ij,ij->i', curr_preds[i], policies_by_models[i]) for i in range(self.n_models)])
            maximal_policy = np.argmax(self_assessed_revs, axis=0)

            transcript.policies_by_models_by_round.append(policies_by_models)
            transcript.max_policy_index_by_round.append(maximal_policy)
            transcript.meta_policy_choice_by_round.append(policies_by_models[maximal_policy, np.arange(policies_by_models.shape[1]),:])
            transcript.meta_model_pred_by_round.append(np.array(curr_preds)[maximal_policy, np.arange(n)])

            # now, update the predictions according to each of the k debiasing events for that round
            curr_policy = policies_by_models[model_index]
            for k in range(self.n_models):
                flag = (curr_policy[:,coord] >= val) & (curr_policy[:,coord] < val + self.gran) & (maximal_policy == k)
                curr_preds[model_index][flag] -= self.bias_array[i][k] 
            
            transcript.preds.append(np.copy(curr_preds))

        return np.array(curr_preds), transcript

    def getPredPayoff(self, transcript):
        """
        Given transcript of predictions, return payoff of final ensemble
        """
        return np.sum(np.multiply(transcript.meta_model_pred_by_round[-1], transcript.meta_policy_choice_by_round[-1]), axis=1)
    
    def getRealPayoff(self, transcript, ys):
        return np.sum(np.multiply(ys, transcript.meta_policy_choice_by_round[-1]), axis=1)
        
