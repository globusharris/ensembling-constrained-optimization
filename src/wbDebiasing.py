import numpy as np
import itertools

#Non round-robin version of debiasing

class wbModel:
    def __init__(self, policies, train_ys, preds_by_models, tolerance):
        self.policies = policies    # list of k policy objects
        self.train_ys = train_ys    # shape n x d
        self.preds_by_models = np.copy(preds_by_models) # shape k x n x d; predictions of each of the models
        self.tolerance = tolerance

        self.n_policies = len(policies)
        self.n_models = self.n_policies
        self.n_samples = len(train_ys)
        self.tolerance = tolerance

         # Note: as currently implemented, assumes all policies have same shape/form. This could be generalized by editing the below code.
        self.n_coords = policies[0].dim
        self.n_bins = policies[0].n_vals
        self.coordinate_values = policies[0].coordinate_values
        self.gran = policies[0].gran

        self.masks = None # shape k x m x d x k x n; masks of level sets to debias wrt for each of the k models 

        # Bookkeeping for out-of-sample computation
        self.targets_by_round = []
        self.bias_by_round = []
        self.recompute_masks = [] # the level sets are only re-computed in the debias algorithm and not the update step, so have to track these indices. 

        # Bookkeeping for mses over rounds (for analysis/fun)
        self.mses_by_round = []

        #self.preds_by_rounds = []
        self.t = 0
        self.masks_by_round = []
        self.policies_by_round = []

    def mse_per_model(self):
        return np.mean((self.preds_by_models - self.train_ys)**2,axis=-2)
    
    def _generate_model_ls_masks(self, policy_outputs):
        """
        Should be usable both in and out of sample, hence passing predictions in as input. 
        Generates the d x m level sets of each of the k input models, where d is the dimension of the models' predictions and m is the number of LS per coordinate. 
        Each level set is expressed as a length-n Boolean mask; where if the mask is true at index i it means that datapoint i is a member of the level set. 
        
        policy_outputs: shape k x n x d
        output masks: shape k x d x m x n; masks[k,d,m] is a length-n Boolean mask of the level set m of 
        coordinate d of the predictions of model k
        """
        n_samples = policy_outputs.shape[1]
        masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, n_samples))
        for (k,d,m) in itertools.product(range(self.n_policies), range(self.n_coords), range(self.n_bins)):
            val = self.coordinate_values[m]
            if m < self.n_bins - 1:
                mask = (policy_outputs[k,:,d]>=val) & (policy_outputs[k,:,d]<val+self.gran)
            else: # special case for last bin to deal with edges
                mask = (policy_outputs[k,:,m]>=val) & (policy_outputs[k,:,d]<=val+self.gran)
            masks[k,d,m] = mask
        return masks

    def _generate_maximal_model_masks(self, preds_by_models, policy_outputs):
        """
        Should be usable both in and out of sample!
        preds_by_models: k x n x d
        policy_outputs: k x n x d
        output masks: shape k x n; masks[k] is a length-n Boolean mask of the collection of training 
        points which have model k's induced policy predicted to be maximal.
        """
        n_samples = preds_by_models.shape[1]
        dot_products = np.vecdot(preds_by_models, policy_outputs) # shape k x n of dot products of policy of model i and model i's predictions per point.
        max_models = np.argmax(dot_products, axis=0) # shape n of maximal model per training sample
        masks = np.zeros((self.n_policies, n_samples))
        masks[max_models, np.arange(n_samples)] = True
        return masks
    
    def _generate_masks(self, preds_by_models, policy_outputs):
        max_model_masks = self._generate_maximal_model_masks(preds_by_models, policy_outputs) # shape k x d x m x n
        model_ls_masks = self._generate_model_ls_masks(preds_by_models) # shape k x n 
        # TODO: figure out how Prathamesh's version, which is more efficient, works
        masks = np.repeat(np.expand_dims(model_ls_masks,-2),2,axis=3)*max_model_masks 
        return masks


    def _calculate_bias(self):
        """
        For all c in C (as defined by self.masks), calculates E[y - h(x)|x in c]
        output: (bias, probs)
        bias shape: k x d x m x k x d where each k x d x m x k slice describes a different level set's bias in all d coordinates
        probs shape: k x d x m x k where each k x d x m x k slice is the density of that level set
        """
        # converting masks to floats and expanding dimension so can broadcast
        masks = self.masks.astype(np.float32)
        masks = np.expand_dims(masks, axis=-1)
        diffs = self.train_ys*masks - np.expand_dims(self.preds_by_models, axis=(1,2,3))*masks
        sums = np.sum(diffs, axis=-2)
        ns = np.sum(masks, axis=-2)
        bias = np.where(ns>0, sums/ns.clip(min=1),0) # clip is to avoid div by 0 errors
        probs = np.squeeze(ns/self.n_samples)
        return bias, probs

    def _find_maximum_bias(self, bias, probs):
        """
        Outputs:
        max_weighted_bias: Maximum value of weighted bias of all level sets across all models, 
        where weighting is in terms of the density of the level set on the training data. Scalar quantity, np.float64
        
        target_set: Index of the maximal level set. Tuple of length 4 of (target model index, target model coordinate, target coordinate level set, target maximal model level set). 
        I.e. the first coordinate indexes which models' level set corresponds to the maximal bias and the final three correspond the the index of that level set for that particular model.

        target_bias: Bias vector for the target set. This is a numpy array of shape (d,), corresponding to the bias in each of the d coordinates of the target level set. 
        """
        l_infinity = np.max(np.abs(bias), axis=-1) # shape k x d x m x k
        weighted_bias = probs*l_infinity
        max_weighted_bias = weighted_bias.max()
        target_set = tuple(np.argwhere(weighted_bias==max_weighted_bias)[0]) #converting into tuple for indexing
        target_bias = bias[target_set]
        return max_weighted_bias, target_set, target_bias
    
    def _update(self):
        """
        In Algorithm 2 in the paper, update is called within the while loop for *each* of the k models. 
        It is unspecified how (e.g. the order) of the update steps. E.g. could first run update on model 1, then 2, 
        or otherwise.
        Here, we update them all *simultaneously*. I.e., we find the maximally biased level sets over every model, then 
        update that particular model, and continue. 
        Note that there is a shape difference in  and the bias arrays between this algorithm and the version
        presented in the black-box algorithm. Here, preds_by_models is shape k x n x d, where k indexes over each of the k models,
        and the bias array is shape k x d x m x k, where each of the k slices corresponds to the level sets to be unbiased of model k.
        """
        while True:
            bias, probs = self._calculate_bias()
            max_weighted_bias, target_set, target_bias = self._find_maximum_bias(bias, probs)
            target_model = target_set[0] # extracts out which model will be updated
            if max_weighted_bias > self.tolerance:
                # book-keeping for out-of-sample algorithm
                self.targets_by_round.append(target_set)
                self.bias_by_round.append(target_bias)
                self.t += 1
                # debias the target model
                mask = self.masks[target_set].astype(bool).flatten()
                mask_size = mask.sum()
                self.preds_by_models[target_model, mask]+=np.tile(target_bias, (mask_size, 1))
            else: 
                break 
            
            # for debugging: tracking mses etc
            self.mses_by_round.append(self.mse_per_model())
            #debug
            #self.preds_by_rounds.append(np.copy(self.preds_by_models)) # only append if not breaking
            self.masks_by_round.append(self.masks)
            
        return None

    def debias(self):   
        #self.preds_by_rounds.append(np.copy(self.preds_by_models)) #DEBUG
        while True:
            # update all the policies
            # policy_outputs will have shape k x n x d
            policy_outputs = np.array([self.policies[i].run_given_preds(self.preds_by_models[i]) for i in range(len(self.policies))]) # shape k x n x d
            # generate masks. These are of shape k x d x m x k' x n, where the masks at [k,d',m'] correspond to the level sets of model k
            # and the second k' is indexing over which model is maximal constrained to that level set of that model. 
            self.masks = self._generate_masks(self.preds_by_models, policy_outputs)  # shape k x d x m x k x n
            bias, probs = self._calculate_bias()
            max_weighted_bias,_,_ = self._find_maximum_bias(bias, probs)
            if max_weighted_bias > self.tolerance:
                self.recompute_masks.append(self.t) # bookkeeping for out of sample computation
                self._update() #Note: update algorithm internally updates all of the k predictors. 
            else: 
                break   
        return self.preds_by_models
    
    def predict(self, oos_init_preds):
        oos_preds_by_models = np.copy(oos_init_preds) #shape k x n x d
        #pred_by_rounds = []
        #pred_by_rounds.append(np.copy(oos_preds_by_models)) #
        #oos_policy_outputs = np.array([self.policies[i].run_given_preds(oos_preds_by_models[i]) for i in range(len(self.policies))]) 
        for (idx, target_set) in enumerate(self.targets_by_round):
            
            # Check if need to recompute masks/policies
            if idx in self.recompute_masks:
                oos_policy_outputs = np.array([self.policies[i].run_given_preds(oos_preds_by_models[i]) for i in range(len(self.policies))])
                oos_masks = self._generate_masks(oos_preds_by_models, oos_policy_outputs) # identify the level sets of the updated model
            
            target_model = target_set[0] #identify the model 
            mask = oos_masks[target_set].astype(bool).flatten()
            mask_size = mask.sum()
            # update the predictions
            oos_preds_by_models[target_model, mask]+=np.tile(self.bias_by_round[idx], (mask_size, 1))

            #pred_by_rounds.append(np.copy(oos_preds_by_models)) #DEBUG
        return oos_preds_by_models
    
    def ensemble(self, preds_by_models):
        """
        preds_by_models: shape k x n x d
        """
        _, n_samples, _ = preds_by_models.shape
        # run policy on each
        policy_outputs = np.array([self.policies[i].run_given_preds(preds_by_models[i]) for i in range(len(self.policies))]) # shape k x n x d
        # find pointwise expected reward of all policies
        rewards = np.vecdot(preds_by_models, policy_outputs) # shape k x n
         #get maximal policy per point
        maximal_policy_indices = np.argmax(rewards, axis=0)
        # get policy which picks maximal policy per datapoint
        ensemble_policy = policy_outputs[maximal_policy_indices, np.arange(n_samples)] # shape n x d
        ensemble_model = preds_by_models[maximal_policy_indices, np.arange(n_samples)]
        # calculate expected self-evaluation of model
        expected_self_eval = np.max(rewards, axis=0).sum()/n_samples
        return ensemble_policy, ensemble_model, expected_self_eval

    def calc_ensemble_return(self, ensemble_policy, true_labels):
        """
        ensemble_policy: shape n x d
        true_labels: shape n x d
        """
        ensemble_returns = np.sum(ensemble_policy*true_labels, axis=1) # shape n
        expected_return = np.mean(ensemble_returns) #scalar
        return expected_return, ensemble_returns



