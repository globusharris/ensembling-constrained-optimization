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
        self.n_coords = policies[0].dim
        self.n_bins = policies[0].n_vals
        self.n_samples = len(train_ys)
        self.tolerance = tolerance
        self.gran = policies[0].gran

        self.policy_outputs = None # shape k x n x d
        self.masks = None # shape k x m x d x k x n; masks of level sets to debias wrt for each of the k models 

        # # run all policies in-sample on current predictions. Shape: list of length k+1 with entries of shape n x d 
        # self.policies =[self.own_policy.run_given_preds(self.curr_preds)] + self.other_policies
        # # generate level sets of all of the policies across all dimensions.
        # self.coordinate_values = self.own_policy.coordinate_values 

        # # generate the level sets of all of the policies as collection of Boolean masks
        # self.masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, self.n_samples), dtype='bool')
        # [self._generate_masks(self.masks, self.policies, policy_idx) for policy_idx in range(self.n_policies)] 

        # # tracking for out-of-sample computation
        # self.targets_by_round = []
        # self.bias_by_round = []
    
    def _generate_model_ls_masks(self, preds_by_models):
        """
        Should be usable both in and out of sample, hence passing predictions in as input. 
        Generates the d x m level sets of each of the k input models, where d is the dimension of the models' predictions and m is the number of LS per coordinate. 
        Each level set is expressed as a length-n Boolean mask; where if the mask is true at index i it means that datapoint i is a member of the level set. 
        
        preds_by_model: shape k x n x d
        output masks: shape k x d x m x n; masks[k,d,m] is a length-n Boolean mask of the level set m of 
        coordinate d of the predictions of model k
        """
        masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, self.n_samples))
        for (k,d,m) in itertools.product(range(self.n_policies), range(self.n_coords), range(self.n_bins)):
            val = self.coordinate_values[m]
            if d < self.n_bins - 1:
                mask = (preds_by_models[k,:,d]>=val) & preds_by_models[k,:,d]<val+self.gran
            else: # special case for last bin to deal with edges
                mask = (preds_by_models[k,:,d]>=val) & preds_by_models[k,:,d]<=val+self.gran
            masks[k,d,m] = mask
        return masks

    def _generate_maximal_model_masks(self, preds_by_models, policy_outputs):
        """
        Should be usable both in and out of sample!
        preds_by_models: k x n x d
        all_policies: k x n x d
        output masks: shape k x n; masks[k] is a length-n Boolean mask of the collection of training 
        points which have model k's induced policy predicted to be maximal.
        """
        dot_products = np.vecdot(preds_by_models, policy_outputs) # shape k x n of dot products of policy of model i and model i's predictions per point.
        max_models = np.argmax(dot_products, axis=0) # shape n of maximal model per training sample
        masks = np.zeros((self.n_policies, self.n_samples))
        masks[max_models, np.arange(self.n_samples)] = True
        return masks
    
    def _generate_masks(self, preds_by_model):
        max_model_masks = self._generate_maximal_model_masks(preds_by_model) # shape k x d x m x n
        model_ls_masks = self._generate_model_ls_masks(preds_by_model) # shape k x n 
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
        diffs = self.train_ys*masks - self.curr_preds*masks
        sums = np.sum(diffs, axis=-2)
        ns = np.sum(masks, axis=-2)
        bias = np.where(ns>0, sums/ns.clip(min=1),0) # clip is to avoid div by 0 errors
        probs = np.squeeze(ns/self.n_samples)
        return bias, probs

    def _find_maximum_bias(self, bias, probs):
        l_infinity = np.max(np.abs(bias), axis=-1) # shape k x d x m x k
        weighted_bias = probs*l_infinity
        max_weighted_bias = weighted_bias.max()
        target_set = tuple(np.argwhere(weighted_bias==max_weighted_bias)[0]) #converting into tuple for indexing
        target_bias = bias[target_set]
        return max_weighted_bias, target_set, target_bias
    
    def _update(self, model_idx):
        """
        Debias on level sets
        """
        # while True:
        #     bias, probs = self._calculate_bias()
        #     max_weighted_bias,target_set, target_bias = self._find_maximum_bias(bias, probs)
        #     if max_weighted_bias > self.tolerance:
        #         self.targets_by_round.append(target_set)
        #         self.bias_by_round.append(target_bias)
        #         mask = self.masks[target_set].astype(bool).flatten()
        #         mask_size = mask.sum()
        #         self.curr_preds[mask]+=np.tile(target_bias, (mask_size, 1)) 
        #     else:
        #         break    
        return None

    def debias(self):    
        while True:
            # update all the policies
            # policy_outputs will have shape k x n x d
            self.policy_outputs = np.array([self.policies[i].run_given_preds(self.preds_by_model[i]) for i in range(len(self.policies))]) # shape k x n x d
            
            # generate masks. These are of shape k x d x m x k x n, where the masks at [k',d',m'] correspond to the level sets of model k'
            # and the second k' is indexing over which model is maximal constrained to that level set of that model. 
            self.masks = self._generate_masks(self.preds_by_model)  # shape k x d x m x k x n
            bias, probs = self._calculate_bias()
            max_weighted_bias,_,_ = self._find_maximum_bias(bias, probs)
            if max_weighted_bias > self.tolerance:
                for model_idx in range(self.n_models):
                self.update(model_idx)
            else: 
                break
            
        return self.curr_preds
    
    def predict(self, oos_init_preds, oos_other_policies):
        # oos_preds = np.copy(oos_init_preds)
        # oos_n = len(oos_preds)
        # policies = [self.own_policy.run_given_preds(oos_preds)] + oos_other_policies
        # oos_masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, oos_n), dtype='bool')
        # [self._generate_masks(oos_masks, policies, policy_idx) for policy_idx in range(self.n_policies)]
        # for (idx, target) in enumerate(self.targets_by_round):
        #     mask = oos_masks[target].astype(bool).flatten()
        #     mask_size = mask.sum()
        #     oos_preds[mask ] += np.tile(self.bias_by_round[idx], (mask_size, 1))
        # return oos_preds