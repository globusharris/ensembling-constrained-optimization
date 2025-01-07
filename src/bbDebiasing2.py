import numpy as np
import itertools

#Non round-robin version of debiasing

class bbModel:
    def __init__(self, own_policy, other_policies, train_ys, curr_preds, tolerance):
        self.own_policy = own_policy
        self.other_policies = other_policies
        self.train_ys = train_ys
        self.curr_preds = np.copy(curr_preds)
        self.tolerance = tolerance

        self.n_policies = len(other_policies)+1 # +1 to include own policy
        self.n_coords = own_policy.dim
        self.n_bins = own_policy.n_vals
        self.n_samples = len(train_ys)
        self.tolerance = tolerance

        # run all policies in-sample on current predictions. Shape: list of length k+1 with entries of shape n x d 
        self.policies =[self.own_policy.run_given_preds(self.curr_preds)] + self.other_policies
        # generate level sets of all of the policies across all dimensions.
        self.coordinate_values = self.own_policy.coordinate_values 

        # generate the level sets of all of the policies as collection of Boolean masks
        self.masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, self.n_samples), dtype='bool')
        [self._generate_masks(self.masks, self.policies, policy_idx) for policy_idx in range(self.n_policies)] 

        # tracking for out-of-sample computation
        self.targets_by_round = []
        self.bias_by_round = []


    def _generate_masks(self, masks, policies, policy_idx):
        policy = policies[policy_idx]
        for (coord, i) in itertools.product(range(self.n_coords), range(self.n_bins)):
            val = self.coordinate_values[i]
            if i < self.n_bins-1:
                mask = (policy[:,coord]>=val) & (policy[:,coord]<val+self.own_policy.gran)
            else:  # special case for last bin to deal with edges
                mask = (policy[:,coord]>=val) & (policy[:,coord]<=val+self.own_policy.gran)
            masks[policy_idx, coord, i] = mask
        return masks

    def _calculate_bias(self):
        """
        For all c in C (masks array), calculates E[y - h(x)|x in c]
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
        l_infinity = np.max(np.abs(bias), axis=-1)
        weighted_bias = probs*l_infinity
        max_weighted_bias = weighted_bias.max()
        target_set = tuple(np.argwhere(weighted_bias==max_weighted_bias)[0]) #converting into tuple for indexing
        target_bias = bias[target_set]
        return max_weighted_bias, target_set, target_bias

    def _update(self):
        """
        Debias on level sets
        """
        while True:
            bias, probs = self._calculate_bias()
            max_weighted_bias,target_set, target_bias = self._find_maximum_bias(bias, probs)
            if max_weighted_bias > self.tolerance:
                self.targets_by_round.append(target_set)
                self.bias_by_round.append(target_bias)
                mask = self.masks[target_set].astype(bool).flatten()
                mask_size = mask.sum()
                self.curr_preds[mask]+=np.tile(target_bias, (mask_size, 1)) 
            else:
                break    
        return None

    def debias(self):
        """
        Currently assumes that all of policies have same granularity of binning, dimension, etc. 
        """    
        while True:    
            bias, probs = self._calculate_bias()
            max_weighted_bias,_,_ = self._find_maximum_bias(bias, probs)
            if max_weighted_bias > self.tolerance:
                self._update()
            else:
                break
            self.policies[0] = self.own_policy.run_given_preds(self.curr_preds)    # get policy of updated predictions
            self._generate_masks(self.masks, self.policies, 0) # update level sets of own policy
        
        return self.curr_preds
    
    def predict(self, oos_init_preds, oos_other_policies):
        oos_preds = np.copy(oos_init_preds)
        oos_n = len(oos_preds)
        policies = [self.own_policy.run_given_preds(oos_preds)] + oos_other_policies
        oos_masks = np.zeros((self.n_policies, self.n_coords, self.n_bins, oos_n), dtype='bool')
        [self._generate_masks(oos_masks, policies, policy_idx) for policy_idx in range(self.n_policies)]
        for (idx, target) in enumerate(self.targets_by_round):
            mask = oos_masks[target].astype(bool).flatten()
            mask_size = mask.sum()
            oos_preds[mask] += np.tile(self.bias_by_round[idx], (mask_size, 1))
        return oos_preds