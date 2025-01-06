import numpy as np
import itertools

#Non round-robin version of debiasing

def calculate_bias(curr_preds, true_ys, masks):
    """
    For all c in C (masks array), calculates E[y - h(x)|x in c]
    """
    # converting masks to floats and expanding dimension so can broadcast
    masks = masks.astype(np.float32)
    masks = np.expand_dims(masks, axis=-1)
    diffs = true_ys*masks - curr_preds*masks
    sums = np.sum(diffs, axis=-2)
    ns = np.sum(masks, axis=-2)
    bias = np.where(ns>0, sums/ns.clip(min=1),0) # clip is to avoid div by 0 errors
    probs = np.squeeze(ns/len(true_ys))
    return bias, probs

def find_maximum_bias(bias, probs):
    l_infinity = np.max(np.abs(bias), axis=-1)
    weighted_bias = probs*l_infinity
    max_bias = weighted_bias.max()
    target_set = tuple(np.argwhere(weighted_bias==weighted_bias.max())[0]) #converting into tuple for indexing
    return max_bias, target_set

def update(curr_preds, true_ys, masks, tolerance):
    """
    sets = list of numpy arrays where each is length n of Boolean flags for that particular set.
    """
    while True:
        bias, probs = calculate_bias(curr_preds, true_ys, masks)
        max_bias,target_set = find_maximum_bias(bias, probs)
        if max_bias > tolerance:
            mask = masks[target_set].astype(bool).flatten()
            mask_size = mask.sum()
            curr_preds[mask]+=np.tile(bias[target_set], (mask_size, 1)) 
        else:
            break
    
    return curr_preds

def blackbox_debias(curr_preds, true_ys, own_policy, other_policies, tolerance):
    """
    Currently assumes that all of policies have same granularity of binning, dimension, etc. 
    """
    
    n_policies = len(other_policies)+1 # +1 to include own policy
    n_coords = own_policy.dim
    n_bins = own_policy.n_vals
    n_samples = len(true_ys)

    # run all policies on current predictions. Shape: list of length k+1 with entries of shape n x d 
    policies =[own_policy.run_given_preds(curr_preds)] + other_policies
    # generate level sets of all of the policies across all dimensions.
    coordinate_values = own_policy.coordinate_values 
    masks = np.zeros((n_policies, n_coords, n_bins, n_samples), dtype='bool')    
    def generate_masks(policy_idx):
        policy = policies[policy_idx]
        for (coord, i) in itertools.product(range(n_coords), range(n_bins)):
            val = coordinate_values[i]
            if i < n_bins-1:
                mask = (policy[:,coord]>=val) & (policy[:,coord]<val+own_policy.gran)
            else:  # special case for last bin to deal with edges
                mask = (policy[:,coord]>=val) & (policy[:,coord]<=val+own_policy.gran)
            masks[policy_idx, coord, i] = mask
    
    [generate_masks(policy_idx) for policy_idx in range(n_policies)]
    while True:    
        bias, probs = calculate_bias(curr_preds, true_ys, masks)
        max_bias,_ = find_maximum_bias(bias, probs)
        if max_bias > tolerance:
            curr_preds = update(curr_preds, true_ys, masks, tolerance)
        else:
            break

        policies[0] = own_policy.run_given_preds(curr_preds)    # get policy of updated predictions
        generate_masks(0) # update level sets of own policy
    
    return curr_preds