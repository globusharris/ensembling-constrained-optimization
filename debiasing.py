import numpy as np

def debias(h, xs, ys, cond, cond_inputs):
    """
    Debias model on portion specified by cond
    h: initial model 
    xs: feature vector
    ys: m-dimensional output vector of labels
    cond: debiasing condition; boolean function which expects cond_inputs as its inputs
    """
    # calculate indices where condition holds and subset xs and ys accordingly:
    condition_flags = cond(xs, cond_inputs)
    xs_subset = xs[condition_flags]
    ys_subset = ys[condition_flags]

    # calculate bias on subset of data where condition is satisfied:
    bias_terms = h(xs_subset) - ys_subset

    # calculate (m-dimensional) empirical mean of E[h(x)-r | cond]
    empirical_bias = np.average(bias_terms, axis=1)
    
    def h_debiased(feature_data):
        np.apply_along_axis(cond, 0, feature_data, cond_inputs)
        return h(xs)-bias