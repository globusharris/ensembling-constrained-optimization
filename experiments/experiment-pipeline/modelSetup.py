import numpy as np
import dill

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.linear_model import LinearRegression

"""
Training initial models and storing
"""

# First type of model: good on one coordinate, bad on others
# Second type of models: good on one subgroup, bad on others

def meta_model_by_coord(xs, ys, coord, model_type, model_params):
    label_dim = ys.shape[1]

    if model_type=='gradient-boost':      
        model = GradientBoostingRegressor(**model_params)
        model.fit(xs, ys[:,coord])  
    elif model_type=='linear':
        model = LinearRegression()
        model.fit(xs, ys[:,coord])

    def h(xs):
            labels = np.zeros((len(xs), label_dim))
            labels[:,coord] = model.predict(xs)
            other_coords = np.arange(label_dim)[np.arange(label_dim)!=coord]
            labels[:, other_coords] = np.tile(np.mean(ys[:,other_coords], axis=0), (len(labels),1))
            return labels
    
    return h

def meta_model_by_group(xs, ys, group, model_type, model_params):
    """ 
    Assumes that xs have last coordinate define group membership.
    """
    label_dim = ys.shape[1]

    # check that specified group is valid
    if np.sum(xs[:,-1]==group)==0:
         print("Invalid group")
         return None
    
    target_indices = (xs[:,-1]==group)

    if model_type=='gradient-boost':      
        model = MultiOutputRegressor(GradientBoostingRegressor(**model_params))     
        model.fit(xs[target_indices], ys[target_indices])  
    elif model_type=='linear':
        model = LinearRegression()
        model.fit(xs[target_indices], ys[target_indices])

    def h(xs):
            target_indices = (xs[:,-1]==group)
            labels = np.zeros((len(xs), label_dim))
            labels[target_indices] = model.predict(xs[target_indices])
            other_coords = np.arange(label_dim)[np.logical_not(target_indices)]
            labels[other_coords] = np.tile(np.mean(ys[other_coords], axis=0), (np.sum(other_coords),1))
            return labels
    
    return h


def main():

    # load in data
    data_path = '../../data/synthetic'
    xs = np.loadtxt(f"{data_path}/features.csv", delimiter=',')
    lin_ys = np.loadtxt(f"{data_path}/linear-labels.csv", delimiter=',')
    poly_ys = np.loadtxt(f"{data_path}/poly-labels.csv", delimiter=',')

    storage_path_lin = "./init-models/linear-label"
    storage_path_poly = "./init-models/poly-label"
    gb_model_params = {'learning_rate': 0.1, 'max_depth':6, 'random_state':42}

    label_dim = lin_ys.shape[1]

    for coord in range(label_dim):
        model = meta_model_by_coord(xs, lin_ys, coord, "gradient-boost", gb_model_params)
        path = f"{storage_path_lin}/gb/coord_{coord}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_coord(xs, poly_ys, coord, "gradient-boost", gb_model_params)
        print(model(xs).shape)
        path = f"{storage_path_poly}/gb/coord_{coord}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_coord(xs, lin_ys, coord, "linear", {})
        path = f"{storage_path_lin}/lin/coord_{coord}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_coord(xs, poly_ys, coord, "linear", {})
        path = f"{storage_path_poly}/lin/coord_{coord}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
    
    groups = np.unique(xs[:,-1])

    for group in groups:
        model = meta_model_by_group(xs, lin_ys, group, 'gradient-boost', gb_model_params)
        path = f"{storage_path_lin}/gb/group_{group}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_group(xs, poly_ys, group, 'gradient-boost', gb_model_params)
        path = f"{storage_path_poly}/gb/group_{group}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_group(xs, lin_ys, group, 'linear', gb_model_params)
        path = f"{storage_path_lin}/lin/group_{group}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
        model = meta_model_by_group(xs, poly_ys, group, 'linear', gb_model_params)
        path = f"{storage_path_poly}/lin/group_{group}.pkl"
        with open(path, 'wb') as file:  
            dill.dump(model, file)
    

if __name__ == "__main__":
    main()