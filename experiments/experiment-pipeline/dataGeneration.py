import numpy as np

"""
Generating synthetic data and storing
"""

def feature_gen(n, dim, cov_min, cov_max, mean_min, mean_max, num_categories):
    """
    Generate feature data, which has one categorical feature and the rest generated as random Gaussian
    according to a random covariance matrix.
    """

    # First, generating the Gaussian data features, of which there will be dim-1

    # Cholesky decomposition of random matrix to get a valid covariance matrix
    M = np.random.uniform(cov_min, cov_max, size=(dim-1, dim-1))
    cov = M.T @ M
    mean = np.random.uniform(mean_min, mean_max, size=dim-1)
    gauss_data = np.random.multivariate_normal(mean, cov, size=n)

    # Next, generate categorical data
    cat_data = np.random.choice(np.arange(num_categories), size=n)

    # Combine
    data = np.hstack((gauss_data, cat_data[:, np.newaxis]))
    
    return data

def poly_label_fun(xs, n_terms, term_size, coeff_min, coeff_max, max_exponent):
        """
        xs: Feature data to label
        n_terms: Number of terms in generating poliynomial
        term_size: Number of features to use in each term
        coeff_min: Min coefficient value of term
        coeff_max: Max coefficient value of term
        max_exponent: Max exponent value of each term. 

        Generates labels w polynomial relation to features, normalized to 0-1

        Note: You could, in current generating process, end up with two different terms in polynomial
        that could be simplified into one term. I don't think this is particularly important. 
        """

        # Polynomial Generation
        feature_dim = xs.shape[1]
        # pick coefficients of each polynomial term 
        coefficients = np.random.uniform(coeff_min, coeff_max, size = n_terms)
        # pick variables to include in that term
        variables = [np.random.choice(np.arange(feature_dim), size=term_size, replace=False) for i in range(n_terms)]
        # pick values in exponent for each of the variables for that term
        powers = [np.random.choice(np.arange(max_exponent+1), size=term_size) for i in range(n_terms)]
        # evaluate each term of the polynomial
        terms = np.array([coefficients[i] * np.prod(xs[:,variables[i]]**powers[i], axis=1) for i in range(n_terms)])
        labels = np.sum(terms, axis=0)
        
        return labels

def poly_label_gen(xs, label_dim, n_poly_terms, term_size, coeff_min, coeff_max, max_exponent, noise_level):
    """
    Assumes that the final row of xs is categorical indicator. For each of the categories described there, uses a different generating function for the ys.

    xs: Feature data
    label_dim: Dimension of label vector; each column will be uncorrelated w others. 
    n_poly_terms: Number of terms in each generating polynomial
    term_size: Size of each polynomial term (in terms of number of features to use)
    coeff_min: Min coefficient value of term
    coeff_max: Max coefficient value of term
    max_exponent: Max exponent value of each term. 
    """
    n = len(xs)
    categories = np.unique(xs[:,-1])
    labels = np.zeros((len(xs), label_dim))
    for cat in categories:
        cat_index = (xs[:,-1]==cat)
        for j in range(label_dim):
            labels[cat_index, j] = poly_label_fun(xs[cat_index], n_poly_terms, term_size, coeff_min, coeff_max, max_exponent)
    cov = np.diag([noise_level]*label_dim)
    noise = np.random.multivariate_normal([0]*label_dim, cov, size=n)
    labels += noise
    return labels

def linear_label_gen(xs, label_dim, noise_level):
    n = xs.shape[0]
    n_features = xs.shape[1]

    slopes = np.random.uniform(size=(n_features, label_dim))
    cov = np.diag([noise_level]*label_dim)
    errs = np.random.multivariate_normal([0]*label_dim, cov, size = n)
    labels = np.matmul(xs, slopes)+errs
    return labels

# def correlated_label_gen(xs, ):   https://stats.stackexchange.com/questions/15011/generate-a-random-variable-with-a-defined-correlation-to-an-existing-variables
#     return None

def main():
    
    # Feature parameters
    n = 10000
    n_features = 20
    cov_min = -3
    cov_max = 3
    mean_min = -5
    mean_max = 5
    num_categories = 5

    xs = feature_gen(n, n_features, cov_min, cov_max, mean_min, mean_max, num_categories)

    # Label parameters
    label_dim = 4
    n_terms = 5
    term_size = 2
    coeff_min = -1
    coeff_max = 1
    max_exponent = 2
    noise = 0.01

    poly_ys = poly_label_gen(xs, label_dim, n_terms, term_size, coeff_min, coeff_max, max_exponent, noise)
    linear_ys = linear_label_gen(xs, label_dim, noise)

    data_path = '../../data/synthetic'

    np.savetxt(f"{data_path}/features.csv", xs, delimiter=",")
    np.savetxt(f"{data_path}/poly-labels.csv", poly_ys, delimiter=",")
    np.savetxt(f"{data_path}/linear-labels.csv", linear_ys, delimiter=",")

if __name__ == "__main__":
    main()