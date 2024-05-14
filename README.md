# README

Code Structure:

* src
    * bbDebiasing.py: Implementation of "bias bounties" method of debiasing
    * maxEnsembleDebias.py: Implementation of max ensembling method of debiasing. 
    * policies.py: Policies. Currently just takes max coordinate. 
* experiments
    * synthetic-experiments.ipynb: Synthetic Experiments
    * electricity-experiments.ipynb: Initial experiments on electric transformer dataset
    * options-experiments.ipynb: Currently just loading in the data, haven't written any experiments. 
* old-work
    * A bunch of garbage code that I don't want to get rid of. 
* testing
    * Various testing of code functionality

To build environment:
conda env create -f environment.yml
conda activate bb-portfolio


Because of problems with the default opensource optimization solvers used by cvxpy, have switched to Gurobi. This requires a separate download of Gurobi and a license; they have academic licenses for free. 

See https://www.gurobi.com/features/academic-named-user-license/ for instructions. After downloading, if running code in VSCode, will need to quit and reopen VSCode. 

To run experiments on other data sources:

Can use basic-db-script.py code in experiment-pipeline to run. You'll need to specify a file path to data, which the file
expects to be broken into two csvs, one features and one labels (though this can easily be changed). You'll also need to supply a filepath to a director of initial pickled models; see modelSetup.py for basic code for how to pickle your models once they've been generated. 

To run synthetic experiments:

1. Generate data:
```python dataGeneration.py```

This generates the data and stores it in data/synthetic. There are two methods for genering labels: the first has linear labels w features, the second have more complex polynomial relations. Currently, the dimensions etc of data generated may be toggled within the "main" function in this file.

2. Generate initial models:

```python modelSetup.py```

This trains a variety of different models on the generated data, and stores them in an init-model file, divided first by what the label type was, and then the type of model. Each of the models is an expert on some subregion of the dataset: either they were trained on a subgroup of data or on a single coordinate. Everywhere a model isn't an expert, it predicts the label mean
of that coordinate. 

The models are stored, so you only need train them once. 

3. Generate debiased models:

```python debiasingModels.py label-version model-type specialization policy-name```

This runs both the BB and MaxEnsemble debiasing methods, using the input models and a policy you specify in the commandline with policy-name. The commandline arguments are as follows:

label-version: either linear-label or poly-label; referring to which labels you are running with
model-type: either linear or gb
specialization: coord or group
policy_name: simplex, linear-constraint, or variance. 

4. Analysis:

You can reload in the debiased models to run analysis on, using, e.g. 

```
with open(filepath, 'rb') as file:
           model = dill.load(file)
```



