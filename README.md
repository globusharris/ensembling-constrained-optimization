# README

This codebase implements the ensembling algorithms in *Model Ensembling for Constrained Optimization* by Ira Globus-Harris, Varun Gupta, Michael Kearns, and Aaron Roth. If you use this code, please cite our paper:

```
@article{globusEnsembling24,
  title={Model Ensembling for Constrained Optimization},
  author={Globus-Harris, Ira and Gupta, Varun and Kearns, Michael and Roth, Aaron}
  year={2024}
}
```

## Structure of codebase:

* src: Basic source code for the actual algorithms
    * bbDebiasing.py: Implementation of black box method for ensembling
    * maxEnsembleDebias.py: Implementation of white box for ensembling
    * policies.py: Code for the policies induced by the downstream optimization problem on predictions. Currently includes:
        * "Simplex" policy: For testing; just allocates 1 to largest coordinate.
        * Linear optimization: Simple linear optimization.
        * Covariance constrained obtimization: Constrained by covariance matrix of the labels. 
* experiments: Experiment pipeline and figure generation for paper. 
    * paper-figs.ipynb: Code for generating the figures in papers. Assumes you already have run the debiasing of algorithms, which is set up in the experiment-pipeline subfolder
    * experiment-pipeline: Subfolder for running the actual experiments. 
        Instructions for running the experiments is below; the main shell script to rerun everything is experiment-script.sh.
        * experiment-script.sh: shell script for running the experiments in the paper. 
        * dataGeneration.py: This generates the data and stores it in data/synthetic. There are two methods for genering labels: the first has linear labels w features, the second have more complex polynomial relations. Currently, the dimensions etc of data generated may be toggled within the "main" function in this file. Paper experiments only use linear labels. 
        * modelSetup.py: This trains a variety of different models on the generated data, and stores them in an init-model file. The file folders are divided into subcategories, first by what the label type was, and then the type of model. Each of the models is an expert on some subregion of the dataset: either they were trained on a subgroup of data or on a single coordinate. Everywhere a model isn't an expert, it predicts the label mean of that coordinate. The models are stored, so you only need train them once.
        * debiasingModels.py: This runs both the BB and MaxEnsemble debiasing methods, using the input models and a policy you specify in the commandline with policy-name. The commandline arguments are as follows:
            * label-version: either linear-label or poly-label; referring to which labels you are running with
            * model-type: either linear or gb
            * specialization: coord or group
            * policy_name: simplex, linear-constraint, or variance. 
            * max_depth: Maximum depth; maximum number of rounds that debiasing algorithms will run for 
            * subsample_size: Number of datapoints from the training data to subsample for the debiasing process. If the optimization problem (policy) you are running on is slow, then since it has to run on each training point at each round of debiasing, if you don't subsample it can be too slow to be tractable. Trades off with generalization out-of-sample.
        * analysisHelp.py: Variety of helpful functions for analysis and plotting. Can be run as a script on all of your existing debiased models by running through command line, or can just use as helper functions (which is what paper-figs.ipynb does)
        * Subfolders: A variety of subfolders will be generated when you run the experiments:
            * init-models: Where the initial trained models (without debiasing/self-consistency) are stored.
            * debiased-models: Where the pickle files of the debiased models are stored. 

## Running the Code

### Building environment and installing dependencies

#### Install Gurobi and get relevant licenses

Because of problems with the default opensource optimization solvers used by cvxpy, have switched to Gurobi. This requires a separate download of Gurobi and a license; they have academic licenses for free. See https://www.gurobi.com/features/academic-named-user-license/ for instructions. After downloading, if running code in VSCode, will need to quit and reopen VSCode. 

#### Build environment

Run the following to create an environment using dependencies in the yml file and then activate it. 
```
conda env create -f environment.yml
conda activate bb-portfolio
```

### Running experiments from paper

There is a shell script in ```experiments/experiment-pipeline``` to run all the code to regenerate the data and models in the paper. 

1. Make the script executable

```
chmod +x experiments/experiment-pipeline/experiment-script.sh 
```

2. Run the script

Run script. Recommend running in background and piping output to a status file. 
```
cd experiments/experiment-pipeline
./experiment-script.sh &> status.txt &
```
Note: This code takes quite a few hours to run on our machine. When running on our server, we had trouble with disconnecting from the server leading to the processes getting occassionally terminated. Running everything inside a tmux session helped this, which you can do as follows:
```
tmux
< run whatever code you wanted to; e.g. ./experiment-script.sh &> status.txt &>
ctrl+b 
d
```
This creates a pseudo-terminal, which you can run your code in. Then you detatch the pseudo terminal by pressing ctrl+b and then "d". If you want to access that pseudo-terminal again, you can type:
```
tmux attach
```

3. Run the figure generation

This is all in the paper-figs.ipynb folder. Note that you will be unable to run the figure generation without running the script first, as the figures assume that the debiased model files have been pickled and stored. 

### Running experiments from other datasources

You will need initial data and models to start with, and can use the debiasingModels.py file as a template for how to call the debiasing algoithms.

Note: Currently, the black box algorithm tracks payoff over each round of debiasing, for the purpose of our paper figures. This actually substantially slows down the runtime, since the optimization problem is the primary bottleneck, and is unnecessary if you only care about the performance of the final unbiased algorithm's performance. You can remove this tracking by commenting out line 92 of src/bbDebiasing.py.





