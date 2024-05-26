#!/bin/bash

#Run the data generation 
python dataGeneration.py
#Run the data generation
python dataGeneration.py

# Run model generation
python modelSetup.py

echo 1
max_depth=5000
subsample_size=400

echo 2
python debiasingModels.py linear-label gb coord variance $max_depth $subsample_size
echo 3
python debiasingModels.py linear-label gb group variance $max_depth $subsample_size
echo 4
python debiasingModels.py linear-label gb coord linear-constraint $max_depth $subsample_size
echo 5
python debiasingModels.py linear-label gb group linear-constraint 8000 $subsample_size
