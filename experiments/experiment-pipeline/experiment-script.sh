#!/bin/bash

# Run the data generation 
# python dataGeneration.py
# Run the data generation
#python dataGeneration.py

# # Run model generation
#python modelSetup.py

echo 1
#python debiasingModels.py linear-label gb coord variance 1000
#python debiasingModels.py linear-label gb group variance 5000
#python debiasingModels.py linear-label gb all variance 1000

python debiasingModels.py linear-label gb coord linear-constraint 1000

echo 2
#python debiasingModels.py linear-label lin coord variance 1000
#python debiasingModels.py linear-label lin group variance 5000
#python debiasingModels.py linear-label lin all variance 1000

python debiasingModels.py linear-label lin coord linear-constraint 1000

echo 3
#python debiasingModels.py poly-label gb coord variance 1000
#python debiasingModels.py poly-label gb group variance 5000
#python debiasingModels.py poly-label gb all variance 1000

python debiasingModels.py poly-label gb coord linear-constraint 1000


echo 4
#python debiasingModels.py poly-label lin coord variance 1000
#python debiasingModels.py poly-label lin group variance 5000
#python debiasingModels.py poly-label lin all variance 1000
python debiasingModels.py poly-label lin coord linear-constraint 1000
echo starting1 
echo $(date)
python debiasingModels.py linear-label gb coord variance 1000
# python debiasingModels.py linear-label gb group variance 600
# python debiasingModels.py linear-label gb all variance 600
echo starting2 
echo $(date)
python debiasingModels.py linear-label lin coord variance 1000
# python debiasingModels.py linear-label lin coord variance 600
# python debiasingModels.py linear-label lin coord variance 600
echo starting3
echo $(date)
python debiasingModels.py poly-label gb coord variance 1000
# python debiasingModels.py poly-label gb group variance 600
# python debiasingModels.py poly-label gb all variance 600
echo starting4 
echo $(date)
python debiasingModels.py poly-label lin coord variance 1000
echo starting5 
echo $(date) 
