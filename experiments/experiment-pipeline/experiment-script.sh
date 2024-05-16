#!/bin/bash

# Run the data generation
#python dataGeneration.py

# # Run model generation
#python modelSetup.py

echo starting1 >> status.txt
echo $(date) >> status.txt
python debiasingModels.py linear-label gb coord variance 1000
# python debiasingModels.py linear-label gb group variance 600
# python debiasingModels.py linear-label gb all variance 600
echo starting2 >> status.txt
echo $(date) >> status.txt
python debiasingModels.py linear-label lin coord variance 1000
# python debiasingModels.py linear-label lin coord variance 600
# python debiasingModels.py linear-label lin coord variance 600
echo starting3 >> status.txt
echo $(date) >> status.txt
python debiasingModels.py poly-label gb coord variance 1000
# python debiasingModels.py poly-label gb group variance 600
# python debiasingModels.py poly-label gb all variance 600
echo starting4 >> status.txt
echo $(date) >> status.txt
python debiasingModels.py poly-label lin coord variance 1000
echo starting5 >> status.txt
echo $(date) >> status.txt
