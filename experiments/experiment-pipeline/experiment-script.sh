#!/bin/bash

# Run the data generation 
#python dataGeneration.py

# # Run model generation 
#python modelSetup.py

#python debiasingModels.py linear-label gb coord variance 600
# python debiasingModels.py linear-label gb group variance 600
# python debiasingModels.py linear-label gb all variance 600

python debiasingModels.py linear-label lin coord variance 2
# python debiasingModels.py linear-label lin coord variance 600
# python debiasingModels.py linear-label lin coord variance 600

#python debiasingModels.py poly-label gb coord variance 600
# python debiasingModels.py poly-label gb group variance 600
# python debiasingModels.py poly-label gb all variance 600
