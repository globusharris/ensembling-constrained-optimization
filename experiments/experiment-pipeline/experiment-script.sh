#!/bin/bash

# Run the data generation 
python dataGeneration.py

# # Run model generation 
python modelSetup.py

# python debiasingModels.py linear-label gb coord variance
# python debiasingModels.py linear-label gb group variance
# python debiasingModels.py linear-label gb all variance

python debiasingModels.py linear-label lin coord variance
# python debiasingModels.py linear-label lin coord variance
# python debiasingModels.py linear-label lin coord variance

python debiasingModels.py poly-label gb coord variance
# python debiasingModels.py poly-label gb group variance
# python debiasingModels.py poly-label gb all variance

