# This is a little helper designed to help to install this package for use with
# reticulate in R
# It's not very neat and it's not particularly failsafe, but it hopefully gets
# the job done.

set -e

# First, make the environment and install the dependencies:
conda create -y -n svgp python=3.7 numpy scipy pandas scikit-learn 

# Nonsense to get conda to work properly
eval "$(conda shell.bash hook)"
conda activate svgp

# Install tensorflow and tensorflow probability using pip
pip install tensorflow tensorflow-probability

# Install the SVGP package
cd .. && python setup.py develop

# cd back into this directory
cd R

# Clone ml_tools (and remove if it exists first to make sure we have the latest
# version)
rm -rf ml_tools && git clone git@github.com:martiningram/ml_tools.git

cd ml_tools && python setup.py develop

echo "Success!"
