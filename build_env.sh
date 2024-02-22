# Name of the virtual environment
ENV_NAME="libmultilabel4dpr"

# Creating a new conda environment
echo "Clean and Creating a new conda environment named $ENV_NAME"
conda activate base
conda remove -n $ENV_NAME --all --yes
conda create -n $ENV_NAME python=3.7 -y

# Activating the environment
echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

# Installing libraries in requirements.txt
echo "Installing libraries in requirements.txt"
cat requirements.txt | xargs -n 1 -L 1 pip3 install; \

echo "Setup complete. Activate the conda environment using: conda activate $ENV_NAME"

