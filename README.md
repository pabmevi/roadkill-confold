Environment files produced for use in Anaconda.
For the complete envionment use the pytorch_tabular environment pytorch_tabular_environment.yml

Clone the code from the repository:
git clone git@github.com:rAIson-Lab/extinction-risk-inference.git

Then make sure to initialize the code from the recursive environment:
git submodule update --init --recursive

Then you can install the necessary libraries from the anaconda environment:
env create -f pytorch_tabular_environment.yml

for using the attention based models, install pytorch tabular after the Anaconda environemnt is created using pip due to there not being a Anaconda library for pytorch_tabular:
pip install -U “pytorch_tabular[extra]”