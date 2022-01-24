# Logistic Regression (LR) from scratch

Implementation of logistic regression from scratch.
Both batch gradient decent and mini-batch algorithms are implemented.
Runs the algorithm with cross validation.

Info about the dataset:
https://archive.ics.uci.edu/ml/index.php.

Base Environment

Create a virtual environment with Anaconda:

conda create -n 462assignment python=3.6
conda activate 462assignment

Load the requirements:

python3 -m pip install -r requirements.txt


Part1:

python3 logistic_regression.py part1 step1
python3 logistic_regression.py part1 step2


##### LR #####

• Step1: LR with batch gradient descent (updates weights after a full pass over
data)
• Step2: LR with stochastic gradient descent (updates weights after mini batches)



