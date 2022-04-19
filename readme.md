creating automated pipeline using dvc and git


created env and installed req packages

template.py -> creates the project structure

git init

dvc init


dvc add data
update repo

update params.yaml
update dvc.yaml -> define pipeline

get_data -> reading data and saving raw data locally
load_data -> used to load the data
split_data -> used to split raw data to train test
model ->uses optuna and create best model
train -> general training file
