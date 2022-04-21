## Creating automated pipeline using dvc and git

1. Created virtual env and installed required packages
```
python pip install -r requirements.txt
```
2. template.py -> creates the project structure
```
python template.py
```
3. Initialize git
```
git init
```
4. Initialize dvc
```
dvc init
```
5. Write necessary modules
```
get_data -> reading data and saving raw data locally
load_data -> used to load the data
split_data -> used to split raw data to train test
model ->uses optuna and create best model
train -> general training file
```
6. Update dvc.yaml to add the required stages in the pipeline

7. Run ``` dvc repro ``` or ``` dvc repro -f ``` to run the entire pipeline


