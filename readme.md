## Creating automated pipeline using dvc and git

Using the [UCI Concrete cement prediction](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength) as a base usecase, this project
uses [DVC](https://dvc.org/),[Optuna](https://optuna.org/) and [mlflow](https://mlflow.org/) for an end-to-end implementation of Machine Learning
pipeline.

<br>

To test the project, first clone the repo
```
git clone https://github.com/Yadu9238/mlpl.git
```
Download the dataset from the uci repo and place it under 'data/raw/'. <br>

Use
```
dvc repro -f
```
to run the entire pipeline.


### To create the project from scratch.

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

```bash
├── data/                              *contains datasets*
│   ├── processed
│   └── raw/
├── dvc.lock
├── dvc.yaml
├── logs/                             
│   ├── model.log
│   ├── preprocessing.log
│   └── training.log
├── notebooks
├── params.yaml                        * parameters for the project *
├── readme.md
├── report/
│   ├── params.json
│   └── scores.json
├── requirements.txt
├── saved_models
├── src/
│   ├── get_data.py                    *helper function to read data from config*
│   ├── load_data.py                   *get data from remote source *   
│   ├── log.py                         *helper function for logging*
│   ├── model.py                       *build model *
│   ├── split_data.py                  *used for splitting data into test and train data *
│   └── train.py                       *train the model and save the best one *
├── template.py                        *script to generate initial file dirs *

```

6. Update dvc.yaml to add the required stages in the pipeline

7. Run ``` dvc repro ``` or ``` dvc repro -f ``` to run the entire pipeline

