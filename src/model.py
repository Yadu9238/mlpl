from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna 
from optuna.integration.mlflow import MLflowCallback
from get_data import read_params



def create_model(trial):
    model_type = trial.suggest_categorical("model",['LinearReg','RandomForest','GradientBoosting'])

    if model_type == 'LinearReg':
        model = LinearRegression()
    if model_type == 'RandomForest':
        max_depth = trial.suggest_int('max_depth',3,100)
        n_estimators = trial.suggest_int('n_estimators',50,200)
        model = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)
    if model_type == 'GradientBoosting':
        max_depth = trial.suggest_int('max_depth',3,100)
        n_estimators = trial.suggest_int('n_estimators',50,200)
        model = GradientBoostingRegressor(max_depth=max_depth,n_estimators=n_estimators)
    
    return model 

def objective(trial,X_train,X_test,y_train,y_test):
    
    model = create_model(trial)
    model.fit(X_train,y_train)

    return model.score(X_test,y_test)


def build_model(X_train,X_test,y_train,y_test):
    mlflc = MLflowCallback(metric_name = 'r2_score')
    study = optuna.create_study(study_name='Hyper',direction = 'maximize')
    study.optimize(lambda trial:objective(trial,X_train,X_test,y_train,y_test),n_trials=20,callbacks=[mlflc])
    print("Number of trials : {}".format(len(study.trials)))

    print("best trial:",study.best_trial)
    res = study.best_trial
    print("Value: {}".format(res.value))

    print("params:")
    for k,v in res.params.items():
        print("    {}:{}".format(k,v))
    
    return create_model(res)