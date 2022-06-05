from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import optuna
from optuna.integration.mlflow import MLflowCallback
from get_data import read_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
from log import logger
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core import Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute


subscription_id = '046415a5-fb0a-42cf-b7c6-1a5c9616c874'
resource_group = 'RS'
workspace_name = 'MLOps-ws'

spn_credentials = {
        'tenant_id': "372ee9e0-9ce0-4033-a64a-c07073a91ecd",
        'service_principal_id': "fdb84aa3-4b1f-4287-8692-9c0fe8ec1f61",
        'service_principal_password': "Q5bV9QRfUYXhYkUp8w~AaMWSuaerMESfEn",
    }
ws = Workspace(subscription_id, resource_group, workspace_name,auth = ServicePrincipalAuthentication(**spn_credentials))
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
'''
strenv = Environment('cement-strength')
strenv.python.user_managed_dependencies = False
strenv.docker.enabled = True

'''
exp_name = 'azure-test-2305'
exper = Experiment(workspace = ws,name = exp_name)
#experiment = Experiment(workspace = ws,name = 'azure-man')
run = exper.start_logging()
details = run.get_details()

print("details are:")
print(details)
#run = Run.start_logging(workspace = ws, history_name = 'test')
#run = Run.get_submitted_run()

logger = logger('logs/', 'model.log')
#config = read_params('params.yaml')
#models = config['models']['Regression']

models = ['LinearRegression','GradientBoostingRegressor','RandomForestRegressor']

mlflc = MLflowCallback(
    tracking_uri = ws.get_mlflow_tracking_uri(),
    metric_name='r2_score'
    )
# artifact_path = config['load_data']['raw_data']
# test = [x for x in models]
# print(test)
mlflow.set_experiment(exp_name)

def eval(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    run.log('RMSE',rmse)
    run.log('MAE',mae)
    run.log('R2',r2)
    return rmse, mae, r2


def create_model(trial, max_feature):

    model_type = trial.suggest_categorical("model", [x for x in models])

    logger.info("Trial number: "+str(trial.number))
    logger.info("Model: "+str(model_type))

    if model_type == 'LinearRegression':
        model = LinearRegression()

    if model_type == 'RandomForestRegressor':

        max_depth = trial.suggest_int(
            'max_depth',
            2,
            max_feature
        )

        n_estimators = trial.suggest_int(
            'n_estimators',
            10,
            200
        )

        logger.info("Max_depth: "+str(max_depth))
        logger.info("n_estimators: "+str(n_estimators))

        model = RandomForestRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators
        )

    if model_type == 'GradientBoostingRegressor':

        max_depth = trial.suggest_int(
            'max_depth',
            2,
            max_feature
        )

        n_estimators = trial.suggest_int(
            'n_estimators',
            10,
            200
        )

        logger.info("Max_depth: "+str(max_depth))
        logger.info("n_estimators: "+str(n_estimators))

        model = GradientBoostingRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators
        )
    # mlflow.sklearn.log_model(model,model_type)
    return model

def objective_with_args(X_train, X_test, y_train, y_test):

    @mlflc.track_in_mlflow()
    def objective(trial):
        # mlflow.sklearn.autolog()
        model = create_model(trial, X_train.shape[1])
        model.fit(X_train, y_train)
        # live.log(trial,model.score(X_test,y_test))
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval(y_pred, y_test)

        logger.info("Root mean square error: "+str(rmse))
        logger.info("Mean absolute error: "+str(mae))
        logger.info("R2 score: "+str(r2))
        logger.info("################################")
        mlflow.log_metric('RMSE', rmse)
        mlflow.log_metric('MAE', mae)
        mlflow.sklearn.log_model(model, "model")
        return model.score(X_test, y_test)
    return objective


def build_model(X_train, X_test, y_train, y_test):
    logger.info("In build_model")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(study_name=details['runId'], direction='maximize')
    study.optimize(objective_with_args(X_train, X_test,
                   y_train, y_test), n_trials=20, callbacks=[mlflc])
    print("Number of trials : {}".format(len(study.trials)))
    # study.
    '''
    print("best trial:",study.best_trial)
    res = study.best_trial
    print("Value: {}".format(res.value))

    print("params:")
    for k,v in res.params.items():
        print("    {}:{}".format(k,v))
    '''
    res = study.best_trial
    #run.complete()
    return (create_model(res, X_train.shape[1]), res.params)
