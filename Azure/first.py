
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core import Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import ServicePrincipalAuthentication

subscription_id = '046415a5-fb0a-42cf-b7c6-1a5c9616c874'
resource_group = 'RS'
workspace_name = 'MLOps-ws'


#ws = Workspace.from_config()


from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment, ScriptRunConfig
from azureml.widgets import RunDetails
env = Environment('Pyt')

sp = ServicePrincipalAuthentication(
        tenant_id= "372ee9e0-9ce0-4033-a64a-c07073a91ecd",
        service_principal_id= "fdb84aa3-4b1f-4287-8692-9c0fe8ec1f61",
        service_principal_password= "Q5bV9QRfUYXhYkUp8w~AaMWSuaerMESfEn",
    )
ws = Workspace.get( workspace_name,auth = sp,subscription_id = subscription_id, resource_group = resource_group)
#env.docker.enabled = True
print("ready, ",ws.name)

cluster_name = 'mlops-cluster'

cluster = ComputeTarget(workspace = ws,name = cluster_name)
print("found cluster,",cluster)
env.python.conda_dependencies = CondaDependencies.create(
	conda_packages = ['pandas','scikit-learn','numpy'],
	pip_packages = ['joblib','azureml-sdk','mlflow','optuna','azureml-mlflow'],
	pin_sdk_version = False)
#env = Environment.from_pip_requirements(name = 'firstenv',file_path = 'requirements.txt')
script_config = ScriptRunConfig(source_directory='src',
                                script='train.py',
                                compute_target=cluster,
				arguments = ['--name','Cement dataset']) 

#env = Environment(name = 'second')
env.python.user_managed_dependencies = False
#env.docker.enabled = True
script_config.run_config.environment = env

exp_name = 'azure-test-2305'
exper = Experiment(workspace = ws,name = exp_name)
run = exper.submit(config = script_config)
run.wait_for_completion(show_output = True)