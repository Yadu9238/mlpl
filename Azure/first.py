
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core import Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import ServicePrincipalAuthentication

subscription_id = 'your-id'
resource_group = 'your-id'
workspace_name = 'your-id'


#ws = Workspace.from_config()


from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment, ScriptRunConfig
from azureml.widgets import RunDetails
env = Environment('Pyt')

sp = ServicePrincipalAuthentication(
        tenant_id= "id",
        service_principal_id= "pid",
        service_principal_password= "pwd",
    )
ws = Workspace.get( workspace_name,auth = sp,subscription_id = subscription_id, resource_group = resource_group)
#env.docker.enabled = True
print("ready, ",ws.name)

cluster_name = 'cluster-name'

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
				arguments = ['--name','dataset name']) 

#env = Environment(name = 'second')
env.python.user_managed_dependencies = False
#env.docker.enabled = True
script_config.run_config.environment = env

exp_name = 'Exp name'
exper = Experiment(workspace = ws,name = exp_name)
run = exper.submit(config = script_config)
run.wait_for_completion(show_output = True)
