Resource setup:<br>
1.install azure cli, download from website and login
<p align = "center">
<img src = "img/cli.PNG" title = "Azure cli" height = "10%">
  <em>Download the azure cli from the official page</em>
 </p>
2. use ui for creating resource group or cli
<p align = "center">
<img src = "img/rg.PNG" title = "Azure Resource group">
  <em>Create a resource group</em>
 </p>
```
az group create --name <name> --location <loc>
```

3.Create machine learning workspace
Use UI to create ML workspace under resource group created before or the cli 
<p align = "center">
<img src = "img/ml.PNG" title = "Azure Machine Learning workspace">
  <em>Create a Machine learning workspace under the resource group</em>
 </p>
```
az ml workspace create -w <name> -g <group-name>

4.setup blob storage and upload data
<p align = "center">
<img src = "img/storage.PNG" title = "Azure Storage account">
  <em>Create a storage account under the resource group</em>
 </p>
5.create service principal for authentication
```
az ad sp create-for-rbac --name <service-name>
```
<p align = "center">
<img src = "img/add_role1.PNG" title = "Azure IAM">
  <em>Goto IAM of azure machine learning workspace and add role assignment</em>
 </p>
<p align = "center">
<img src = "img/add_role2.PNG" title = "Azure IAM">
  <em>Select role as contributer</em>
 </p>
 <p align = "center">
<img src = "img/add_role3.PNG" title = "Azure IAM">
  <em>Add the service principal created under members</em>
 </p>

6. As of now, data is manually uploaded to azure blob storage.
for retrieving data use Dataset from azureml.core

7. setup env:

 Use CondaDependencies to add the required conda and pip packages.

    ```
    def create_aml_environment(aml_interface):
    aml_env = Environment(name=AML_ENV_NAME)
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package("numpy==1.18.2")
    conda_dep.add_pip_package("pandas==1.0.3")
    conda_dep.add_pip_package("scikit-learn==0.22.2.post1")
    conda_dep.add_pip_package("joblib==0.14.1")
    whl_filepath = retrieve_whl_filepath()
    whl_url = Environment.add_private_pip_wheel(
        workspace=aml_interface.workspace,
        file_path=whl_filepath,
        exist_ok=True
    )
    conda_dep.add_pip_package(whl_url)
    aml_env.python.conda_dependencies = conda_dep
    aml_env.docker.enabled = True
    return aml_env
    ```
 or u can add the packages directly in main driver code

    ```
    env.python.conda_dependencies = CondaDependencies.create(
	conda_packages = ['packages'],
	pip_packages = ['packages'],
	pin_sdk_version = False)
    ```

8. Sample driver code for main file:

```

from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core import Experiment, Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Experiment, ScriptRunConfig
from azureml.widgets import RunDetails
env = Environment('your-env-name')

subscription_id = 'your-sub-id'
resource_group = 'your-rg-name'
workspace_name = 'your-ws-name'

# It is recommended to add the authentication details as env variables
sp = ServicePrincipalAuthentication(
        tenant_id= "your-tenant-id",
        service_principal_id= "your-sp-id",
        service_principal_password= "your-sp-pass",
    )
ws = Workspace.get( workspace_name,auth = sp,subscription_id = subscription_id, resource_group = resource_group)

print("ready, ",ws.name)
cluster_name = 'your-cluster-name'
cluster = ComputeTarget(workspace = ws,name = cluster_name)

print("found cluster,",cluster)

env.python.conda_dependencies = CondaDependencies.create(
	conda_packages = [packages],
	pip_packages = [packages],
	pin_sdk_version = False)

script_config = ScriptRunConfig(source_directory='your-src-dir',
                                script='Main-file-to-run',
                                compute_target=cluster,
				) 


env.python.user_managed_dependencies = False
script_config.run_config.environment = env

exp_name = 'your-exp-name'
exper = Experiment(workspace = ws,name = exp_name)
run = exper.submit(config = script_config)
run.wait_for_completion(show_output = True)

```
