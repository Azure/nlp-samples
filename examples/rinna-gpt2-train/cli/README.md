# Azure ML CLI 2.0 (Preview)

Execute Job on Azure ML CLI 2.0 (Preview). jobs are authored in YAML format.

## Login to Azure

login to Azure

```bash
az login
```

check Azure subscription
```bash
az account show -o table # check if you are logged in to a right Azure subscription
az account set --subscription <subscription_id> # change Azure subscription if needed
```

## Installation Azure ML CLI 2.0 (Preview)

Install Azure ML CLI 2.0 (Preview)

```bash
./setup.sh
# Inside the shell, run the following command:
# az extension add -n ml -y   <-- If you use devcontainer, it already installed. Seee Dockerfile under .devcontiner folder.
# az configure --defaults group="azureml-mlops" workspace="azureml-mlops"  <-- change resource group name and workspace name to your environment.
```

## Setup Environment

Create Azure ML Environments based on [Dockerfile](./Dockerfile)

```bash
./asset.sh # create Azure ML Environments based on DockerFile
```


## Training

Fine Tune GPT-2 Model

```bash
# single node (use transformers Trainer API directly)
./train.sh  # Fine Tune rinna GPT-2 Model
```
or 
```bash
# distributed training (use modified run_clm.py)
./train_distributed.sh  # Fine Tune rinna GPT-2 Model
```