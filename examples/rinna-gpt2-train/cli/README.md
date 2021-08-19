# Azure ML CLI 2.0 (Preview)

Execute Job on Azure ML CLI 2.0 (Preview). jobs are authored in YAML format.

## Login to Azure

login to Azure

```bash
az login
```

## Installation Azure ML CLI 2.0 (Preview)

Install Azure ML CLI 2.0 (Preview)

```bash
./setup.sh
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
./train.sh  # Fine Tune using HaggingFace Transformers Trainer API
```

