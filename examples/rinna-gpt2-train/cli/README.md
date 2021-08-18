# Azure ML CLI 2.0 (Preview)

Execute Job on Azure ML CLI 2.0 (Preview). jobs are authored in YAML format.

## Installation Azure ML CLI 2.0 (Preview)

Install Azure ML CLI 2.0 (Preview)

```bash
./setup.sh
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

