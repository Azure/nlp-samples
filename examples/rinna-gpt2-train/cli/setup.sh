#!/bin/sh
az extension add -n ml -y
az configure --defaults group="azureml-mlops" workspace="azureml-mlops"