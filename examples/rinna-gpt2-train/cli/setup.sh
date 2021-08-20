#!/bin/sh
az extension add -n ml -y
az configure --defaults group="mlops" workspace="azureml-mlops"