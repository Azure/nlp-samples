#!/bin/sh
az extension add -n ml -y
az group create -n "azureml" -l "japaneast"
az configure --defaults group="azureml" workspace="azureml"
az ml workspace create