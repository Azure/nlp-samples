#!/bin/sh

# compute
az ml compute create -n gpu-cluster --type AmlCompute  --min-instances 0 --max-instances 4 --size Standard_NC8as_T4_v3

# environment
az ml environment create --file huggingface_ort_env.yml

