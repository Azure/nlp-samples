#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rg', default='mlops')
    parser.add_argument('--workspace', default="azureml-mlops")
    parser.add_argument('--model', default='test-model')
    parser.add_argument('--cluster', default='aml-cluster')
    parser.add_argument('--name', default='rinna-gpt2-aks-mlops')
    args = parser.parse_args()


    # # ONNX に変換したモデルを Kubernetes 上にデプロイ
    import os
    from azureml.core import Workspace
    from azureml.core.conda_dependencies import CondaDependencies 
    from azureml.core.model import InferenceConfig
    from azureml.core.environment import Environment
    from azureml.core.webservice import AksWebservice
    from azureml.core.compute.aks import AksCompute 
    from azureml.core.model import Model
    import random
    import numpy as np
    import torch
    import json
    import requests

    from azureml.core.authentication import ServicePrincipalAuthentication

    print('Connecting Azure ML Workspace')
    azure_credentials = os.environ.get("AZURE_CREDENTIALS", default="{}")
    cred = json.loads(azure_credentials)

    sp = ServicePrincipalAuthentication(
        tenant_id=cred["tenantId"],
        service_principal_id=cred["clientId"],
        service_principal_password=cred["clientSecret"]
    )
    ws = Workspace.get(name=args.workspace, auth=sp, subscription_id=cred["subscriptionId"], resource_group=args.rg)
    

    # 推論環境の定義ファイル生成と環境設定

    env_file_path = os.path.join("environment.yml")
    score_file_path = os.path.join("score.py")

    env = Environment.from_conda_specification(name="rinna-predict-env", file_path=env_file_path)
    env.register(ws)
    inference_config = InferenceConfig(entry_script=score_file_path, environment=env)

    # デプロイ設定
    deploy_config = AksWebservice.deploy_configuration(
        cpu_cores = 1,
        memory_gb = 4,
        tags = {'framework': 'onnx'},
        auth_enabled = False,
        description = 'rinna gpt-2'
    )

    target_aks = AksCompute(ws, args.cluster)

    # モデル指定
    model = Model(ws, args.model)

    # デプロイ
    print('Deploy ONNX model as API on AKS')
    service_name = args.name
    print("Service", service_name)
    webservice = Model.deploy(ws, service_name, [model], inference_config, deploy_config, target_aks)
    webservice.wait_for_deployment(True)
    print(webservice.state)

    # 推論
    endpoint = webservice.scoring_uri
    input_data = json.dumps({'data': "機械学習"})
    res = requests.post(url=endpoint, data=input_data, headers={'Content-Type': 'application/json'})
    print('Test API prediction')
    res.json()
