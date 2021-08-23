#!/usr/bin/env python
# coding: utf-8

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', default="azureml-mlops")
    parser.add_argument('--model', default='test-model')
    parser.add_argument('--version', type=int, default=37)
    parser.add_argument('--save', default='rinna-GPT2-quantized-model')
    args = parser.parse_args()

    # キャッシュ保存用のディレクトリを用意
    import os
    cache_dir = os.path.join(".", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Azure ML Workspace への接続
    from azureml.core import Experiment, Workspace, Environment
    from azureml.core.compute import ComputeTarget
    from azureml.core import ScriptRunConfig
    from azureml.core.runconfig import PyTorchConfiguration
    from azureml.core.authentication import InteractiveLoginAuthentication
    import mlflow
    import json
    from azureml.core.authentication import ServicePrincipalAuthentication
    
    azure_credentials = os.environ.get("AZURE_CREDENTIALS", default="{}")
    cred = json.loads(azure_credentials)

    sp = ServicePrincipalAuthentication(
        tenant_id=cred.tenantId,
        service_principal_id=cred.clientId,
        service_principal_password=cred.clientSecret
    )
    ws = Workspace.get(name=args.workspace, auth=sp, subscription_id=cred.subscriptionId)

    # AML 上で実行する場合は上記2行コメントアウト、以下実行
    # ws = Workspace.from_config()

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


    # ## モデル読み込み

    client = mlflow.tracking.MlflowClient()
    if args.version:
        registered_model = client.get_model_version(name=args.model, version=args.version)
    else:
        registered_model = client.get_latest_versions(name=args.model)
    client.download_artifacts(registered_model.run_id, 'outputs/models', cache_dir)

    # GPT-2 モデルにビームサーチを組み合わせるヘルパー class で読み込んだ GPT-2 モデルをラップ
    from onnxruntime.transformers.gpt2_beamsearch_helper import Gpt2BeamSearchHelper, GPT2LMHeadModel_BeamSearchStep
    from transformers import AutoConfig
    import torch

    model_name_or_path = os.path.join(cache_dir, 'outputs/models')
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=model_name_or_path)
    model = GPT2LMHeadModel_BeamSearchStep.from_pretrained(model_name_or_path, config=config, batch_size=1, beam_size=4, cache_dir=cache_dir)
    device = torch.device("cpu")
    model.eval().to(device)


    # 推論で使う関数用にモデルの情報を取得
    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd
    num_layer = model.config.n_layer


    # tokenizer を読み込み
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium", cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.do_lower_case = True


    # ## PyTorch GPT-2 モデルをビームサーチの1ステップを含む ONNX に変換 


    # ONNX に変換
    onnx_model_path = os.path.join(cache_dir, "rinna_gpt2_beam_step_search.onnx")

    if not os.path.exists(onnx_model_path):
        Gpt2BeamSearchHelper.export_onnx(model, device, onnx_model_path) # add parameter use_external_data_format=True when model size > 2 GB
    else:
        print("GPT-2 ONNX model exists.")


    # 最適化と量子化
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
    from onnxruntime.transformers.quantize_helper import QuantizeHelper

    optimized_model_path = os.path.join(cache_dir, "rinna_gpt2_beam_step_search_optimized.onnx")
    quantized_model_path = os.path.join(cache_dir, "rinna_gpt2_beam_step_search_optimized_int8.onnx")

    if not os.path.exists(optimized_model_path):
        Gpt2Helper.optimize_onnx(onnx_model_path, optimized_model_path, False, model.config.num_attention_heads, model.config.hidden_size)
    else:
        print("Optimized GPT-2 ONNX model exists.")

    if not os.path.exists(quantized_model_path):   
        QuantizeHelper.quantize_onnx_model(optimized_model_path, quantized_model_path)
    else:
        print("Quantized GPT-2 Int8 ONNX model exists.")

    # 量子化した ONNX をモデルとして登録
    mlflow.set_experiment('register_onnx')
    with mlflow.start_run() as run:
        remote_model_path = os.path.join('outputs','onnx', "rinna_gpt2_beam_step_search_optimized_int8.onnx")
        mlflow.log_artifact(quantized_model_path, remote_model_path)
        model_uri = "runs:/{}/".format(run.info.run_id) + remote_model_path
        mlflow.register_model(model_uri, args.save)
