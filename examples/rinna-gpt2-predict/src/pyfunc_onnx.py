# InferenceSession を含む mlflow.pyfunc 対応の class
import mlflow.pyfunc

class ONNXrinnaGPT2(mlflow.pyfunc.PythonModel):
    def __init__(self, gpt2_onnx_path, num_tokens_to_produce = 30, beam_size=4, use_onnxruntime_io=False, use_cpu=True):
        import os
        import torch
        self.cache_dir = os.path.join(".", "cache_models")
        self.onnx_bytes = self.get_onnx_bytes(gpt2_onnx_path)
        # self で InferenceSession を持つと mlflow.pyfunc.save_model の pickle でこける
        # https://github.com/microsoft/onnxruntime/issues/643
        # https://github.com/microsoft/onnxruntime/pull/800
        self.ort_session = None # 内部的にSessionを保持する場合に使用
        self.model_name_or_path = "rinna/japanese-gpt2-medium"
        self.tokenizer = self.get_tokenizer()
        self.num_layer = 24
        self.num_attention_heads = 16
        self.hidden_size = 1024

        self.num_tokens_to_produce = num_tokens_to_produce
        self.beam_size = beam_size
        self.use_onnxruntime_io = use_onnxruntime_io
        if use_cpu:
            self.device =  torch.device("cpu")
        else:    
            self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_onnx_bytes(self, path):
        with open(path, 'rb') as f:
            return f.read()
     
        
    def get_tokenizer(self):
        from transformers import T5Tokenizer
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.do_lower_case = True
        #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def get_example_inputs(self, prompt_text):
        import torch 
        encodings_dict = self.tokenizer.batch_encode_plus(prompt_text, padding=True)

        input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
        attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(position_ids < 0, 0)

        #Empty Past State for generating first word
        empty_past = []
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        past_shape = [2, batch_size, self.num_attention_heads, 0, self.hidden_size // self.num_attention_heads]
        for i in range(self.num_layer):
            empty_past.append(torch.empty(past_shape).type(torch.float32).to(self.device))

        return input_ids, attention_mask, position_ids, empty_past

    def inference_with_io_binding(self, session, config, input_ids, position_ids, attention_mask, past, beam_select_idx, input_log_probs, input_unfinished_sents, prev_step_results, prev_step_scores, step, context_len):
        from onnxruntime.transformers.gpt2_beamsearch_helper import Gpt2BeamSearchHelper, GPT2LMHeadModel_BeamSearchStep
        output_shapes = Gpt2BeamSearchHelper.get_output_shapes(batch_size=1,
                                                               context_len=context_len,
                                                               past_sequence_length=past[0].size(3),
                                                               sequence_length=input_ids.size(1),
                                                               beam_size=self.beam_size,
                                                               step=step,
                                                               config=config,
                                                               model_class="GPT2LMHeadModel_BeamSearchStep")
        output_buffers = Gpt2BeamSearchHelper.get_output_buffers(output_shapes, self.device)

        io_binding = Gpt2BeamSearchHelper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past, output_buffers, output_shapes, beam_select_idx, input_log_probs, input_unfinished_sents, prev_step_results, prev_step_scores)
        session.run_with_iobinding(io_binding)

        outputs = Gpt2BeamSearchHelper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes, return_numpy=False)
        return outputs


    def update(self, output, step, batch_size, beam_size, context_length, prev_attention_mask, device):
        """
        Update the inputs for next inference.
        """
        import numpy
        import torch

        last_state = (torch.from_numpy(output[0]).to(device)
                            if isinstance(output[0], numpy.ndarray) else output[0].clone().detach().cpu())

        input_ids = last_state.view(batch_size * beam_size, -1).to(device)

        input_unfinished_sents_id = -3
        prev_step_results = (torch.from_numpy(output[-2]).to(device) if isinstance(output[-2], numpy.ndarray)
                                    else output[-2].clone().detach().to(device))
        position_ids = (torch.tensor([context_length + step - 1
                                            ]).unsqueeze(0).repeat(batch_size * beam_size, 1).to(device))

        if prev_attention_mask.shape[0] != (batch_size * beam_size):
            prev_attention_mask = prev_attention_mask.repeat(batch_size * beam_size, 1)
        attention_mask = torch.cat(
            [
                prev_attention_mask,
                torch.ones([batch_size * beam_size, 1]).type_as(prev_attention_mask),
            ],
            1,
        ).to(device)

        beam_select_idx = (torch.from_numpy(output[input_unfinished_sents_id - 2]).to(device) if isinstance(
            output[input_unfinished_sents_id - 2], numpy.ndarray) else output[input_unfinished_sents_id - 2].clone().detach().to(device))
        input_log_probs = (torch.from_numpy(output[input_unfinished_sents_id - 1]).to(device) if isinstance(
            output[input_unfinished_sents_id - 1], numpy.ndarray) else output[input_unfinished_sents_id - 1].clone().detach().to(device))
        input_unfinished_sents = (torch.from_numpy(output[input_unfinished_sents_id]).to(device) if isinstance(
            output[input_unfinished_sents_id], numpy.ndarray) else
                                        output[input_unfinished_sents_id].clone().detach().to(device))
        prev_step_scores = (torch.from_numpy(output[-1]).to(device)
                                    if isinstance(output[-1], numpy.ndarray) else output[-1].clone().detach().to(device))

        past = []
        if isinstance(output[1], tuple):  # past in torch output is tuple
            past = list(output[1])
        else:
            for i in range(self.num_layer):
                past_i = (torch.from_numpy(output[i + 1])
                            if isinstance(output[i + 1], numpy.ndarray) else output[i + 1].clone().detach())
                past.append(past_i.to(device)) 

        inputs = {
            'input_ids': input_ids,
            'attention_mask' : attention_mask,
            'position_ids': position_ids,
            'beam_select_idx': beam_select_idx,
            'input_log_probs': input_log_probs,
            'input_unfinished_sents': input_unfinished_sents,
            'prev_step_results': prev_step_results,
            'prev_step_scores': prev_step_scores,
        }
        ort_inputs = {
            'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
            'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),
            'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy()),
            'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
            'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
            'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
            'prev_step_results': numpy.ascontiguousarray(prev_step_results.cpu().numpy()),
            'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
        }
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
    
        return inputs, ort_inputs, past

    def predict(self, input_text):
        import numpy
        import torch
        import onnxruntime
        print("Text generation using", "OnnxRuntime with IO binding" if self.use_onnxruntime_io else "OnnxRuntime", "...")    
        input_ids, attention_mask, position_ids, past = self.get_example_inputs([input_text])
        beam_select_idx = torch.zeros([1, input_ids.shape[0]]).long()
        input_log_probs = torch.zeros([input_ids.shape[0], 1])
        input_unfinished_sents = torch.ones([input_ids.shape[0], 1], dtype=torch.bool)
        prev_step_scores = torch.zeros([input_ids.shape[0], 1])
        inputs = {
            'input_ids': input_ids,
            'attention_mask' : attention_mask,
            'position_ids': position_ids,
            'beam_select_idx': beam_select_idx,
            'input_log_probs': input_log_probs,
            'input_unfinished_sents': input_unfinished_sents,
            'prev_step_results': input_ids,
            'prev_step_scores': prev_step_scores,
        }
        ort_inputs = {
            'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy()),
            'attention_mask' : numpy.ascontiguousarray(attention_mask.cpu().numpy()),
            'position_ids': numpy.ascontiguousarray(position_ids.cpu().numpy()),
            'beam_select_idx': numpy.ascontiguousarray(beam_select_idx.cpu().numpy()),
            'input_log_probs': numpy.ascontiguousarray(input_log_probs.cpu().numpy()),
            'input_unfinished_sents': numpy.ascontiguousarray(input_unfinished_sents.cpu().numpy()),
            'prev_step_results': numpy.ascontiguousarray(input_ids.cpu().numpy()),
            'prev_step_scores': numpy.ascontiguousarray(prev_step_scores.cpu().numpy()),
        }
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past_i.cpu().numpy())
        batch_size = input_ids.size(0)
        beam_size = self.beam_size
        context_length = input_ids.size(-1)
        
        if self.ort_session == None:
            ort_session = onnxruntime.InferenceSession(self.onnx_bytes)
        # self で保持すると mlflow.pyfunc.save_model の cloudpickle でこけるため、都度セッションを作成
            for step in range(self.num_tokens_to_produce):
                if self.use_onnxruntime_io:
                    outputs = self.inference_with_io_binding(ort_session, config, inputs['input_ids'], inputs['position_ids'], inputs['attention_mask'], past, inputs['beam_select_idx'], inputs['input_log_probs'], inputs['input_unfinished_sents'], inputs['prev_step_results'], inputs['prev_step_scores'], step, context_length)
                else:
                    outputs = ort_session.run(None, ort_inputs) 
                inputs, ort_inputs, past = self.update(outputs, step, batch_size, beam_size, context_length, inputs['attention_mask'], self.device)

                if not inputs['input_unfinished_sents'].any():
                    break
        else:
            for step in range(self.num_tokens_to_produce):
                if self.use_onnxruntime_io:
                    outputs = self.inference_with_io_binding(self.ort_session, config, inputs['input_ids'], inputs['position_ids'], inputs['attention_mask'], past, inputs['beam_select_idx'], inputs['input_log_probs'], inputs['input_unfinished_sents'], inputs['prev_step_results'], inputs['prev_step_scores'], step, context_length)
                else:
                    outputs = self.ort_session.run(None, ort_inputs) 
                inputs, ort_inputs, past = self.update(outputs, step, batch_size, beam_size, context_length, inputs['attention_mask'], self.device)

                if not inputs['input_unfinished_sents'].any():
                    break

        predict_sentences = [self.tokenizer.decode(candidate, skip_special_tokens=True) for candidate in inputs['prev_step_results']]
        
        return predict_sentences