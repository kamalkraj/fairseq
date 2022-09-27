import torch

model_input_path = "multirun/2022-09-26/17-12-22/0/checkpoints/checkpoint_6_235000.pt"
model_output_path = "coco-lm-base/pytorch_model.bin"

print(f"Loading fairseq state dictionary")
fairseq_sd = torch.load(model_input_path, map_location=torch.device('cpu'))['model']

print(f"Mapping fairseq state dict to hf state dict")
hf_sd = {}
for k, v in fairseq_sd.items():
    if "encoder.sentence_encoder" in k:
        if k == "encoder.sentence_encoder.embed_tokens.weight":
            hf_sd['embeddings.word_embeddings.weight'] = v
        if k == 'encoder.sentence_encoder.embed_positions.weight':
            hf_sd['embeddings.position_embeddings.weight'] = v
        if k == 'encoder.sentence_encoder.emb_layer_norm.weight':
            hf_sd['embeddings.LayerNorm.weight'] = v
        if k == 'encoder.sentence_encoder.emb_layer_norm.bias':
            hf_sd['embeddings.LayerNorm.bias'] = v
        if k == 'encoder.sentence_encoder.relative_attention_bias.weight':
            hf_sd['relative_attention_bias.weight'] = v
        if "layers" in k:
            layer_num = k.split('.')[3]
            if "self_attn.in_proj.weight" in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.self_attn.in_proj.weight'] = v
            if 'self_attn.in_proj.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.self_attn.in_proj.bias'] = v
            if "self_attn.out_proj.weight" in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.self_attn.out_proj.weight'] = v
            if 'self_attn.out_proj.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.self_attn.out_proj.bias'] = v
            if 'self_attn_layer_norm.weight' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.LayerNorm.weight'] = v
            if 'self_attn_layer_norm.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.attention.LayerNorm.bias'] = v
            if 'fc1.weight' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.intermediate.dense.weight'] = v
            if 'fc1.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.intermediate.dense.bias'] = v
            if 'fc2.weight' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.output.dense.weight'] = v
            if 'fc2.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.output.dense.bias'] = v
            if 'final_layer_norm.weight' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.output.LayerNorm.weight'] = v
            if 'final_layer_norm.bias' in k:
                hf_sd['encoder.layer.'+str(layer_num)+'.output.LayerNorm.bias'] = v

torch.save(hf_sd, model_output_path)