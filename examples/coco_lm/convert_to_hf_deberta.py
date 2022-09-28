import torch

model_input_path = "multirun/2022-09-28/06-13-12/0/checkpoints/checkpoint_last.pt"
model_output_path = "coco-lm-deberta-base/pytorch_model.bin"

print(f"Loading fairseq state dictionary")
fairseq_sd = torch.load(model_input_path, map_location=torch.device('cpu'))['model']

print(f"Mapping fairseq state dict to hf state dict")
hf_sd = {}
for k, v in fairseq_sd.items():
    if "encoder.sentence_encoder" in k:
        hf_sd[k[25:]] = v

torch.save(hf_sd, model_output_path)