import torch
import os

# List your .pth files here
pth_files = [
    'final_model_[0,0]no_LoRA1747411531.pth',
    'final_model_[0,25]1747411689.pth',
    'final_model_[25,45]1747411827.pth',
    'final_model_[0,45]1747412154.pth',
    
]

def get_param_size(pth_file):
    state_dict = torch.load(pth_file, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    total_params = sum(p.numel() if hasattr(p, 'numel') else torch.tensor(p).numel() for p in state_dict.values())
    return total_params

param_sizes = {}
for file in pth_files:
    if os.path.exists(file):
        param_sizes[file] = get_param_size(file)
    else:
        print(f"File not found: {file}")

for file, size in param_sizes.items():
    print(f"{file}: {size} parameters")

if len(set(param_sizes.values())) == 1:
    print("All files have the same number of parameters.")
else:
    print("Parameter sizes differ between files.")