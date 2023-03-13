import torch
import json

print(json.dumps({
    "version": torch.__version__,
    "cuda_is_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "current_device": torch.cuda.current_device(),
    "device_name": torch.cuda.get_device_name(),
    "device_capability": torch.cuda.get_device_capability(),
}, indent=2))
print(torch.cuda.memory_summary())
