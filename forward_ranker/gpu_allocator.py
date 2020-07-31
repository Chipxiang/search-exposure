from load_data import obj_reader
from load_data import obj_writer
import os
import torch


def select_device(gpu_root):
    argmin = -1
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if not os.path.exists(gpu_root):
            obj_writer([0, 0, 0, 0], gpu_root)
        gpu_usage_list = obj_reader(gpu_root)
        min = 100000
        argmin = 0
        for i, count in enumerate(gpu_usage_list):
            if count < min:
                argmin = i
                min = count
        gpu_usage_list[argmin] += 1
        device = torch.device("cuda:" + str(argmin))
        obj_writer(gpu_usage_list, gpu_root)
    return argmin, device


def cleanup_gpu_list(current_gpu, gpu_root):
    gpu_usage_list = obj_reader(gpu_root)
    gpu_usage_list[current_gpu] -= 1
    obj_writer(gpu_usage_list, gpu_root)
