import torch
import torch.nn as nn
from pathlib import Path
import math

from models.experimental import attempt_load

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

class TracedModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640,640), g=False): 
        super(TracedModel, self).__init__()
        
        print(" Convert model to Traced-model... ") 
        self.stride = model.stride
        self.names = model.names
        self.model = model

        self.model = revert_sync_batchnorm(self.model)
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        
        rand_example = torch.rand(1, 3, img_size, img_size)
        
        if g:
            save_name = "traced_model_g.pt"
        else:
            save_name = "traced_model.pt"
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        #traced_script_module = torch.jit.script(self.model)
        traced_script_module.save(save_name)
        print(" traced_script_module saved! ")
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)
        print(" model is traced! \n") 

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out

def load_model(weights_path, device, half, g):
    
    model = attempt_load(weights=weights_path, map_location=device)
    stride = int(model.stride.max())  # model stride
    img_size = 1024
    imgsz = check_img_size(img_size, s=stride) 
    
    model = TracedModel(model, device, img_size, g)
    
    if half:
        model.half() 
        
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  
        
    return model, stride

def load_targets_model(target, device, half):
    
    weights = f'./trained_detectors/target_{target}.pt'
    model, stride = load_model(weights, device, half, g=False)
    
    return model, stride

def load_gestures_model(device, half):
    
    weights = f'./trained_detectors/gestures.pt'
    model, stride = load_model(weights, device, half, g=True)
    
    return model, stride