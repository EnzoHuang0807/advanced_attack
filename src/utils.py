import os
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_get_model


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError
        

def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    mean = [0.5014, 0.4793, 0.4339]
    std = [0.1998, 0.1963, 0.2025]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def load_model(model_name, device, targeted=True):
        
        def load_single_model(model_name):

            model = ptcv_get_model(model_name, pretrained=True)
            if targeted:
                return wrap_model(model.eval().to(device))
            else:
                return model.eval().to(device)

        if isinstance(model_name, list):
            return EnsembleModel([load_single_model(name) for name in model_name])
        else:
            return load_single_model(model_name)
        
  
def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))


def save_universal_image(uap, filename):
    adversary= (uap.detach().permute((1,2,0)).cpu().numpy() * 255 + 12).astype(np.uint8)
    Image.fromarray(adversary).save(filename)
        
