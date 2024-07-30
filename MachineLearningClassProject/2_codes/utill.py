
import torch
import numpy as np
import pickle
import pandas as pd
import scipy.constants as cs
import numpy as np
from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms

def gradient_penalty(critic,real, fake,cond, device = 'cpu'):
    BATCH_SIZE, C, H, W= real.shape
    epsilon = torch.rand((BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    #BATCH_SIZE, input_size = real.shape
    #epsilon = torch.rand((BATCH_SIZE,1)).repeat(1, input_size).to(device)

    interpolated_images = real*epsilon + fake * (1-epsilon)
    
    # caculate critic scores 
    mixed_scores = critic(interpolated_images, cond)
    gradient = torch.autograd.grad(
        inputs = interpolated_images,
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph = True,
        )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

    
def load_fashion_mnist(batch_size=64):
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transform)

    # Create validation set from training set, given that torchvision only gives
    # us the train/test split.  
    num_train = len(train_dataset)
    indices = list(range(num_train))
    
    random_seed = 123
    np.random.seed(random_seed)  # think: why do we do this?
    np.random.shuffle(indices)

    train_idx = indices
    train_sampler = data.SubsetRandomSampler(train_idx)
    test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                         download=True, transform=transform)
    
    train_dataset.data = torch.concat([train_dataset.data, test_dataset.data], dim = 0)
    
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False)
    
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    
    
    return train_loader,  test_loader #valid_loader,

def embedder(x, l=10):

    if np.isscalar(x) is True or x.dim() == 0:
        x = torch.tensor([x])
        isscalar = True
    else:
        isscalar = False

    #x = torch.tensor(x)
    device = x.device

    l_0 = l // 2
    L = 2 * torch.arange(l_0) / l
    L = L.to(device)
    r_1 = torch.sin(x / (100 ** L[:, None])).T
    r_2 = torch.cos(x / (100 ** L[:, None])).T
    #r_total = torch.concat([r_1[:, :, None], r_2[:, :, None]], dim=-1)
    r_total = torch.concat([r_1, r_2], dim=-1)
    if isscalar is True:
        return r_total.reshape(-1)
    else:
        return r_total.reshape(r_1.shape[0], -1)
