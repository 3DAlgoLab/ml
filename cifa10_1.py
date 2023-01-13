#%%
from torchvision import datasets
from pathlib import Path

data_path = "./data/cifar10"

Path(data_path).mkdir(parents=True, exist_ok=True)

cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)
type(cifar10).__mro__

#%%
len(cifar10)
# %%
len(cifar10_val)
# %%
from torchvision import transforms


class_names = "airplane automobile bird cat deer dog frog horse ship truck".split()
print(class_names)


to_tensor = transforms.ToTensor()
img, label = cifar10[55]
print(class_names[label])
img_t = to_tensor(img)
img_t.shape
# %%
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
import matplotlib.pyplot as plt

plt.imshow(img)

# %%
