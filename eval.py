import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, utils
import torchvision.transforms as transforms
from model import StegNet
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import skorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def denormalize(image, std, mean):
	for t in range(3):
		image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
	return image


def steg_loss(S_prime, C_prime, S, C, beta):
	loss_cover = F.mse_loss(C_prime, C)
	loss_secret = F.mse_loss(S_prime, S)
	loss = loss_cover + beta*loss_secret
	return loss, loss_cover, loss_secret

def show_img(img):
    try:
        plt.imshow(img)
        plt.show()
    except ValueError:
        print("img size not matched")


cifar10_mean = [0.491, 0.482, 0.446]
cifar10_std = [0.247, 0.243, 0.261]


# cifar10_mean = [0.5, 0.5, 0.5]
# cifar10_std = [0.5, 0.5, 0.5]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])

batch_size = 8
trainset = datasets.CIFAR10(root='../datasets', train=True,transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='../datasets', transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

model = StegNet().to(device)
model.load_state_dict(torch.load("cifar10_models/70_model.pkl"))


for i, data in enumerate(train_loader):
    images, _ = data
    images = images.to(device)
    covers = images[:len(images) // 2]
    # covers = torch.unsqueeze(torch.squeeze(covers),0)
    secrets = images[len(images) // 2:]
    # covers = Variable(covers, requires_grad=False)
    # secrets = Variable(secrets, requires_grad=False)
    denormed_covers = covers[0,:].cpu().detach().numpy().reshape((32,32,3))
    denormed_covers =denormalize(denormed_covers,cifar10_std,cifar10_mean)
    show_img(denormed_covers)

    hidden, output = model(secrets, covers)

    loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, 1)
    for i in range(hidden.shape[0]):
        hidden_img = hidden[i,:]
        output_img = output[i,:]
        print(hidden_img.shape)
            # img = mpimg.imread(hidden_img)
        hidden_img_numpy = hidden_img.cpu().detach().numpy().reshape((32,32,3))
        output_img_numpy = output_img.cpu().detach().numpy()

        # hidden_img_plot = plt.imshow(hidden_img_numpy)
        # plt.show()
        denorm_img = denormalize(output_img_numpy,cifar10_std,cifar10_mean)
        denorm_img = denorm_img.reshape((32,32,3))
        show_img(denorm_img)

