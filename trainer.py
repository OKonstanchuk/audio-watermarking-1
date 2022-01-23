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

cifar10_mean = [0.491, 0.482, 0.446]
cifar10_std = [0.247, 0.243, 0.261]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
trainset = datasets.CIFAR10(root='../datasets', train=True,download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='../datasets',download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
# train_loader = torch.utils.DataLoader(
# 	datasets.ImageFolder(
# 		TRAIN_PATH,
# 		transforms.Compose([
# 			transforms.Scale(256),
# 			transforms.RandomCrop(224),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean=mean, std=std)])),
# 	batch_size=10, pin_memory=True, num_workers=1,
# 	shuffle=True, drop_last=True)
#
# test_loader = torch.utils.DataLoader(
# 	datasets.ImageFolder(
# 		TEST_PATH,
# 		transforms.Compose([
# 			transforms.Scale(256),
# 			transforms.RandomCrop(224),
# 			transforms.ToTensor(),
# 			transforms.Normalize(mean=mean, std=std)])),
# 	batch_size=5, pin_memory=True, num_workers=1,
# 	shuffle=True, drop_last=True)

#
# def train(train_loader, beta, lr):
#
# 	losses = []
# 	for epoch in range(epochs):
# 		model.train()
# 		train_loss = []
#
# 		for i, data in enumerate(train_loader):
#
# 			images, _ = data
#
# 			covers = images[:len(images)//2]
# 			secrets = images[len(images)//2:]
# 			covers = Variable(covers, requires_grad=False)
# 			secrets = Variable(secrets, requires_grad=False)
#
# 			optimizer.zero_grad()
# 			hidden, output = model(secrets, covers)
#
# 			loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, beta)
# 			loss.backward()
# 			optimizer.step()
#
# 			train_loss.append(loss.data[0])
# 			losses.append(loss.data[0])
#
# 			torch.save(model.state_dict(), MODEL_PATH+'.pkl')
# 			avg_train_loss = np.mean(train_loss)
# 			print('Train Loss {1:.4f}, cover_error {2:.4f}, secret_error{3:.4f}'. format(loss.data[0], loss_cover.data[0], loss_secret.data[0]))
# 		print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
#             epoch+1, epochs, avg_train_loss))
#
# 		return model, avg_train_loss, losses

epoch = 100
beta = 1
learning_rate = .0001
# train(train_loader, beta, learning_rate)
model = StegNet().to(device)
optimizer = optim.Adam(model.parameters())

losses = []
for ep in range(epoch):
	train_loss = []
	for i, data in enumerate(train_loader):
		images, _ = data
		images = images.to(device)
		covers = images[:len(images) // 2]
		secrets = images[len(images)//2:]
		# covers = Variable(covers, requires_grad=False)
		# secrets = Variable(secrets, requires_grad=False)

		optimizer.zero_grad()
		hidden, output = model(secrets, covers)
		loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, beta)
		loss.backward()
		optimizer.step()

		train_loss.append(loss.item())
		losses.append(loss.item())
		# print(loss.item(), loss_cover.item(), loss_secret.item())

	print('Epoch: {:d} || Train Loss: {:.4f}, cover_error: {:.4f}, secret_error: {:.4f}'.
		  format(ep, loss.item(), loss_cover.item(), loss_secret.item()))
	torch.save(model.state_dict(), str(ep) + "_model.pkl")
	avg_train_loss = np.mean(train_loss)




