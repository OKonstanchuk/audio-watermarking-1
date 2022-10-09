# Констанчук Олеся 09.10.2022
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from torchvision import datasets, utils
import torchvision.transforms as transforms
from model_audio import StegNet
import skorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)

def denormalize(image, std, mean):
	for t in range(3):
		image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
	return image


def steg_loss(S_prime, C_prime, S, C, beta):
	loss_cover = F.mse_loss(C_prime, C)
	loss_secret = F.mse_loss(S_prime, S)
	loss = loss_cover + beta*loss_secret
	return loss, loss_cover, loss_secret


# load examples
audios = []
with open("audio_examples.pkl",'rb') as f:
    while True:
        try:
            x = pickle.load(f)
            audios.append(x)
        except EOFError:
            break

train_examples = 5000
samplerate = 44100

# generate watermarkings
watermarkings = []
watermarking_num = 10
for i in range(watermarking_num):
	mark = np.random.randint(2, size=samplerate)
	watermarkings.append(mark)

# random order for watermarkings
mark_order = np.random.randint(watermarking_num,size=train_examples)
# #check number distribution
# num_dict = {}
# for i in range(10):
# 	num_sum = 0
# 	for order_num in mark_order:
# 		if i == order_num:
# 			num_sum += 1
# 	num_dict[i] = num_sum
# print(num_dict)
# {0: 505, 1: 523, 2: 535, 3: 496, 4: 487, 5: 474, 6: 468, 7: 533, 8: 492, 9: 487}

# custum dataset
audio_mark_tuple = []
for i,a in enumerate(audios[:train_examples]):
	watermarking_order = mark_order[i]
	audio_mark = (a, watermarkings[watermarking_order])
	audio_mark_tuple.append(audio_mark)
# print(audio_mark_tuple[0])


class AudioMarking(Dataset):
	def __init__(self, tuple_data):
		self.tuple_data = tuple_data

	def __len__(self):
		return len(self.tuple_data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		audio = self.tuple_data[idx][0]
		audio = torch.tensor(audio)

		mark = self.tuple_data[idx][1]
		mark = torch.tensor(mark)
		return audio,mark

train_set = AudioMarking(audio_mark_tuple)
train_loader = DataLoader(train_set)
model = StegNet().to(device)
optimizer = optim.Adam(model.parameters())
beta = 1
epoch = 100
train_loss=[]
losses=[]
for ep in range(epoch):
	train_loss = []
	for i, data in enumerate(train_loader):
		audio,mark = data
		audio = audio.type(torch.FloatTensor)
		mark = mark.type(torch.FloatTensor)
		audio = audio.to(device)
		mark = mark.to(device)
		optimizer.zero_grad()
		hidden, output = model(mark.unsqueeze(0).unsqueeze(0), audio.unsqueeze(0).unsqueeze(0))
		loss, loss_cover, loss_secret = steg_loss(output, hidden, mark, audio, beta)
		loss.backward()
		optimizer.step()

		train_loss.append(loss.item())
		losses.append(loss.item())
		# print(loss.item(), loss_cover.item(), loss_secret.item())

	print('Epoch: {:d} || Train Loss: {:.4f}, cover_error: {:.4f}, secret_error: {:.4f}'.
		  format(ep, loss.item(), loss_cover.item(), loss_secret.item()))
	torch.save(model.state_dict(), str(ep) + "audio_mark_model.pkl")
	avg_train_loss = np.mean(train_loss)
	print(avg_train_loss)






#
#
# beta = 1
# learning_rate = .0001
# # train(train_loader, beta, learning_rate)
# model = StegNet().to(device)
# optimizer = optim.Adam(model.parameters())
#
# losses = []
# for ep in range(epoch):
# 	train_loss = []
# 	for i, data in enumerate(train_loader):
# 		images, _ = data
# 		images = images.to(device)
# 		covers = images[:len(images) // 2]
# 		secrets = images[len(images)//2:]
# 		# covers = Variable(covers, requires_grad=False)
# 		# secrets = Variable(secrets, requires_grad=False)
#
# 		optimizer.zero_grad()
# 		hidden, output = model(secrets, covers)
# 		loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, beta)
# 		loss.backward()
# 		optimizer.step()
#
# 		train_loss.append(loss.item())
# 		losses.append(loss.item())
# 		# print(loss.item(), loss_cover.item(), loss_secret.item())
#
# 	print('Epoch: {:d} || Train Loss: {:.4f}, cover_error: {:.4f}, secret_error: {:.4f}'.
# 		  format(ep, loss.item(), loss_cover.item(), loss_secret.item()))
# 	torch.save(model.state_dict(), str(ep) + "_model.pkl")
# 	avg_train_loss = np.mean(train_loss)
#



