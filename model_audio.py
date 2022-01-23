import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gaussian_noise(tensor, mean=0, stddev=0.1):
	noise = torch.nn.init.normal(torch.Tensor(tensor.size()), 0, 0.1)
	noise = noise.to(device)
	return tensor + noise


class PrepNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.p1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU())

		self.p2 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,3), padding=1),
            nn.ReLU())

		self.p3 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=(1,5), padding=(1,2)),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,5), padding=(1,2)),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,5), padding=(1,2)),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=(1,5), padding=(1,2)),
            nn.ReLU())

		self.p4 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(9,3), padding=(0,1)),
			nn.ReLU())

		self.p5 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(1,3), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(11,3), padding=(0, 1)),
			nn.ReLU())

		self.p6 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(9,5), padding=(0, 2)),
			nn.ReLU())

	def forward(self, x):
		p1 = self.p1(x)
		p2 = self.p2(x)
		p3 = self.p3(x)

		x = torch.cat((p1, p2, p3), 1)

		p4 = self.p4(x)
		p5 = self.p5(x)
		p6 = self.p6(x)
		x = p4 + p5 + p6
		# x = torch.cat((p4, p5, p6), 1)
		x = x / 3

		return x


class HidingNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.h1 = nn.Sequential(
			nn.Conv2d(51, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU())

		self.h2 = nn.Sequential(
			nn.Conv2d(51, 50, kernel_size=(1, 4), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=(1, 2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=(1, 2)),
			nn.ReLU())

		self.h3 = nn.Sequential(
			nn.Conv2d(51, 50, kernel_size=(1, 5), padding=(1, 2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1, 2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1, 2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1, 2)),
			nn.ReLU())

		self.h4 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(9, 3), padding=(0, 1)),
			nn.ReLU())

		self.h5 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(1, 4), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(11, 4), padding=(0, 2)),
			nn.ReLU())

		self.h6 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(9, 5) ,padding=(0, 2)),
			nn.ReLU())

		self.h7 = nn.Sequential(
			nn.Conv2d(150, 1, kernel_size=(1, 1), padding=(0, 0)))

	def forward(self, x):
		h1 = self.h1(x)
		h2 = self.h2(x)
		h3 = self.h3(x)

		x = torch.cat((h1, h2, h3), 1)

		h4 = self.h4(x)
		h5 = self.h5(x)
		h6 = self.h6(x)

		x = torch.cat((h4, h5, h6), 1)
		x = self.h7(x)
		x_n = gaussian_noise(x.data, 0, 0.1)
		return x, x_n

class RevealNet(nn.Module):
	def __init__(self):
		super(RevealNet, self).__init__()
		self.r1 = nn.Sequential(
			nn.Conv2d(1, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 3), padding=1),
			nn.ReLU())
		self.r2 = nn.Sequential(
			nn.Conv2d(1, 50, kernel_size=(1, 4), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=(1, 2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 4), padding=(1, 2)),
			nn.ReLU())
		self.r3 = nn.Sequential(
			nn.Conv2d(1, 50, kernel_size=(1, 5), padding=(1,2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1,2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1,2)),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(1, 5), padding=(1,2)),
			nn.ReLU())
		self.r4 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(11, 3), padding=1),
			nn.ReLU())
		self.r5 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(1, 4), padding=1),
			nn.ReLU(),
			nn.Conv2d(50, 50, kernel_size=(11, 4), padding=(0, 2)),
			nn.ReLU())
		self.r6 = nn.Sequential(
			nn.Conv2d(150, 50, kernel_size=(9, 5), padding=(0, 2)),
			nn.ReLU())
		self.r7 = nn.Sequential(
			nn.Conv2d(150, 1, kernel_size=1, padding=0))

	def forward(self, x):
		r1 = self.r1(x)
		r2 = self.r2(x)
		r3 = self.r3(x)
		x = torch.cat((r1, r2, r3), 1)
		r4 = self.r4(x)
		r5 = self.r5(x)
		r6 = self.r6(x)
		x = torch.cat((r4, r5, r6), 1)
		# x = self.finalR(x)
		x= self.r7(x)
		return x

class StegNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.s1 = PrepNet().to(device)
		self.s2 = HidingNet().to(device)
		self.s3 = RevealNet().to(device)

	def forward(self, secret, cover):
		x1 = self.s1(secret)
		x = torch.cat((x1, cover), 1)
		x2, x2_n = self.s2(x)
		x3 = self.s3(x2_n)

		return x2, x3


prepnet = PrepNet().to(device)
hidingnet = HidingNet().to(device)
revealnet = RevealNet().to(device)

print("******************** PrepNet ********************")
summary(prepnet, (1, 1, 44100))
#
print("******************** HidingNet ********************")
summary(hidingnet, (51, 1, 44100))

print("******************** RevealNet ********************")
summary(revealnet, (1, 1, 44100))


# x1 = torch.zeros([3,299,299])
# x2 = torch.ones([3,299,299])
# y = torch.cat((x1,x2),1)
# print(y.shape)