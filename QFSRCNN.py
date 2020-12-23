from torch import nn
from torch import quantization
class FSRCNN(nn.Module):
	def __init__(self, scale_factor, num_channels = 1, d = 56, s = 12, m = 4):
		super(FSRCNN, self).__init__()

		self.quant = quantization.QuantStub()
		self.dequant = quantization.DeQuantStub()

		self.l1 = nn.Conv2d(num_channels, d, kernel_size = 5, padding = 5//2)
		self.l2 = nn.PReLU(d)

		self.l3 = nn.Conv2d(d, s, kernel_size = 1)
		self.l4 = nn.PReLU(s)
		self.l5 = nn.Conv2d(s, s, kernel_size = 3, padding = 3//2)#1
		self.l6 = nn.PReLU(s)
		self.l7 = nn.Conv2d(s, s, kernel_size = 3, padding = 3//2)#2
		self.l8 = nn.PReLU(s)
		self.l9 = nn.Conv2d(s, s, kernel_size = 3, padding = 3//2)#3
		self.l10 = nn.PReLU(s)
		self.l11 = nn.Conv2d(s, s, kernel_size = 3, padding = 3//2)#4
		self.l12 = nn.PReLU(s)
		self.l13 = nn.Conv2d(s, d, kernel_size = 1)
		self.l14 = nn.PReLU(s)

		self.l15 = nn.ConvTranspose2d(d, num_channels, kernel_size = 9, stride = scale_factor, padding = 9//2, output_padding = scale_factor - 1)

	def forward(self, x):
		x = self.quant(x)
		x = self.l1(self.quant(x))#conv
		x = self.l2(self.dequant(x))#prelu
		x = self.l3(self.quant(x))#c
		x = self.l4(self.dequant(x))#p
		x = self.l5(self.quant(x))#c
		x = self.l6(self.dequant(x))#p
		x = self.l7(self.quant(x))#c
		x = self.l8(self.dequant(x))#p
		x = self.l9(self.quant(x))#c
		x = self.l10(self.dequant(x))#p
		x = self.l11(self.quant(x))#c
		x = self.l12(self.dequant(x))#p
		x = self.l13(self.quant(x))#c
		x = self.l14(self.dequant(x))#p
		x = self.l15(self.quant(x))#ct
		x = self.dequant(x)
		return x