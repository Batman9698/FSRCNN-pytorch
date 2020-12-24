import math
from torch import nn


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class QFSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels = 1, d = 56, s = 12, m = 4):
        super(QFSRCNN, self).__init__()

        

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
        self.l14 = nn.PReLU(d)

        self.l15 = nn.ConvTranspose2d(d, num_channels, kernel_size = 9, stride = scale_factor, padding = 9//2, output_padding = scale_factor - 1)

    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        
        return x

