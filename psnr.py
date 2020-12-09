import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image1', type = str, required = True)
parser.add_argument('--image2', type = str, required = True)
args = parser.parse_args()

img1 = cv2.imread(args.image1)
img2 = cv2.imread(args.image2)
print(cv2.PSNR(img1, img2), "dB")
