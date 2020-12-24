import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import QFSRCNN
from utils import convert_ycbcr_to_rgb, preprocess, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on:', device)
    model = QFSRCNN(scale_factor=args.scale).to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')
    
    bicubic = image.resize((image.width * args.scale, image.height * args.scale), resample = pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
    _, ycbcr = preprocess(bicubic, device)
    image, _ = preprocess(image, device)
    with torch.no_grad():
        preds = model(image).clamp(0.0, 1.0)

    #psnr = calc_psnr(hr, preds)
    #print('PSNR: {:.2f}'.format(psnr))
    print('Done')
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    kernel = kernel = np.ones((3,3),np.float32)/9
    output = cv2.filter2D(output,-1,kernel)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_qfsrcnn_x{}.'.format(args.scale)))
