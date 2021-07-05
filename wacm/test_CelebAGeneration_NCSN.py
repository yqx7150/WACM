import os
import torch
import torch.nn as nn
from models.cond_refinenet_dilated import CondRefineNetDilated

__all__ = ['Test_CelebAGeneration_NCSN']

from wt import *


class Test_CelebAGeneration_NCSN():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint_295000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1
        samples = 1

        for z in range(1):
            for c in range(3000):

                x0 = nn.Parameter(torch.Tensor(samples * batch_size, 3, self.config.data.image_size,
                                               self.config.data.image_size).uniform_(-1, 1)).cuda()
                x01 = x0.clone()

                step_lr = 0.00002
                sigmas = np.array(
                    [1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.02782559,
                     0.01668101,
                     0.01])

                n_steps_each = 100
                max_psnr = 0
                max_ssim = 0
                for idx, sigma in enumerate(sigmas):
                    lambda_recon = 1. / sigma ** 2
                    labels = torch.ones(1, device=x0.device) * idx
                    labels = labels.long()

                    step_size = step_lr * (sigma / sigmas[-1]) ** 2

                    print('sigma = {}'.format(sigma))
                    for step in range(n_steps_each):
                        print('current step %03d iter' % step)

                        noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)

                        grad_x0 = scorenet(x01, labels).detach()

                        x0 = x01 + step_size * (grad_x0)

                        x0_U = np.zeros([1, self.config.data.image_size, self.config.data.image_size, 3],
                                        dtype=np.float32)
                        x0_U[0, :, :, 0] = x0[0, 2, ...].clone().detach().cpu().numpy()
                        x0_U[0, :, :, 1] = x0[0, 1, ...].clone().detach().cpu().numpy()
                        x0_U[0, :, :, 2] = x0[0, 0, ...].clone().detach().cpu().numpy()

                        x01 = x0.clone().to(device) + noise_x.to(device)

                        print("current {} step".format(step))

                x_save = x0_U
                x_save_R = x_save[:, :, :, 2:3]
                x_save_G = x_save[:, :, :, 1:2]
                x_save_B = x_save[:, :, :, 0:1]
                x_save = np.concatenate((x_save_R, x_save_G, x_save_B), 3)
                self.write_images(torch.tensor(x_save.transpose(0, 3, 1, 2)), '{}.png'.format(c), samples, z)

    def write_images(self, x, name, n=1, z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1 * d, n * d, 3], dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i * d:(i + 1) * d, j * d:(j + 1) * d, :] = (256 * (x[i * n + j])).clip(0, 255).astype(np.uint8)[:,
                                                                 :, ::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_'.format(z) + name), panel)  #

