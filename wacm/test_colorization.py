import os
import shutil
from natsort import natsorted
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
from models.cond_refinenet_dilated import CondRefineNetDilated
from skimage.measure import compare_psnr,compare_ssim


__all__ = ['Test_colorizaiton']

from wt import *


def write_Data(path, result_all_13, i, z):
    with open(os.path.join(path,"psnr_output_{}".format(z)+".txt"),"w+") as f:
        for i in range(len(result_all_13)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all_13[i][0]) + '    SSIM : ' + str(result_all_13[i][1]))
            f.write('\n')

class Test_colorizaiton():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint_495000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        batch_size = 1 
        samples = 1
        files_list = glob.glob('./test_images/test_'+self.config.data.dataset+'_100/*.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all_12 = np.zeros([12,2])
        result_all = np.zeros([101,2])
        for z,file_path in enumerate(files_list):
            for c in range(12):
                img = cv2.imread(file_path)
                img = cv2.resize(img, (self.config.data.image_size, self.config.data.image_size))*1.0
                
                original_image = img.copy()
                
                img = img[:, :, :3] / 255
                gray = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0

                cA, cH, cV, cD = dwt_rgb(gray)

                cA = cv2.merge([cA, cA, cA]).transpose(2, 0, 1)
                cH = cv2.merge([cH, cH, cH]).transpose(2, 0, 1)
                cV = cv2.merge([cV, cV, cV]).transpose(2, 0, 1)
                cD = cv2.merge([cD, cD, cD]).transpose(2, 0, 1)

                

                gray = torch.tensor(gray, dtype=torch.float32).unsqueeze(0)
                gray = torch.stack([gray, gray, gray], dim=1)
                cA = torch.tensor(cA, dtype=torch.float32).unsqueeze(0)
                cH = torch.tensor(cH, dtype=torch.float32).unsqueeze(0)
                cV = torch.tensor(cV, dtype=torch.float32).unsqueeze(0)
                cD = torch.tensor(cD, dtype=torch.float32).unsqueeze(0)


                x0 = nn.Parameter(torch.Tensor(samples*batch_size,12,cA.shape[2],cA.shape[3]).uniform_(-1,1)).cuda()
                x01 = x0.clone()

                step_lr=0.0000599 * 0.26
                sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01]) 
                
                n_steps_each = 100
                max_psnr = 0
                max_ssim = 0
                for idx, sigma in enumerate(sigmas):
                    lambda_recon = 1./sigma**2
                    labels = torch.ones(1, device=x0.device) * idx
                    labels = labels.long()

                    step_size = step_lr * (sigma / sigmas[-1]) ** 2
                    
                    print('sigma = {}'.format(sigma))
                    for step in range(n_steps_each):
                        print('current step %03d iter' % step)  
                        
                        noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)

                        grad_x0 = scorenet(x01, labels).detach()
                     
                        x0 = x01 + step_size * (grad_x0)
                        
                        cA_mix = (x0[:,0,...] + x0[:,4,...] + x0[:,8,...])/3.0
                        cH_mix = (x0[:,1,...] + x0[:,5,...] + x0[:,9,...])/3.0
                        cV_mix = (x0[:,2,...] + x0[:,6,...] + x0[:,10,...])/3.0
                        cD_mix = (x0[:,3,...] + x0[:,7,...] + x0[:,11,...])/3.0
                        
                        error_cA = cA_mix.to(device) - cA[:, 0, ...].to(device)
                        error_cH = cH_mix.to(device) - cH[:, 0, ...].to(device)
                        error_cV = cV_mix.to(device) - cV[:, 0, ...].to(device)
                        error_cD = cD_mix.to(device) - cD[:, 0, ...].to(device)
                        
                        x0[:, 0, ...] = x0[:, 0, ...] - step_size * lambda_recon * error_cA
                        x0[:, 4, ...] = x0[:, 4, ...] - step_size * lambda_recon * error_cA
                        x0[:, 8, ...] = x0[:, 8, ...] - step_size * lambda_recon * error_cA

                        x0[:, 1, ...] = x0[:, 1, ...] - step_size * lambda_recon * error_cH
                        x0[:, 5, ...] = x0[:, 5, ...] - step_size * lambda_recon * error_cH
                        x0[:, 9, ...] = x0[:, 9, ...] - step_size * lambda_recon * error_cH

                        x0[:, 2, ...] = x0[:, 2, ...] - step_size * lambda_recon * error_cV
                        x0[:, 6, ...] = x0[:, 6, ...] - step_size * lambda_recon * error_cV
                        x0[:, 10, ...] = x0[:, 10, ...] - step_size * lambda_recon * error_cV

                        x0[:, 3, ...] = x0[:, 3, ...] - step_size * lambda_recon * error_cD
                        x0[:, 7, ...] = x0[:, 7, ...] - step_size * lambda_recon * error_cD
                        x0[:, 11, ...] = x0[:, 11, ...] - step_size * lambda_recon * error_cD

                        size=(self.config.data.image_size/2)**2

                        error_avg_rcH = (x0[:, 1, ...].sum()/size).to(device) - (cH[:, 0, ...].sum()/size).to(device)
                        error_avg_rcV = (x0[:, 2, ...].sum()/size).to(device) - (cV[:, 0, ...].sum()/size).to(device)
                        error_avg_rcD = (x0[:, 3, ...].sum()/size).to(device) - (cD[:, 0, ...].sum()/size).to(device)

                        error_avg_gcH = (x0[:, 5, ...].sum()/size).to(device) - (cH[:, 0, ...].sum()/size).to(device)
                        error_avg_gcV = (x0[:, 6, ...].sum()/size).to(device) - (cV[:, 0, ...].sum()/size).to(device)
                        error_avg_gcD = (x0[:, 7, ...].sum()/size).to(device) - (cD[:, 0, ...].sum()/size).to(device)

                        error_avg_bcH = (x0[:, 9, ...].sum()/size).to(device) - (cH[:, 0, ...].sum()/size).to(device)
                        error_avg_bcV = (x0[:, 10, ...].sum()/size).to(device) - (cV[:, 0, ...].sum()/size).to(device)
                        error_avg_bcD = (x0[:, 11, ...].sum()/size).to(device) - (cD[:, 0, ...].sum()/size).to(device)
                        
                        x02 = x0.clone()
                        
                        drate=1

                        x02[:, 1, ...] = x0[:, 1, ...] - error_avg_rcH * drate
                        x02[:, 5, ...] = x0[:, 5, ...] - error_avg_gcH * drate
                        x02[:, 9, ...] = x0[:, 9, ...] - error_avg_bcH * drate

                        x02[:, 2, ...] = x0[:, 2, ...] - error_avg_rcV * drate
                        x02[:, 6, ...] = x0[:, 6, ...] - error_avg_gcV * drate
                        x02[:, 10, ...] = x0[:, 10, ...] - error_avg_bcV * drate

                        x02[:, 3, ...] = x0[:, 3, ...] - error_avg_rcD * drate
                        x02[:, 7, ...] = x0[:, 7, ...] - error_avg_gcD * drate
                        x02[:, 11, ...] = x0[:, 11, ...] - error_avg_bcD * drate

                        x0_r = torch.tensor(idwt(x02[0, 0, ...], x02[0, 1, ...], x02[0, 2, ...], x02[0, 3, ...]))
                        x0_g = torch.tensor(idwt(x02[0, 4, ...], x02[0, 5, ...], x02[0, 6, ...], x02[0, 7, ...]))
                        x0_b = torch.tensor(idwt(x02[0, 8, ...], x02[0, 9, ...], x02[0, 10, ...], x02[0, 11, ...]))

                        x0_U = np.zeros([1,self.config.data.image_size,self.config.data.image_size,3],dtype=np.float32)
                        x0_U[0, :, :, 0] = x0_r
                        x0_U[0, :, :, 1] = x0_g
                        x0_U[0, :, :, 2] = x0_b

                        x01 = x0.clone().to(device) + noise_x.to(device)
                       	
                        x_rec = x0_U
                        original_image = np.array(original_image,dtype = np.float32)

                        for i in range(x_rec.shape[0]):
                            psnr = compare_psnr(x_rec[i,...]*255.0,original_image,data_range=255)
                            ssim = compare_ssim(x_rec[i,...],original_image/255.0,data_range=1,multichannel=True)
                            print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                            
                        if max_psnr < psnr :
                            result_all_12[c,0] = psnr
                            max_psnr = psnr
                        if max_ssim < ssim:
                            result_all_12[c,1] = ssim
                            max_ssim = ssim
                        write_Data(self.args.image_folder,result_all_12,c,z)
                    
                x_save = x0_U
                x_save_R = x_save[:,:,:,2:3]
                x_save_G = x_save[:,:,:,1:2]
                x_save_B = x_save[:,:,:,0:1]
                x_save = np.concatenate((x_save_R,x_save_G,x_save_B),3)
                self.write_images(torch.tensor(x_save.transpose(0,3,1,2)), '{}.png'.format(c),samples,z)
                result_all[z,:] = result_all_12[np.argmax(result_all_12[:,0],0),:]


    def write_images(self, x,name,n=1,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_'.format(z) + name), panel)#

