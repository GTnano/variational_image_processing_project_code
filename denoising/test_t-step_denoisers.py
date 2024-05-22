import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr

import os
import sys

# import extra.utils

sys.path.append("../")
from models import utils
from training.data import dataset
import pandas as pd

# HR:
from extra.utils import batch_images
import matplotlib.pyplot as plt
# HR - end

# GUY
import numpy as np
import cv2
from imnest_ivhc import imnest_ivhc
# GUY - end

torch.set_num_threads(4)
torch.manual_seed(0)

val_dataset = dataset.H5PY("../training/data/preprocessed/BSD/test.h5")
print(len(val_dataset))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def test(model, sigma, t):
    device = model.device
    psnr_val = torch.zeros(len(val_dataloader))
    ssim_val = torch.zeros(len(val_dataloader))

    ssim = StructuralSimilarityIndexMeasure()

    ssim = ssim.to(device)

    # HR:
    noisy_im = []
    denoised_im = []
    init_psnr = torch.zeros(len(val_dataloader))

    true_std = []
    est_std = []
    std_err = []
    std_gamma = []

    # HR - end
    # sigma = 25
    flag = False
    for idx, im in enumerate(val_dataloader):
        for item in [2, 3, 11, 18, 22, 26, 31, 33, 34, 40, 51, 62, 67, 212]: # For some reason images crashes the code for heigher sigmas
            if idx + 1 == item:
                flag = True
                break
        if flag:
            flag = False
            continue

        # im = im.to(device) GUY
        im_noisy = im + sigma/255*torch.empty_like(im).normal_()

        # -------------------- Estimate STD --------------------
        noise_std = sigma
        im_noisy_np = np.squeeze(im_noisy.numpy())
        # im_noisy_np = cv2.normalize(im_noisy_np, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        # print('------------------------')
        # print(type(im_noisy_np))
        # print(im_noisy_np)
        # print('------------------------')
        [est_sigma_p, est_sigma_o, nlf] = imnest_ivhc(im_noisy_np, 0)
        # est_error = np.abs(noise_std-est_sigma_p)
        est_error = abs(noise_std-est_sigma_p*255)
        est_gamma = est_sigma_o/est_sigma_p
        # print('------------------------')
        # print('%d - est = %2.3f, true = %2.3f'%(idx+1, est_sigma_p, noise_std))
        # print('%d - Noise std estimation error: %2.3f'%(idx+1, est_error))
        true_std.append(noise_std)
        est_std.append(est_sigma_p*255)
        std_err.append(est_error)
        std_gamma.append(est_gamma)
        # print('Estimated processing degree: %2.3f'%(est_gamma))
        # print('------------------------')
        # -------------------- Estimate STD --------------------

        im = im.to(device)
        im_noisy = im_noisy.to(device)
        im_noisy.requires_grad = False
        im_init = im_noisy

        im_denoised = utils.tStepDenoiser(model, im_noisy, t_steps=t)
        psnr_val[idx] = psnr(im_denoised, im, data_range=1)
        ssim_val[idx] = ssim(im_denoised, im)
        # print(f"{idx+1} - running average : {psnr_val[:idx+1].mean().item():.3f}dB")

        # HR:
        init_psnr[idx] = psnr(im_init, im, data_range=1)
        print(f"{idx + 1} - running average : {psnr_val[:idx+1].mean().item():.3f}dB, init PSNR : {init_psnr[idx].item():.3f}dB, final PSNR: {psnr_val[idx].item()}")
        noisy_im.append(torch.squeeze(im_noisy))
        denoised_im.append(torch.squeeze(im_denoised))

    # return(psnr_val.mean().item(), ssim_val.mean().item()) comment by HR
    df = pd.DataFrame({'true_std': true_std, 'est_std': est_std, 'std_err': std_err, 'std_gamma': std_gamma})
    df.to_csv(f'std_estimation_results_{noise_std}.csv')
    return(psnr_val, init_psnr, ssim_val.mean().item(), noisy_im, denoised_im)
#     HR - end


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')
    
    parser.add_argument('-t', '--t', default=[1], type=int, nargs="*",
                        help='model selection (list): number of steps used at training subset of (1,2,5,10,20,30,50)')

    parser.add_argument('-nl', '--noise_level', default=7, type=int,
                    help='model selection: noise level used at training and for testing')
    
   
    args = parser.parse_args()

    device = args.device
    sigma_train = args.noise_level
    sigma_test = args.noise_level
    list_t = args.t

    exp_n = f"Sigma_{sigma_train}"

    # test for various t-steps models
    for t in list_t:
        
        exp_name = f"test/{exp_n}_t_{t}"

        print(f"**** Testing model ***** {exp_name}")

        model = utils.load_model(exp_name, device=device)
        model.eval()

        
        # various options to prune the model, usual slightly alter the results and improve the speed
        #model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)

        model.initializeEigen(size=400)
        L_precise = model.precise_lipschitz_bound(n_iter=200)
        model.L.data = L_precise

       
        with torch.no_grad():
            # HR:
            # psnr_, ssim_ = test(model, t=t, sigma=sigma_test) #comment by HR
            psnr_, init_psnr, ssim_m, noisy_im, denoised_im = test(model, t=t, sigma=sigma_test)
            print(f"\nnoisy_im length: {len(noisy_im)}")
        nn = 2

        for idx in range(len(noisy_im)):
            noisy_im[idx] = noisy_im[idx].cpu().numpy() 
        for idx in range(len(denoised_im)):
            denoised_im[idx] = denoised_im[idx].cpu().numpy() 

        plt.figure()
        plt.suptitle("t-step denoisers")
        for idx in range(nn*nn):
            plt.subplot(2, nn*nn, idx+1)
            plt.imshow(noisy_im[idx], cmap='gray')
            plt.title(f"psnr: {init_psnr[idx].item():.2f}")
            plt.axis('off')
            plt.subplot(2, nn * nn, idx + nn*nn+1)
            plt.imshow(denoised_im[idx], cmap='gray')
            plt.title(f"psnr: {psnr_[idx].item():.2f}")
            plt.tight_layout()
            plt.axis('off')
        plt.show()
        
        # vert_noise, hori_noise = extra.utils.create_img_list(noisy_im)
        # vert_denoise, hori_denoise = extra.utils.create_img_list(denoised_im)
        # batched_noisy = batch_images(vert_noise[:nn * nn], nn, nn)
        # batched_de = batch_images(vert_denoise[:nn * nn], nn, nn)


        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(batched_noisy, cmap='gray')
        # plt.tight_layout()
        # plt.title('Noisy Images')
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(batched_de, cmap='gray')
        # plt.tight_layout()
        # plt.title('After Denoising')
        # plt.axis('off')
        # plt.show()

        print(f"PSNR: {psnr_.mean().item():.2f} dB")

        # save
        path = "./test_scores/test_scores_db.csv"
        columns = ["sigma_test", "sigma_train", "model_name", "psnr", "denoiser_type", "t"]
        if os.path.isfile(path):
            db = pd.read_csv(path)
        else:
            db = pd.DataFrame(columns=columns)

        # line to append
        line = {"sigma_test": sigma_test, "sigma_train": sigma_train, "model_name": exp_name, "denoiser_type": "tsteps", "t": t}
        # remove data with same keys
        ind = [True] * len(db)
        for col, val in line.items():
            ind = ind & (db[col] == val)
        db = db.drop(db[ind].index)

        line["psnr"] = psnr_
        
        db = pd.concat((db, pd.DataFrame([line], columns=columns)), ignore_index=True)

        db.to_csv(path, index=False)
       