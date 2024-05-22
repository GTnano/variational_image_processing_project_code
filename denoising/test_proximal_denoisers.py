import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr

import argparse
import os
import sys

sys.path.append("../")
from models import utils
from training.data import dataset

sys.path.append('../inverse_problems')
from utils_inverse_problems.reconstruction_map_crr import AdaGD_Recon, AGD_Recon


# HR:
import extra.utils
from extra.utils import batch_images
import matplotlib.pyplot as plt
# HR - end

ssim = StructuralSimilarityIndexMeasure()
torch.set_num_threads(4)
torch.manual_seed(0)

# test dataset
# val_dataset = dataset.H5PY("../training/data/preprocessed/BSD/test.h5")
val_dataset = dataset.H5PY("../training/data/preprocessed/MRI/test.h5") # HR
test_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

noise_type = 'speckle' # 'gaussian'

def test(model, lmbd, mu, sigma=25, tol=1e-6):
    """Perform testing on the test dataset"""
    device = model.device
    psnr_val = torch.zeros(len(test_dataloader))
    ssim_val = torch.zeros(len(test_dataloader))
    n_restart_val = torch.zeros(len(test_dataloader))
    n_iter_val = torch.zeros(len(test_dataloader))
    # HR:
    noisy_img = []
    denoised_img = []
    init_psnr = torch.zeros(len(test_dataloader))
    # HR - end
    for idx, im in enumerate(test_dataloader):
        im = im.to(device)
        if noise_type == 'speckle':
            noise = im * sigma/255*torch.empty_like(im).normal_()
        else:
            noise = sigma/255*torch.empty_like(im).normal_()
        im_noisy = im + noise

        #im_denoised, n_iter, n_restart = utils.accelerated_gd(im_noisy, model, ada_restart=True, lmbd=lmbd, tol=tol, mu=mu, use_strong_convexity=True)
        
        im_denoised, n_iter = utils.AdaGD(im_noisy, model, lmbd=lmbd, tol=tol, mu=mu)
        noisy_img.append(torch.squeeze(im_noisy))  # HR
        denoised_img.append(torch.squeeze(im_denoised))  # HR
        # metrics
        init_psnr[idx] = psnr(im_noisy, im, data_range=1)
        psnr_val[idx] = psnr(im_denoised, im, data_range=1)
        ssim_val[idx] = ssim(im_denoised, im)
        n_iter_val[idx] = n_iter
        n_restart_val[idx] = 0

        # print(f"{idx+1} - running average: {psnr_val[:idx+1].mean().item():.3f}, {n_iter_val[:idx+1].mean().item():.3f}")
        print(f"{idx + 1} - init PSNR : {init_psnr[idx].item():.3f}dB, final PSNR: {psnr_val[idx].item()}")
    # return(psnr_val.mean().item(), ssim_val.mean().item(), n_iter_val.mean().item())  # comment by HR
    return(psnr_val, init_psnr, ssim_val.mean().item(), n_iter_val.mean().item(), noisy_img, denoised_img)  # HR


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='device to use')
    
    parser.add_argument('-t', '--t', default=[5], type=int, nargs="*",
                        help='model selection (list): number of steps used at training subset of (1,2,5,10,20,30,50)')
    
    parser.add_argument('-nltr', '--noise_level_train', default=5, type=int,
                    help='noise level used at training')
    
    parser.add_argument('-nlts', '--noise_level_test', default=25, type=int,
                help='noise level used for current test')
    
    parser.add_argument('-s', '--save', default=0, type=int,
            help='export results')
    
    args = parser.parse_args()

    device = args.device

    ssim = ssim.to(device)

    sigma_train = args.noise_level_train
    sigma_test = args.noise_level_test

    save = args.save

    list_t = args.t
    exp_n = f"Sigma_{sigma_train}"

    # loop over the various t-step models
    for t in list_t:
        
        if noise_type == 'speckle':
            exp_name = f"test/{exp_n}_t_{t}_speckle"
        else:
            exp_name = f"test/{exp_n}_t_{t}"
        
        print(f"**** Validation for model ***** {exp_name}")

        # load model

        model = utils.load_model(exp_name, device=device)
        model.eval()
        model.activation.cache_constraint()

        ## prune if needed
        model.prune(prune_filters=True, collapse_filters=False, change_splines_to_clip=False)

        # compute Lipschitz constant
        model.initializeEigen(size=400)
        L_precise = model.precise_lipschitz_bound(n_iter=100)
        model.L.data = L_precise


        # find best hyperparameters lambds and mu
        val_score_path = f"./validation_scores/validation_scores_sigma_val_{sigma_test}_{exp_name.lower()}.csv"

        if os.path.exists(val_score_path):
            df = pd.read_csv(val_score_path).reset_index(drop=True)
            lmbd = df.loc[df["psnr"].idxmax()]["p1"]
            mu = df.loc[df["psnr"].idxmax()]["p2"]
            print(f"Hyperparameters value: lmbd={lmbd}, mu={mu}")
        else:
            raise ValueError("No validation data found for this model")

        with torch.no_grad():
            # psnr_, ssim_, n_iter = test(model, lmbd=lmbd, mu=mu, sigma=sigma_test)  # comment by HR

            # HR:
            psnr_, init_psnr, ssim_, n_iter, noisy_im, denoised_im = test(model, lmbd=lmbd, mu=mu, sigma=sigma_test)
        # nn = 2
        # vert_noise, hori_noise = extra.utils.create_img_list(noisy_img)
        # vert_denoise, hori_denoise = extra.utils.create_img_list(denoised_im)
        # batched_noisy = batch_images(vert_noise[:nn * nn], nn, nn)
        # batched_de = batch_images(vert_denoise[:nn * nn], nn, nn)
        # plt.figure()
        #
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
        nn = 2
        plt.figure()
        plt.suptitle("t-step denoisers")
        for idx in range(nn * nn):
            plt.subplot(2, nn * nn, idx + 1)
            plt.imshow(noisy_im[idx], cmap='gray')
            plt.title(f"psnr: {init_psnr[idx].item():.2f}")
            plt.axis('off')
            plt.subplot(2, nn * nn, idx + nn * nn + 1)
            plt.imshow(denoised_im[idx], cmap='gray')
            plt.title(f"psnr: {psnr_[idx].item():.2f}")
            plt.tight_layout()
            plt.axis('off')
        plt.show()
        # end - HR
        print(f"PSNR: {psnr_.mean().item():.2f} dB")

        # save
        if save == 1:
            path = "./test_scores/test_scores_ada_db.csv"
            columns = ["sigma_test", "sigma_train", "model_name", "psnr", "denoiser_type", "t"]
            if os.path.isfile(path):
                db = pd.read_csv(path)
            else:
                db = pd.DataFrame(columns=columns)

            line = {"sigma_test": sigma_test, "sigma_train": sigma_train, "model_name": exp_name, "denoiser_type": "proximal", "t": t}

            ind = [True] * len(db)
            for col, val in line.items():
                ind = ind & (db[col] == val)
            db = db.drop(db[ind].index)
            line["psnr"] = psnr_
            db = pd.concat((db, pd.DataFrame([line], columns=columns)), ignore_index=True)

            db.to_csv(path, index=False)
