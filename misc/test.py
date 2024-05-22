import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from models import utils
from training.data import dataset
from imnest_ivhc import imnest_ivhc

torch.set_num_threads(4)
torch.manual_seed(0)

test_dataset = dataset.H5PY("training/data/preprocessed/MRI/test.h5")
print(len(test_dataset))
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

plot_idx = None

def test_t_step_denoiser(model, sigma, t):
    device = model.device
    psnr_val = torch.zeros(len(test_dataloader))
    ssim_val = torch.zeros(len(test_dataloader))

    ssim = StructuralSimilarityIndexMeasure()

    ssim = ssim.to(device)

    # HR:
    noisy_im = []
    denoised_im = []
    init_psnr = torch.zeros(len(test_dataloader))

    true_std = []
    est_std = []
    std_err = []
    std_gamma = []

    for idx, im in enumerate(test_dataloader):
        # if plot_idx is not None:
        #     skip_flag = True
        #     for plot_item in plot_idx:
        #         if plot_item == idx:
        #             skip_flag = False
        #             break
        #     if skip_flag: continue

        im_noisy = im + sigma/255*torch.empty_like(im).normal_()

        # # -------------------- Estimate STD --------------------
        # noise_std = sigma
        # im_noisy_np = np.squeeze(im_noisy.numpy())
        # [est_sigma_p, est_sigma_o, nlf] = imnest_ivhc(im_noisy_np, 0)
        # est_error = abs(noise_std-est_sigma_p*255)
        # est_gamma = est_sigma_o/est_sigma_p
        # true_std.append(noise_std)
        # est_std.append(est_sigma_p*255)
        # std_err.append(est_error)
        # std_gamma.append(est_gamma)
        # # -------------------- Estimate STD --------------------

        im = im.to(device)
        im_noisy = im_noisy.to(device)
        im_noisy.requires_grad = False
        im_init = im_noisy

        im_denoised = utils.tStepDenoiser(model, im_noisy, t_steps=t)
        psnr_val[idx] = psnr(im_denoised, im, data_range=1)
        ssim_val[idx] = ssim(im_denoised, im)

        init_psnr[idx] = psnr(im_init, im, data_range=1)
        print(f"{idx + 1} - running average : {psnr_val[:idx+1].mean().item():.3f}dB, init PSNR : {init_psnr[idx].item():.3f}dB, final PSNR: {psnr_val[idx].item()}")
        noisy_im.append(torch.squeeze(im_noisy))
        denoised_im.append(torch.squeeze(im_denoised))

    # df = pd.DataFrame({'true_std': true_std, 'est_std': est_std, 'std_err': std_err, 'std_gamma': std_gamma})
    # df.to_csv(f'std_estimation_results_{noise_std}.csv')

    return(psnr_val, init_psnr, ssim_val.mean().item(), noisy_im, denoised_im)

def test_proximal_denoiser(model, lmbd, mu, sigma=25, tol=1e-6):
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
        im_noisy = im + sigma/255*torch.empty_like(im).normal_()

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
    device = "cuda"
    list_t = [1]
    list_t_len = len(list_t)

    models_std = [5, 10, 15, 20 ,25]
    models_std_len = len(models_std)
    tests_std = [5, 10, 15, 20 ,25]
    tests_std_len = len(tests_std)

    psnr_mat = np.zeros((len(test_dataset), list_t_len, tests_std_len, models_std_len))
    psnr_init_mat = np.zeros((len(test_dataset), list_t_len, tests_std_len))

    # test for various t-steps models
    for t_idx, t in enumerate(list_t):
        img_mat = []
        img_ref = []

        for model_idx, model_std in enumerate(models_std):
            img_row = []

            exp_n = f"Sigma_{model_std}"
            exp_name = f"test/{exp_n}_t_{t}"
            print(f"**** Testing model ***** {exp_name}")
            model = utils.load_model(exp_name, device=device)
            model.eval()
            model.initializeEigen(size=400)
            L_precise = model.precise_lipschitz_bound(n_iter=200)
            model.L.data = L_precise

            for test_idx, test_std in enumerate(tests_std):
                with torch.no_grad():
                    psnr_, init_psnr, ssim_m, noisy_im, denoised_im = test_t_step_denoiser(model, t=t, sigma=test_std)

                if t_idx == 0:
                    psnr_init_mat[:, t_idx, test_idx] = init_psnr.cpu().numpy()
                psnr_mat[:, t_idx, test_idx, model_idx] = psnr_.cpu().numpy()

                if plot_idx:
                    img_row.append(denoised_im[0].cpu().numpy())

            if plot_idx:
                img_mat.append(img_row)
                img_ref.append(noisy_im[0].cpu().numpy())

        if plot_idx:
            plt.figure()
            plt.suptitle("t-step Denoisers, Trained Model STD vs Added Test Noise STD")
            for ii in range(models_std_len):
                for jj in range(tests_std_len+1):
                    plt.subplot(models_std_len, tests_std_len+1, ii*(tests_std_len+1) + jj + 1)
                    if jj == 0:
                        plt.imshow(img_ref[ii], cmap='gray')
                        plt.title(f"$\sigma_{{test}}$={models_std[ii]}, psnr={psnr_ref[ii]:.2f}", fontsize=10)
                    else:
                        plt.imshow(img_mat[jj-1][ii], cmap='gray')
                        plt.title(f"$\sigma_{{test}}$={tests_std[jj-1]}, $\sigma_{{model}}$={models_std[ii]}, psnr={psnr_mat[jj-1][ii]:.2f}", fontsize=10)

                    plt.tight_layout()
                    plt.axis('off')
            plt.show()