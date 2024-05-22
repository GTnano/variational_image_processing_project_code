import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure 
from torchmetrics.functional import peak_signal_noise_ratio as psnr

import sys

from models import utils
from training.data import dataset
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
from imnest_ivhc import imnest_ivhc

torch.set_num_threads(4)
torch.manual_seed(0)

val_dataset = dataset.H5PY("training/data/preprocessed/MRI/test.h5")
print(len(val_dataset))
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

iterate_both = True
salt_ratio = 0.05
pepper_ratio = 0.05

def test(model, sigma, t, choose_idx: int = None, noise_type : str = None):
    end_flag = False
    device = model.device
    psnr_val = torch.zeros(len(val_dataloader))
    ssim_val = torch.zeros(len(val_dataloader))

    ssim = StructuralSimilarityIndexMeasure()

    ssim = ssim.to(device)

    original_im = []
    noisy_im = []
    denoised_im = []
    psnr_val_arr = []
    init_psnr_arr = []
    init_psnr = torch.zeros(len(val_dataloader))

    # true_std = []
    # est_std = []
    # std_err = []
    # std_gamma = []
    if iterate_both:
        noise_values = ['gaussian', 'speckle', 'salt_and_pepper']
    else:
        noise_values = [noise_type]

    for noise_type in noise_values:
        for idx, im in enumerate(val_dataloader):
            if choose_idx is not None:
                if idx+1 != choose_idx:
                    continue
                else:
                    end_flag = True
            # if idx + 1 == 212: # For some reason image 212 crashes the code
            #     continue

            # Add noise
            if noise_type == 'speckle':
                noise = im * sigma/255*torch.empty_like(im).normal_()
                im_noisy = im + noise
            elif noise_type == 'salt_and_pepper':
                im_noisy = torch.clone(im)
                salt = np.random.rand(*list(im_noisy.shape)) < salt_ratio
                pepper = np.random.rand(*list(im_noisy.shape)) < pepper_ratio
                im_noisy[salt] = 1
                im_noisy[pepper] = 0
            else:
                noise = sigma/255*torch.empty_like(im).normal_()
                im_noisy = im + noise

            # -------------------- Estimate STD --------------------
            # noise_std = sigma
            # im_noisy_np = np.squeeze(im_noisy.numpy())
            # [est_sigma_p, est_sigma_o, nlf] = imnest_ivhc(im_noisy_np, 0)
            # est_error = abs(noise_std-est_sigma_p*255)
            # est_gamma = est_sigma_o/est_sigma_p
            # true_std.append(noise_std)
            # est_std.append(est_sigma_p*255)
            # std_err.append(est_error)
            # std_gamma.append(est_gamma)
            # plt.figure()
            # plt.imshow(im_noisy.numpy()[0, 0, :, :], cmap='gray')
            # plt.tight_layout()
            # plt.axis('off')
            # plt.show()
            # -------------------- Estimate STD --------------------

            im = im.to(device)
            im_noisy = im_noisy.to(device)
            im_noisy.requires_grad = False
            im_init = im_noisy

            im_denoised = utils.tStepDenoiser(model, im_noisy, t_steps=t)
            psnr_val[idx] = psnr(im_denoised, im, data_range=1)
            ssim_val[idx] = ssim(im_denoised, im)

            init_psnr[idx] = psnr(im_init, im, data_range=1)
            print(f"{idx + 1} - running average : {psnr_val[:idx+1].mean().item():.3f}dB, init PSNR : {init_psnr[idx].item():.3f}dB, final PSNR: {psnr_val[idx].item()}")
            original_im.append(torch.squeeze(im))
            noisy_im.append(torch.squeeze(im_noisy))
            denoised_im.append(torch.squeeze(im_denoised))
            psnr_val_arr.append(psnr_val[idx].item())
            init_psnr_arr.append(init_psnr[idx].item())

            if end_flag:
                break

    return(psnr_val, init_psnr, ssim_val.mean().item(), noisy_im, denoised_im, original_im, psnr_val_arr, init_psnr_arr)

if __name__=="__main__":   
    noise_type = 'salt_and_pepper' # 'gaussian', 'both'
    choose_idx = 17

    device = "cuda"
    list_t = [1]

    # arr_sigma_model = [5, 10, 15, 20 ,25]
    arr_sigma_model = [15]
    
    # arr_sigma_test = [1, 5, 15 ,25, 50]
    arr_sigma_test = [15]

    first_iter_flag = True

    # test for various t-steps models
    for t in list_t:

        # various options to prune the model, usual slightly alter the results and improve the speed
        #model.prune(change_splines_to_clip=False, prune_filters=True, collapse_filters=True)

        img_ref = []
        psnr_ref = []

        if iterate_both:
            noise_values = ['gaussian', 'speckle', 'salt_and_pepper']
            data_dict = {}
            psnr_dict = {}
            img_ref_dict = {}
            psnr_ref_dict = {}
        else:
            noise_values = [noise_type]
            data_dict = None

        for noise_type in noise_values:
            img_mat = []
            multi_image_mat = []
            multi_image_noisy_mat = []
            psnr_mat = []

            for sigma_train_idx in arr_sigma_model:

                img_row = []
                multi_image_row = []
                multi_image_noisy_row = []
                psnr_row = []

                exp_n = f"Sigma_{sigma_train_idx}"
                if noise_type == 'speckle':
                    exp_name = f"test/{exp_n}_t_{t}_speckle"
                if noise_type == 'salt_and_pepper':
                    exp_name = f"test/{exp_n}_t_{t}_salt_and_pepper"
                else:
                    exp_name = f"test/{exp_n}_t_{t}"
                print(f"**** Testing model ***** {exp_name}")
                model = utils.load_model(exp_name, device=device)
                model.eval()
                model.initializeEigen(size=400)
                L_precise = model.precise_lipschitz_bound(n_iter=200)
                model.L.data = L_precise

                for sigma_test_idx in arr_sigma_test:
                    with torch.no_grad():
                        psnr_, init_psnr, ssim_m, noisy_im, denoised_im, original_im, psnr_val_arr, init_psnr_arr = test(model, t=t, sigma=sigma_test_idx, choose_idx=choose_idx, noise_type=noise_type)
                        # print(f"psnr_={psnr_},type={type(psnr_)},size={psnr_.size()}")
                        print(f"\nnoisy_im length: {len(noisy_im)}")

                    if first_iter_flag:
                        init_psnr_nonzero = init_psnr.nonzero()
                        init_psnr_mean = init_psnr[init_psnr_nonzero].mean()
                        psnr_ref.append(init_psnr_mean.cpu().numpy())

                        original_image = original_im[0].cpu().numpy()

                    psnr_nonzero = psnr_.nonzero()
                    psnr_mean = psnr_[psnr_nonzero].mean()

                    img_row.append(denoised_im[0].cpu().numpy())
                    multi_image_row.append([dim.cpu().numpy() for dim in denoised_im])
                    multi_image_noisy_row.append([dim.cpu().numpy() for dim in noisy_im])
                    psnr_row.append(psnr_mean.cpu().numpy())

                first_iter_flag = False

                img_mat.append(img_row)
                multi_image_mat.append(multi_image_row)
                multi_image_noisy_mat.append(multi_image_noisy_row)
                psnr_mat.append(psnr_row)
                img_ref.append(noisy_im[0].cpu().numpy())

            if isinstance(data_dict, dict):
                data_dict[noise_type] = multi_image_mat[0][0]
                psnr_dict[noise_type] = psnr_val_arr
                img_ref_dict[noise_type] = multi_image_noisy_mat[0][0]
                psnr_ref_dict[noise_type] = init_psnr_arr

        # ---------- FIGURE 1 ---------- 
        # plt.figure()
        # plt.suptitle("t-step Denoisers, Trained Model STD vs Added Test Noise STD")
        # for ii in range(len(arr_sigma_model)):
        #     for jj in range(len(arr_sigma_test)+1):
        #         plt.subplot(len(arr_sigma_model), len(arr_sigma_test)+1, ii*(len(arr_sigma_test)+1) + jj + 1)
        #         if jj == 0:
        #             plt.imshow(img_ref[ii], cmap='gray')
        #             plt.title(f"$\sigma_{{test}}$={arr_sigma_model[ii]}, psnr={psnr_ref[ii]:.2f}", fontsize=10)
        #         else:
        #             plt.imshow(img_mat[jj-1][ii], cmap='gray')
        #             plt.title(f"$\sigma_{{test}}$={arr_sigma_test[jj-1]}, $\sigma_{{model}}$={arr_sigma_model[ii]}, psnr={psnr_mat[jj-1][ii]:.2f}", fontsize=10)

        #         plt.tight_layout()
        #         plt.axis('off')
        # plt.show()
        # ---------- FIGURE 1 ---------- 

        # ---------- FIGURE 2 ---------- 
        # plt.figure()
        # plt.suptitle("t-step Denoisers, Trained Model STD vs Added Test Noise STD")

        # plt.subplot(231)
        # plt.imshow(img_mat[0][2], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[2]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][2]:.2f}", fontsize=10)

        # plt.subplot(234)
        # plt.imshow(img_mat[0][2]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[2]}", fontsize=10)

        # plt.subplot(232)
        # plt.imshow(img_mat[2][2], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[2]}, $\sigma_{{model}}$={arr_sigma_model[2]}, psnr={psnr_mat[2][2]:.2f}", fontsize=10)

        # plt.subplot(235)
        # plt.imshow(img_mat[2][2]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[2]}", fontsize=10)

        # plt.subplot(233)
        # plt.imshow(img_mat[4][2], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[2]}, $\sigma_{{model}}$={arr_sigma_model[4]}, psnr={psnr_mat[4][2]:.2f}", fontsize=10)

        # plt.subplot(236)
        # plt.imshow(img_mat[4][2]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[2]}", fontsize=10)

        # plt.tight_layout()
        # plt.axis('off')
        # plt.show()
        # ---------- FIGURE 2 ---------- 

        # ---------- FIGURE 3 ---------- 
        # plt.figure()
        # plt.suptitle("t-step Denoisers, Trained Model STD vs Added Test Noise STD")

        # plt.subplot(251)
        # plt.imshow(img_mat[0][0], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[0]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][0]:.2f}", fontsize=10)

        # plt.subplot(256)
        # plt.imshow(img_mat[0][0]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[0]}", fontsize=10)

        # plt.subplot(252)
        # plt.imshow(img_mat[0][1], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[1]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][1]:.2f}", fontsize=10)

        # plt.subplot(257)
        # plt.imshow(img_mat[0][1]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[1]}", fontsize=10)

        # plt.subplot(253)
        # plt.imshow(img_mat[0][2], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[2]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][2]:.2f}", fontsize=10)

        # plt.subplot(258)
        # plt.imshow(img_mat[0][2]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[2]}", fontsize=10)

        # plt.subplot(254)
        # plt.imshow(img_mat[0][3], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[3]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][3]:.2f}", fontsize=10)

        # plt.subplot(259)
        # plt.imshow(img_mat[0][3]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[3]}", fontsize=10)

        # plt.subplot(255)
        # plt.imshow(img_mat[0][4], cmap='gray')
        # plt.title(f"$\sigma_{{test}}$={arr_sigma_test[3]}, $\sigma_{{model}}$={arr_sigma_model[0]}, psnr={psnr_mat[0][3]:.2f}", fontsize=10)

        # plt.subplot(2, 5, 10)
        # plt.imshow(img_mat[0][4]-original_image, cmap='gray')
        # plt.title(f"$\Delta\sigma_{{test}}$={arr_sigma_test[3]}", fontsize=10)

        # plt.tight_layout()
        # plt.axis('off')
        # plt.show()
        # ---------- FIGURE 3 ---------- 

        # ---------- FIGURE 4 ---------- 
        # It seems the salt and pepper model diverges and gives a NAN image...
        plt.figure()
        plt.suptitle("t-step Denoisers, Trained Model STD vs Added Test Noise STD")
        for ii in range(len(noise_values)):
            for jj in range(len(noise_values)):
                plt.subplot(len(noise_values), len(noise_values), ii*len(noise_values) + jj + 1)
                if ii == 0:
                    plt.imshow(img_ref_dict[noise_values[ii]][jj], cmap='gray')
                    plt.title(f"NLF={noise_values[jj]}, psnr={psnr_ref_dict[noise_values[jj]][jj]:.2f}", fontsize=10)
                else:
                    plt.imshow(data_dict[noise_values[ii-1]][jj], cmap='gray')
                    plt.title(f"model={noise_values[ii-1]}, NLF={noise_values[jj]}, psnr={psnr_dict[noise_values[ii-1]][jj]:.2f}", fontsize=10)

        # plt.tight_layout()
        plt.axis('off')
        plt.show()
        # ---------- FIGURE 4 ---------- 