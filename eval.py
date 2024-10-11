import torch
import torchvision.transforms.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_fid import fid_score
from lpips import LPIPS
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms


preprocess = transforms.Compose([
    transforms.Resize((256, 192))
])

def calculate_ssim(image1, image2):
    return ssim(image1, image2, channel_axis=-1, data_range=1.0)

def calculate_psnr(image1, image2):
    return psnr(image1, image2)

def calculate_lpips(image1, image2, lpips, device):
    image1_tensor = F.to_tensor(image1).unsqueeze(0).to(device)
    image2_tensor = F.to_tensor(image2).unsqueeze(0).to(device)
    distance = lpips(image1_tensor, image2_tensor)
    return distance.item()

def calculate_mse(image1, image2):
    return torch.mean((F.to_tensor(image1) - F.to_tensor(image2)) ** 2).item()

def calculate_fid(folder1, folder2, device):
    return fid_score.calculate_fid_given_paths([folder1, folder2], batch_size=32, device=device, dims=2048)

def calculate_metrics(folder1, folder2):
    image_list1 = os.listdir(folder1)
    image_list2 = os.listdir(folder2)

    metrics = {
        'SSIM': [],
        'PSNR': [],
        'LPIPS': [],
        'MSE': [],
        'FID': []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips = LPIPS(net="squeeze").to(device)

    for image_name1, image_name2 in tqdm(zip(image_list1, image_list2)):
        image_path1 = os.path.join(folder1, image_name1)
        image_path2 = os.path.join(folder2, image_name2)
        image1 = np.array(preprocess(Image.open(image_path1).convert("RGB")))
        image2 = np.array(preprocess(Image.open(image_path2).convert("RGB")))

        ssim_value = calculate_ssim(image1, image2)
        psnr_value = calculate_psnr(image1, image2)
        lpips_value = calculate_lpips(image1, image2, lpips, device)
        mse_value = calculate_mse(image1, image2)

        metrics['SSIM'].append(ssim_value)
        metrics['PSNR'].append(psnr_value)
        metrics['LPIPS'].append(lpips_value)
        metrics['MSE'].append(mse_value)

    fid_value = calculate_fid(folder1, folder2, device)

    mean_metrics = {
        'SSIM': sum(metrics['SSIM']) / len(metrics['SSIM']),
        'PSNR': sum(metrics['PSNR']) / len(metrics['PSNR']),
        'LPIPS': sum(metrics['LPIPS']) / len(metrics['LPIPS']),
        'MSE': sum(metrics['MSE']) / len(metrics['MSE']),
        'FID': fid_value
    }

    return mean_metrics


if __name__ == '__main__':
    folder1 = './log/val/taylor/'
    folder2 = './log/val/output_org_taylor/pose/'

    mean_metrics = calculate_metrics(folder1, folder2)

    print("Mean Metrics:")
    for metric, value in mean_metrics.items():
        print(metric + ": " + str(value))

# taylor
'''
# light
### output_adds ###
Mean Metrics:
SSIM: 0.3193840672326449
PSNR: 16.543131421294532
LPIPS: 0.2551332857459784
MSE: 0.0509297615790274
FID: 74.28452841868518

### output_org ###
Mean Metrics:
SSIM: 0.1756919405814403
PSNR: 15.638043377761239
LPIPS: 0.43907436802983285
MSE: 0.049990305490791795
FID: 101.8949350117781

### output_wo_cross ###
Mean Metrics:
SSIM: 0.00818458306967736
PSNR: 10.105721302338724
LPIPS: 0.6827155947685242
MSE: 0.10110564488503668
FID: 383.6374383920001


# exp
### output_adds ###
Mean Metrics:
SSIM: 0.3183134480674007
PSNR: 17.095131755120697
LPIPS: 0.25551905632019045
MSE: 0.05396455555455759
FID: 69.52749529447337

### output_org ###
Mean Metrics:
SSIM: 0.1406705915659187
PSNR: 13.270186458790192
LPIPS: 0.51557078063488
MSE: 0.06679783780127764
FID: 128.77841181280957

### output_wo_cross ###
Mean Metrics:
SSIM: 0.0066507904177079994
PSNR: 9.941911138887942
LPIPS: 0.7267210914029015
MSE: 0.10534941032528877
FID: 374.74319629954533


# pose
### output_adds ###
Mean Metrics:
SSIM: 0.12896427676141747
PSNR: 13.67742765134191
LPIPS: 0.3189569987356663
MSE: 0.05818643076345324
FID: 85.84288524495071

### output_org ###
Mean Metrics:
SSIM: 0.10581640830258723
PSNR: 13.019833711912401
LPIPS: 0.310723689571023
MSE: 0.06046676100231707
FID: 78.73171516340373

### output_wo_cross ###
Mean Metrics:
SSIM: 0.012002310223053488
PSNR: 9.824413832699117
LPIPS: 0.516722677482499
MSE: 0.108858123421669
FID: 372.8642673041471
'''


# obama
'''
# light
### output_adds ###
Mean Metrics:
SSIM: 0.22716956291487658
PSNR: 15.498265369167576
LPIPS: 0.23744978714320394
MSE: 0.058199308812618256
FID: 72.92013214476137

### output_org ###
Mean Metrics:
SSIM: 0.20402389103526009
PSNR: 13.50228382779078
LPIPS: 0.23554450646042824
MSE: 0.06700598658062518
FID: 115.21247352949466

### output_wo_cross ###
Mean Metrics:
SSIM: 0.0785104065915181
PSNR: 13.846425202144909
LPIPS: 0.5441479914718204
MSE: 0.06149139508811964
FID: 166.2228058843974


# exp
### output_adds ###
Mean Metrics:
SSIM: 0.24194170409475538
PSNR: 17.536866908177274
LPIPS: 0.234453023592631
MSE: 0.05901409717302562
FID: 73.27658505710102

### output_org ###
Mean Metrics:
SSIM: 0.2436976950153509
PSNR: 14.176766385123926
LPIPS: 0.23467091802093717
MSE: 0.05927335489023891
FID: 89.54637283896066

### output_wo_cross ###
Mean Metrics:
SSIM: 0.0668884869627509
PSNR: 14.068110257847865
LPIPS: 0.6025895294215944
MSE: 0.061619753328462444
FID: 166.83550674948313


# pose
### output_adds ###
Mean Metrics:
SSIM: 0.1399371368993222
PSNR: 13.27293787327756
LPIPS: 0.2843097596499655
MSE: 0.07271001348271966
FID: 106.88476117963243

### output_org ###
Mean Metrics:
SSIM: 0.13685219882130945
PSNR: 12.522392593112983
LPIPS: 0.2666451210776965
MSE: 0.07207332116862138
FID: 81.3416861149692

### output_wo_cross ###
Mean Metrics:
SSIM: 0.08627695506855865
PSNR: 13.473710616092099
LPIPS: 0.37205933696693844
MSE: 0.06693932701212664
FID: 110.96844884597793
'''

# biden
'''
# light
### output_biden ###
Mean Metrics:
SSIM: 0.15327816517521592
PSNR: 13.621000858810897
LPIPS: 0.2609451709315181
MSE: 0.06224543231073767
FID: 77.8919404736543

### output_org_biden ###
Mean Metrics:
SSIM: 0.17646924478102352
PSNR: 13.420791700830147
LPIPS: 0.2660720381885767
MSE: 0.06093854936771095
FID: 89.15877451280687

### output_biden_wo_cross ###
Mean Metrics:
SSIM: 0.18797435879814542
PSNR: 13.605153688473447
LPIPS: 0.28119352757930755
MSE: 0.06226013309787959
FID: 125.3361943913801


# exp
### output_biden ###
Mean Metrics:
SSIM: 0.19773493793996838
PSNR: 15.73086920125398
LPIPS: 0.23850265555083752
MSE: 0.05939401758369058
FID: 59.94636167463264

### output_org_biden ###
Mean Metrics:
SSIM: 0.22255739011331013
PSNR: 15.177438236686276
LPIPS: 0.24681356139481067
MSE: 0.05927702396875247
FID: 81.79904372742081

### output_biden_wo_cross ###
Mean Metrics:
SSIM: 0.1557485407434464
PSNR: 16.010971534414857
LPIPS: 0.28769020177423954
MSE: 0.058151428576093164
FID: 88.86816177234246


# pose
### output_biden ###
Mean Metrics:
SSIM: 0.11775525182231153
PSNR: 12.078617799344192
LPIPS: 0.31366600580513476
MSE: 0.07673243330791593
FID: 111.54175223136671

### output_org_biden ###
Mean Metrics:
SSIM: 0.11108379407267807
PSNR: 11.839711684992816
LPIPS: 0.2850599218159914
MSE: 0.0730260482057929
FID: 84.1417248980951

### output_biden_wo_cross ###
Mean Metrics:
SSIM: 0.11726348349509161
PSNR: 12.040320582465183
LPIPS: 0.3269725203514099
MSE: 0.07581957560032607
FID: 124.55580617461156
'''
