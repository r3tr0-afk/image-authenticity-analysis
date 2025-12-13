import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_srm_kernels():
    import numpy as np

    KV3 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    KB3 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    SPAM11 = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float32)
    SQUARE_3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    return [torch.from_numpy(k) for k in [KV3, KB3, SPAM11, SQUARE_3]]

def apply_minmax(x):
    dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    return dil - ero

def visualize_noise_branch(real_path, fake_path):
    paths = [real_path, fake_path]
    titles = ["Real", "Fake"]
    kernels = get_srm_kernels()
    kernel_names = ["KV3 (Vertical)", "KB3 (Horizontal)", "SPAM11 (Texture)", "Square (Point)"]
    
    srm_weights = torch.stack(kernels).unsqueeze(1) 
    
    plt.figure(figsize=(16, 10))
    
    for idx, path in enumerate(paths):
        try:
            img = Image.open(path).convert('RGB').resize((256, 256))
            x = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        g_channel = x[:, 1:2, :, :] 
        
        srm_out = F.conv2d(g_channel, srm_weights, padding=1)
        
        mm_out = apply_minmax(x)
        mm_vis = torch.mean(mm_out, dim=1, keepdim=True)

        row_offset = idx * 2 
        
        plt.subplot(4, 6, row_offset * 6 + 1)
        plt.imshow(img)
        plt.title(f"{titles[idx]} Image")
        plt.axis('off')
        
        plt.subplot(4, 6, row_offset * 6 + 2)
        plt.imshow(mm_vis.squeeze(), cmap='gray', vmin=0, vmax=0.5) 
        plt.title("MinMax (Structure)")
        plt.axis('off')
        
        for k in range(4):
            noise_map = torch.tanh(srm_out[0, k]) 
            
            plt.subplot(4, 6, row_offset * 6 + 3 + k)
            plt.imshow(noise_map, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f"{kernel_names[k]}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    
visualize_noise_branch("../data/raw/real/real_000246.jpg", "../data/raw/fake/fake_000010.jpg")    