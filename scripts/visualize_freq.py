import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import fft2, fftshift

def get_radial_profile(img_gray):
    h, w = img_gray.shape

    f = fft2(img_gray)
    fshift = fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    y, x = np.indices((h, w))
    center = np.array([h // 2, w // 2])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())

    radial_profile = tbin / np.maximum(nr, 1)

    radial_profile_log = np.log10(1 + radial_profile)
    
    return magnitude_spectrum, radial_profile_log

def visualize_frequency_branch(real_path, fake_path=None):
    paths = [real_path]
    titles = ["REAL"]
    if fake_path:
        paths.append(fake_path)
        titles.append("FAKE")
        
    plt.figure(figsize=(12, 14))

    profiles = []
    
    for i, path in enumerate(paths):
        try:
            img = Image.open(path).convert('RGB').resize((256, 256))
            img_np = np.array(img) / 255.0

            gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]

            mag_spec, profile = get_radial_profile(gray)
            profiles.append(profile)

            plt.subplot(4, 2, i + 1)
            plt.imshow(img_np)
            plt.title(f"{titles[i]} - RGB Crop")
            plt.axis('off')

            plt.subplot(4, 2, i + 3)
            plt.imshow(np.log10(1 + mag_spec), cmap='inferno')
            plt.title(f"{titles[i]} - 2D FFT Magnitude")
            plt.axis('off')
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

    if len(profiles) > 0:
        x_axis = np.arange(1, len(profiles[0])) 

        plt.subplot(4, 1, 3)
        plt.plot(x_axis, profiles[0][1:], label=titles[0], color='blue', linewidth=2, alpha=0.8)
        if len(profiles) > 1:
            plt.plot(x_axis, profiles[1][1:], label=titles[1], color='red', linewidth=2, alpha=0.8)

        plt.title("1D Radial Power Spectrum (Log Scale)")
        plt.ylabel("Log Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(4, 1, 4)

        whitened_real = np.diff(profiles[0][1:])

        x_axis_white = x_axis[:-1]
        
        plt.plot(x_axis_white, whitened_real, label=f"{titles[0]} (Derivative)", color='blue', linewidth=1.5, alpha=0.7)
        
        if len(profiles) > 1:
            whitened_fake = np.diff(profiles[1][1:])
            plt.plot(x_axis_white, whitened_fake, label=f"{titles[1]} (Derivative)", color='red', linewidth=1.5, alpha=0.7)

        plt.title("Derivative of Spectrum")
        plt.xlabel("Frequency (Radius from Center)")
        plt.ylabel("Change in Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

visualize_frequency_branch("../data/raw/real/real_000246.jpg", "../data/raw/fake/fake_000010.jpg")