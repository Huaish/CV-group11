import numpy as np

def ideal_filter(image, cutoff_frequency, low_pass=True):
    # Perform FFT to move to frequency domain
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    
    # Create ideal filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if low_pass:
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff_frequency:
                    mask[i, j] = 1
            else:
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) > cutoff_frequency:
                    mask[i, j] = 1
                    
    # Apply filter to image
    dft_shift *= mask
    dft_shift = np.fft.ifftshift(dft_shift)
    ideal_filtered = np.fft.ifft2(dft_shift)
    ideal_filtered = np.real(ideal_filtered)
    
    return ideal_filtered

def gaussian_filter(image, cutoff_frequency, low_pass=True):
    # Perform FFT to move to frequency domain
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Create Gaussian filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = np.exp(-((i - crow) ** 2 + (j - ccol) ** 2) / (2 * cutoff_frequency ** 2))
            if not low_pass:
                mask[i, j] = 1 - mask[i, j]
                
    # Apply filter to image
    dft_shift *= mask
    dft_shift = np.fft.ifftshift(dft_shift)
    gaussian_filtered = np.fft.ifft2(dft_shift)
    gaussian_filtered = np.real(gaussian_filtered).astype(np.uint8)
    
    return gaussian_filtered

def filter(image, cutoff_frequency, low_pass=True, filter_type='ideal'):
    if filter_type == 'ideal':
        return ideal_filter(image, cutoff_frequency, low_pass)
    elif filter_type == 'gaussian':
        return gaussian_filter(image, cutoff_frequency, low_pass)
    else:
        raise ValueError('Invalid filter type')