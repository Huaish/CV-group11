import cv2
import numpy as np
from filter import gaussian_filter
import time
import os
from utils import save_image_grid

def edge_detection(image):
    return cv2.Canny(image, 100, 200)

def generate_image_pyramid(img, levels, scaling_factor=0.75):
    pyramid_images = [img]

    current_image = img
    for i in range(levels):
        if i % 5 == 0 and i > 0:
            current_image = gaussian_filter(current_image, cutoff_frequency=20, low_pass=False)
        new_size = (int(current_image.shape[1] * scaling_factor), int(current_image.shape[0] * scaling_factor))
        downscaled = cv2.resize(current_image, new_size, interpolation=cv2.INTER_LINEAR)
        if downscaled.shape[0] < 2 or downscaled.shape[1] < 2:
            break
        pyramid_images.append(downscaled)
        current_image = downscaled
        
    return pyramid_images[-1]

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def align_by_translation(reference, image_to_align, max_offset=20):
    best_offset = (0, 0)
    min_error = float('inf')
    
    for y_offset in range(-max_offset, max_offset + 1):
        for x_offset in range(-max_offset, max_offset + 1):
            # 對圖像進行平移
            translated_image = np.roll(image_to_align, shift=(y_offset, x_offset), axis=(0, 1))
            
            # 計算平移後圖像與基準圖像的均方誤差
            error = mse(reference, translated_image)
            
            # 找到最小誤差的位置
            if error < min_error:
                min_error = error
                best_offset = (y_offset, x_offset)
    
    # 對圖像應用最佳平移
    aligned_image = np.roll(image_to_align, shift=best_offset, axis=(0, 1))
    
    return aligned_image, best_offset

# ===================== Direct Add =====================
def direct_add(image):
    h, w = image.shape

    div = h//3
    size = image[:div, :]
    
    blue = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)
    green = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)
    red = np.zeros((size.shape[0], size.shape[1], 3), dtype=np.uint8)

    blue[:, :, 0] = image[:div, :]
    green[:, :, 1] = image[div:2*div, :]
    red[:, :, 2] = image[2*div:3*div, :]

    output = cv2.add(red, green)  # add red and green
    output = cv2.add(output, blue)    # add blue
    return output

# ===================== MSE Align =====================
def mse_align(channel, ref_channel="blue"):
    blue_channel, green_channel, red_channel = channel
    
    ref_channel = blue_channel if ref_channel == "blue" else green_channel if ref_channel == "green" else red_channel
    offsets = [(0, 0), (0, 0), (0, 0)]
    for i, align_channel in enumerate([blue_channel, green_channel, red_channel]):
        if align_channel is ref_channel:
            continue
        _, offset = align_by_translation(ref_channel, align_channel)
        offsets[i] = offset
    color_image = np.dstack((np.roll(blue_channel, shift=offsets[0], axis=(0, 1)),
                             np.roll(green_channel, shift=offsets[1], axis=(0, 1)),
                             np.roll(red_channel, shift=offsets[2], axis=(0, 1)),
                             ))
    return color_image

# ===================== Edge Align =====================
def canny_edge_align(channel, ref_channel="blue"):
    blue_channel, green_channel, red_channel = channel
    
    blue_edges = edge_detection(blue_channel)
    green_edges = edge_detection(green_channel)
    red_edges = edge_detection(red_channel)
    
    ref_edges = blue_edges if ref_channel == "blue" else green_edges if ref_channel == "green" else red_edges
    offsets = [(0, 0), (0, 0), (0, 0)]
    for i, align_edges in enumerate([blue_edges, green_edges, red_edges]):
        if align_edges is ref_edges:
            continue
        _, offset = align_by_translation(ref_edges, align_edges)
        offsets[i] = offset
    color_image = np.dstack((np.roll(blue_channel, shift=offsets[0], axis=(0, 1)),
                             np.roll(green_channel, shift=offsets[1], axis=(0, 1)),
                             np.roll(red_channel, shift=offsets[2], axis=(0, 1)),
                             ))
    return color_image

# ===================== Pyramid Align =====================
def gaussian_pyramid_align(image, channel, level=10, scaling_factor=0.75, max_offset=20, ref_channel="blue"):
    pyramid_image = generate_image_pyramid(image, levels=level, scaling_factor=scaling_factor)
    height = pyramid_image.shape[0] // 3
    blue_pyramid = pyramid_image[0:height]
    green_pyramid = pyramid_image[height:2*height]
    red_pyramid = pyramid_image[2*height:3*height]
    
    ref_pyramid = blue_pyramid if ref_channel == "blue" else green_pyramid if ref_channel == "green" else red_pyramid
    offsets = [(0, 0), (0, 0), (0, 0)]
    scaling_total = image.shape[0] / pyramid_image.shape[0]
    for i, align_channel in enumerate([blue_pyramid, green_pyramid, red_pyramid]):
        if align_channel is ref_pyramid:
            continue
        _, offset = align_by_translation(ref_pyramid, align_channel, max_offset=max_offset)
        offset = (offset[0] * scaling_total, offset[1] * scaling_total)
        offsets[i] = offset
    
    color_image = np.dstack((np.roll(channel[0], shift=offsets[0], axis=(0, 1)),
                            np.roll(channel[1], shift=offsets[1], axis=(0, 1)),
                            np.roll(channel[2], shift=offsets[2], axis=(0, 1)),
                         ))
    
    return color_image

if __name__ == "__main__":
    # Read image
    image_path = 'my_data/01725u.jpg'
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)    
    img_list, text_list = [], []
    
    # Split the image into three channels
    height = image.shape[0] // 3
    channel = [image[0:height], image[height:2*height], image[2*height:3*height]]
    
    # Direct add
    start_time = time.time()
    color_img = direct_add(image)
    end_time = time.time()
    print('Direct Add time:', end_time - start_time)
    cv2.imwrite(f'output/task3_{image_name}_add.jpg', color_img)
    img_list.append(color_img)
    text_list.append(f'Direct Add time: {end_time - start_time:.2f}s')
    
    # MSE alignment
    start_time = time.time()
    color_img = mse_align(channel, ref_channel="blue")
    end_time = time.time()
    print('MSE alignment time:', end_time - start_time)
    cv2.imwrite(f'output/task3_{image_name}_mse.jpg', color_img)
    img_list.append(color_img)
    text_list.append(f'MSE alignment time: {end_time - start_time:.2f}s')
    
    # Edge alignment
    start_time = time.time()
    color_img = canny_edge_align(channel, ref_channel="blue")
    end_time = time.time()
    print('Edge alignment time:', end_time - start_time)
    cv2.imwrite(f'output/task3_{image_name}_edge.jpg', color_img)
    img_list.append(color_img)
    text_list.append(f'Edge alignment time: {end_time - start_time:.2f}s')
    
    # Pyramid alignment (Our method)
    start_time = time.time()
    color_img = gaussian_pyramid_align(image, channel, level=10, scaling_factor=0.75, max_offset=20, ref_channel="blue")
    end_time = time.time()
    print('Pyramid alignment time:', end_time - start_time)
    cv2.imwrite(f'output/task3_{image_name}_pyramid.jpg', color_img)
    img_list.append(color_img)
    text_list.append(f'Pyramid alignment time: {end_time - start_time:.2f}s')
    
    save_image_grid(img_list, text_list, rows=1, cols=len(img_list), target_size=(image.shape[0] // 3, image.shape[1]), output_path=f'output/task3_colorized_comparison.jpg')