# %%
import cv2 
import numpy as np

# %%
def get_hist(img_one_channel):
    hist = np.zeros(256, dtype=np.float32)
    
    for y in range(img_one_channel.shape[0]):
        for x in range(img_one_channel.shape[1]):
            hist[img_one_channel[y,x]] += 1

    for i in range(1, 256):
        hist[i] += hist[i-1]
    
    max_val = img_one_channel.shape[0] * img_one_channel.shape[1]
    
    result = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        result[i] = np.round(255.0 * (hist[i] / max_val)).astype(np.uint8)       
        
    return result

# %%
def hist_equalization(img_one_channel):
    result = np.zeros_like(img_one_channel)
    hist = get_hist(img_one_channel)
        
    for y in range(img_one_channel.shape[0]):
        for x in range(img_one_channel.shape[1]):
            result[y,x] = hist[img_one_channel[y,x]]
            
    return result

# %%
def hist_specification(src_img, ref_img):
    T = get_hist(src_img)
    G = get_hist(ref_img)

    H = np.zeros(256, dtype=np.uint8)
    j = 1    
    for a in range(256):
        while T[a] > G[j]:
            j += 1
            
        if not T[a] == G[j]:
            j -= 1

        H[a] = j
        
    # apply histogram specification
    dst_img = np.zeros_like(src_img)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            dst_img[i,j] = H[src_img[i,j]]
            
    return dst_img

# %%
img = cv2.imread('Q1.jpg', cv2.IMREAD_GRAYSCALE)

res_img = hist_equalization(img)
cv2.imwrite('Q1_processed.jpg', res_img)
print('Q1_processed.jpg saved')

# %%
src_img = cv2.imread('Q2_source.jpg', cv2.IMREAD_GRAYSCALE)
ref_img = cv2.imread('Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)

res_img = hist_specification(src_img, ref_img)  
cv2.imwrite('Q2_processed.jpg', res_img)
print('Q2_processed.jpg saved')


