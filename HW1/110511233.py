# %%
import cv2 
import numpy as np

# %%
image = cv2.imread('building.jpg')

# %%
def get_rotmtx(deg):
    ang = deg * np.pi/180
    return np.array([[np.cos(ang), -np.sin(ang)], 
                     [np.sin(ang), np.cos(ang)]])

def nearest(image, x, y):
    return image[int(y), int(x)]

def bilinear(image, x, y):
    h, w = image.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    
    return (image[y0, x0] * (x1 - x) * (y1 - y) +
            image[y0, x1] * (x - x0) * (y1 - y) +
            image[y1, x0] * (x1 - x) * (y - y0) +
            image[y1, x1] * (x - x0) * (y - y0)).round().astype(np.uint8) \
    if x1 < w and y1 < h else image[y0, x0]
    
def bicubic(image, x, y):   
    def cubic_interpolation(p, x):
        coef = np.array([[-1/2,  3/2, -3/2,  1/2], 
                         [1   , -5/2,    2, -1/2], 
                         [-1/2,    0,  1/2,    0], 
                         [0   ,    1,    0,    0]])

        x = np.array([x**3, x**2, x, 1])
        
        return (coef @ p).T @ x
        
    def get_points_x(img, y, x0, x1, x2, x3):
        return np.array([img[y, x0], 
                         img[y, x1], 
                         img[y, x2], 
                         img[y, x3]], dtype=np.int32)
        
    h, w = image.shape[:2]
    
    x0, y0 = int(x) - 1, int(y) - 1
    x1, y1 = x0 + 1, y0 + 1
    x2, y2 = x1 + 1, y1 + 1
    x3, y3 = x2 + 1, y2 + 1
    
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x2 >= h: x2 = h - 1
    if y2 >= w: y2 = w - 1
    if x3 >= h: x3 = h - 1
    if y3 >= w: y3 = w - 1
        
    dx = x - x1
    dy = y - y1
    
    p = get_points_x(image, y0, x0, x1, x2, x3)
    q0 = cubic_interpolation(p, dx)
    
    p = get_points_x(image, y1, x0, x1, x2, x3)
    q1 = cubic_interpolation(p, dx)
    
    p = get_points_x(image, y2, x0, x1, x2, x3)
    q2 = cubic_interpolation(p, dx)
    
    p = get_points_x(image, y3, x0, x1, x2, x3)
    q3 = cubic_interpolation(p, dx)
    
    return np.clip(cubic_interpolation(np.array([q0, q1, q2, q3]), dy), 0, 255).astype(np.uint8)
    
def rotate_image_cw(image, deg, method='nearest'):
    if method == 'nearest':
        method = nearest
    elif method == 'bilinear':
        method = bilinear
    elif method == 'bicubic':
        method = bicubic
    
    h, w, c = image.shape
    new_image = np.zeros((h, w, c), dtype=np.uint8)
    
    x_center, y_center = w//2, h//2
    rotmtx = get_rotmtx(-deg)
    for yy in range(h):
        for xx in range(w):
            x, y = rotmtx @ np.array([xx - x_center, yy - y_center]) + \
                            np.array([x_center, y_center])
            if 0 <= y < h and 0 <= x < w:
                new_image[yy, xx] = method(image, x, y)
                
    return new_image

def upscale_image_2x(image, method='nearest'):
    if method == 'nearest':
        method = nearest
    elif method == 'bilinear':
        method = bilinear
    elif method == 'bicubic':
        method = bicubic
        
    h, w, c = image.shape
    new_image = np.zeros((h*2, w*2, c), dtype=np.uint8)
    
    for yy in range(h*2):
        for xx in range(w*2):
            new_image[yy, xx] = image[yy//2, xx//2] if xx%2 == 0 and yy%2 == 0 else method(image, xx/2, yy/2)
            
    return new_image

# %%
print('Rotating and upscaling images...')

cv2.imwrite('rotated_nearest_neighbor.jpg', 
            rotate_image_cw(image, 30, 'nearest'))
print('Saved rotated_nearest_neighbor.jpg')

cv2.imwrite('rotated_bilinear.jpg', 
            rotate_image_cw(image, 30, 'bilinear'))
print('Saved rotated_bilinear.jpg')

cv2.imwrite('rotated_bicubic.jpg', 
            rotate_image_cw(image, 30, 'bicubic'))
print('Saved rotated_bicubic.jpg')

cv2.imwrite('upscaled_nearest_neighbor.jpg', 
            upscale_image_2x(image, 'nearest'))
print('Saved upscaled_nearest_neighbor.jpg')

cv2.imwrite('upscaled_bilinear.jpg', 
            upscale_image_2x(image, 'bilinear'))
print('Saved upscaled_bilinear.jpg')

cv2.imwrite('upscaled_bicubic.jpg', 
            upscale_image_2x(image, 'bicubic'))
print('Saved upscaled_bicubic.jpg')