import os
import cv2
import random
import numpy as np

from PIL import Image
import skimage as sk
from imgaug import augmenters as iaa
from clustercontrast.utils.data import transforms as T

import warnings
warnings.simplefilter("ignore", UserWarning)

# ########################################################
# sunny
# ########################################################
def sunny(x, severity=1):
    c = [.3, .45, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

# ########################################################
# fog
# ########################################################
def fog(x, severity=1):
    c = [(1.5, 1.5), (2.5, 2), (3.5, 2.5)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:288, :144][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

# ########################################################
# frost
# ########################################################
def frost(x, severity=1):
    c = [(0.9, 0.6),
         (0.7, 0.6),
         (0.5, 0.6)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./data_corruptions/frost/frost1.png', './data_corruptions/frost/frost2.png', './data_corruptions/frost/frost3.png', 
                './data_corruptions/frost/frost4.jpg', './data_corruptions/frost/frost5.jpg', './data_corruptions/frost/frost6.jpg'][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - 288), np.random.randint(0, frost.shape[1] - 144)
    frost = frost[x_start:x_start + 288, y_start:y_start + 144][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

# ########################################################
# rain
# ########################################################
def rain(x, severity=1):
    def get_noise(img, value=10): 
        noise = np.random.uniform(0, 256, img.shape[0:2])
        v = value * 0.01
        noise[np.where(noise < (256 - v))] = 0
        k = np.array([[0, 0.1, 0],
                      [0.1, 8, 0.1],
                      [0, 0.1, 0]])
    
        noise = cv2.filter2D(noise, -1, k)
        return noise

    def rain_blur(noise, length=10, angle=0,w=1):
        trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)  
        dig = np.diag(np.ones(length))
        k = cv2.warpAffine(dig, trans, (length, length)) 
        k = cv2.GaussianBlur(k,(w,w),0)
        blurred = cv2.filter2D(noise, -1, k) 
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    
    def alpha_rain(rain,img,beta = 0.8):
        rain = np.expand_dims(rain,2)
        rain_effect = np.concatenate((img,rain),axis=2)
        rain_result = img.copy()    
        rain = np.array(rain,dtype=np.float32)    
        rain_result[:,:,0]= rain_result[:,:,0] * (255-rain[:,:,0])/255.0 + beta*rain[:,:,0]
        rain_result[:,:,1] = rain_result[:,:,1] * (255-rain[:,:,0])/255 + beta*rain[:,:,0] 
        rain_result[:,:,2] = rain_result[:,:,2] * (255-rain[:,:,0])/255 + beta*rain[:,:,0]
        return rain_result

    c = [450, 650, 850][severity - 1]
    noise = get_noise(x, value=c)
    rain = rain_blur(noise, length=25, angle=random.randint(-25, 25), w=3)
    rain_result = alpha_rain(rain, x, beta=0.6).astype(np.float16)
    return rain_result

# ########################################################
# snow
# ########################################################
def snow(x, severity=1):
    # Apply the snow effect using imgaug
    seq = iaa.Sequential([
        iaa.imgcorruptlike.Snow(severity=severity),  # Snow corruption with severity
        iaa.Snowflakes(flake_size=(0.03, 0.01), speed=(0.03, 0.01))  # Snowflakes with specified size and speed
    ])

    # Apply the transformation
    snow_result = seq(image=x).astype(np.float16)/2 + x/2
    # Return the transformed image
    return snow_result

# ########################################################
# General Corrupt Class
# ########################################################
class Corrupt:
    def __init__(self, corrupt_type="sunny", severity=1):
        """
        Initialize the Corrupt class to apply different types of image corruption.
        :param corrupt_type: Type of corruption (e.g., 'gaussian_noise', 'snow', 'brightness', etc.).
        :param severity: Severity level of the corruption (typically 1, 2, or 3).
        """
        self.corrupt_type = corrupt_type
        self.severity = severity
        self.corrupt_obj = None
            
        if self.corrupt_type == 'sunny':
            self.corrupt_obj = sunny
        elif self.corrupt_type == 'snow':
            self.corrupt_obj = snow
        elif self.corrupt_type == 'rain':
            self.corrupt_obj = rain
        elif self.corrupt_type == 'frost':
            self.corrupt_obj = frost
        elif self.corrupt_type == 'fog':
            self.corrupt_obj = fog
        
    def __call__(self, img):
        """
        Apply the selected corruption function to the image.
        :param img: Input image (PIL Image or NumPy array).
        :return: Corrupted image (NumPy array or PIL Image).
        """
        if isinstance(img, np.ndarray):
            print("Heere")
            corrupt_img = self.corrupt_obj(img, self.severity).astype(np.float32)
            return corrupt_img
        elif isinstance(img, Image.Image):
            corrupt_img = self.corrupt_obj(np.array(img), self.severity).astype(np.uint8)
            return Image.fromarray(corrupt_img)
        else:
            raise TypeError("Input must be a PIL Image or NumPy array")

if __name__ == "__main__":
    data_type = "PIL.Image"
    
    TYPE = ['sunny', 'fog', 'frost', 'rain', 'snow']
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    data_dir = "./data_corruptions/examples/"

    for filename in os.listdir(data_dir):
        print(filename)
        if filename.lower().endswith(image_extensions):
            for ct in TYPE:
                for severity in [1, 2, 3]:
                    corrupt_type = ct
                    ImageCorrupt = Corrupt(corrupt_type=corrupt_type, severity=severity)
                    transform = T.Compose([
                        T.Resize((288, 144), interpolation=3),
                        ImageCorrupt,
                    ])

                    if data_type == "PIL.Image":
                        img_path = os.path.join(data_dir, filename)
                        img = Image.open(img_path)
                        transformed_img = transform(img)
                        output_path = f"./data_corruptions/examples_results/ir_{corrupt_type}_severity_{severity}_{filename}"
                        transformed_img.save(output_path)
                    elif data_type == "numpy":
                        img_path = os.path.join(data_dir, filename)
                        img = cv2.imread(img_path)
                        transformed_img = transform(img)
                        output_path = f"./data_corruptions/examples_results/{corrupt_type}_severity_{severity}_{filename}"
                        cv2.imwrite(output_path, transformed_img)