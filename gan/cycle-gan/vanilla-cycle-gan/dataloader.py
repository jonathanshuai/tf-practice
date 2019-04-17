import numpy as np
import cv2
import os

def inverse_transform(images):
    return ((images + 1) / 2).clip(0, 1)
def transform(images):
    return ((images / 255) * 2 - 1).clip(-1, 1)

def random_flip(image, p=0.5):
    if np.random.random() <= p:
        return cv2.flip(image, 1)
    else:
        return image
    
def random_crop(image, crop_size=128):
    if crop_size == image.shape[0] and crop_size == image.shape[1]:
        return image
    
    y = np.random.randint(image.shape[0] - crop_size)
    x = np.random.randint(image.shape[1] - crop_size)
    
    return image[y:y + crop_size, x:x + crop_size]

def load_image(file_list):
    return cv2.cvtColor(cv2.imread(np.random.choice(file_list)), cv2.COLOR_BGR2RGB)

class CycleDataLoader():
    def __init__(self, file_dir_a, file_dir_b, batch_size=1, crop_size=128, channels=3):
        self.file_list_a = []
        self.file_list_b = []
        
        for root, directory, files in os.walk(file_dir_a):
            for file in files:
                self.file_list_a.append(os.path.join(file_dir_a, file))
                
        for root, directory, files in os.walk(file_dir_b):
            for file in files:
                self.file_list_b.append(os.path.join(file_dir_b, file))
        
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.channels = channels
        
    def get_batch(self):
        batch_a = np.zeros((self.batch_size, self.crop_size, self.crop_size, self.channels))
        batch_b = np.zeros((self.batch_size, self.crop_size, self.crop_size, self.channels))

        for i in range(self.batch_size):
            batch_a[i] = random_flip(random_crop(load_image(self.file_list_a), self.crop_size))
            batch_b[i] = random_flip(random_crop(load_image(self.file_list_b), self.crop_size))
        
        return transform(batch_a), transform(batch_b)