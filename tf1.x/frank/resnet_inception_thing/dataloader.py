from functools import reduce

import cv2
import numpy as np

# Define some data transformations
# Note: Only used cv2 here, but for other augmentations look here:
# http://www.scipy-lectures.org/advanced/image_processing/

# Add some custom transformations for data augmentation

class Transform(object):
    def __init__(self, **kwargs):
        pass

    def transform(self, image):
        return image

class Blur(Transform):
    def __init__(self, size=(7, 7), sig=(0.788, 0.788)):
        """Returns a transformation to apply Gaussian blur.
        size                (int): Size for Gaussian blur.
        sig               (float): Maximum sig for Gaussian blur.
        """
        self._size = size
        self._sig = sig

    def apply_blur(self, image):
        """Returns a image with random Gaussian blur applied.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        """
        image = cv2.GaussianBlur(image, self._size, **self._sig)
        return image


class RandomHueAdd(Transform):
    def __init__(self, min_add=0, max_add=10):
        """Returns a transformation to apply random hue add.
        min_add          (int): Minimum amount of add
        max_add          (int): Maximum amount of add.
        """
        self._min_add = min_add
        self._max_add = max_add


    def transform(self, image):
        """Returns an image with a random hue add applied.
        image     (numpy.ndarray): Image to apply transformation to.
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] += \
            np.uint8(np.random.random() * (self._max_add - self._min_add)) + self._min_add

        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


class RandomSaturation(Transform):
    def __init__(self, min_scale=0.8, max_scale=1.2):
        """Returns an image with random hue scale.
        image     (numpy.ndarray): Image in the form of 3d array to apply noise to.
        min_scale          (int): Minimum scale amount
        max_scale          (int): Maximum scale amount.
        """

        self._min_scale = min_scale
        self._max_scale = max_scale

    def transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        scale = np.random.random() * (self._max_scale - self._min_scale) + self._min_scale
        new_saturation = image[:, :, 1] * scale
        image[:, :, 1] = np.clip(new_saturation, 0, 255).astype('uint8')

        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

class RandomBrightness(Transform):
    def __init__(self, min_add=0, max_add=120):
        """Returns a transformation to apply brightness.
        min_add          (int): Minimum amount of add
        max_add          (int): Maximum amount of add.
        """
        self._min_add = min_add
        self._max_add = max_add

    def transform(self, image):
        """Returns an image with random brightness add.
        image     (numpy.ndarray): Image to apply transform to.
        """

        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        new_value = \
            image[:, :, 2] + np.random.random() * (self._max_add - self._min_add) + self._min_add

        image[:, :, 2] = np.clip(new_value, 0, 255).astype('uint8')

        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

class SPNoise(Transform):
    def __init__(self, prob=0.20, sp_ratio=0.5):
        """Returns a transformation to apply salt and pepper noise.
        prob              (float): Probability of adding either salt or pepper to a
        pixel.
        sp_ratio          (float): Ratio between salt and pepper.
        """

        self._prob = prob
        self._sp_ratio = sp_ratio

    def transform(self, image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    random = np.random.random()
                    if random <= self._salt_prob:
                        image[i, j, k] = 255
                    elif random <= self._prob:
                        image[i, j, k] = 0

        return image


class GaussNoise(Transform):
    def __init__(self, mean=0, std=30):
        """Returns a image with random Gaussian noise applied.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        mean              (float): Mean for Gaussian noise.
        std               (float): Standard deviation for Gaussian noise.
        """
        self._mean = mean
        self._std = std

  
    def transform(self, image):
        noise = np.random.normal(self._mean, self._std, image.shape)
        image = np.add(image, noise.astype('int'))
        image = np.clip(image, 0, 255)

        return image

class RandomRotate(Transform):
    def __init__(self, degrees=180):
        """ Returns an image rotated by a random degree.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        degrees           (float): Rotation by random amount will be in range
        [-degrees, +degrees].
        """
        self._degrees = degrees

    def transform(self, image):
        width = image.shape[1]
        height = image.shape[0]

        to_rotate = 2 * np.random.random() * self._degrees - self._degrees
        M = cv2.getRotationMatrix2D((width / 2, height / 2), to_rotate, 1)
        image = cv2.warpAffine(image, M, (width, height))

        return image

class RandomTranslate(Transform):
    def __init__(self, max_ratio=0.30):
        """ Returns an image translated by a random amount.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        max_ratio         (float): Translation amount will be in range
        [-max_ratio, max_ratio] * size.
        """
        self._max_ratio = max_ratio

    def transform(self, image):
        side_length = image.shape[0]
        max_trans = side_length * self._max_ratio

        x_trans = 2 * np.random.random() * max_trans - max_trans
        y_trans = 2 * np.random.random() * max_trans - max_trans

        M = np.float32([[1, 0, x_trans],
                        [0, 1, y_trans]])

        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        return image

class RandomFlip(Transform):
    def __init__(self):
        """Returns an image randomly flipped.
        """
        
    def transform(self, image):
        axis = np.random.choice(3)
        return cv2.flip(image, axis - 1)
    
class RandomCropResize(Transform):
    def __init__(self, min_ratio=0.20, max_ratio=0.40):
        """ Returns an image cropped and resized by a random amount.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        max_ratio         (float): Crop resize amount will be in range
        [-max_ratio, max_ratio] * size.
        """

        self._min_ratio = min_ratio
        self._max_ratio = max_ratio

    def transform(self, image):
        width = image.shape[1]
        height = image.shape[0]

        ratio = np.random.random() * (self._max_ratio - self._min_ratio) + self._min_ratio

        x_margin = int(ratio * width // 2)
        y_margin = int(ratio * width // 2)

        x_lower, x_upper = x_margin, width - x_margin
        y_lower, y_upper = y_margin, height - y_margin

        cropped_image = image[y_lower:y_upper, x_lower:x_upper]
        resized_image = cv2.resize(cropped_image, (width, height))

        return resized_image

class RandomShear(Transform):
    def __init__(self, degrees=20):
        """ Returns an image sheared by a random degrees.
        image     (numpy.ndarray): Image in the form of 3d array to apply
        transformation to.
        degrees           (float): Random shear amount will be in range
        [-degrees, +degrees].
        """

        self._degrees = degrees

    def transform(self, image):
        width = image.shape[1]
        height = image.shape[0]

        x_center = width // 2
        y_center = height // 2

        to_rotate = 2 * np.random.random() * self._degrees - self._degrees
        to_rotate_rad = (2 * np.pi / 360) * to_rotate
        shift_amt = y_center * np.tan(to_rotate_rad)

        pts1 = np.float32([[0, y_center],
                           [width, y_center],
                           [x_center, height]])

        pts2 = np.float32([[0, y_center],
                           [width, y_center],
                           [x_center + shift_amt, height]])

        M = cv2.getAffineTransform(pts1, pts2)

        image = cv2.warpAffine(image, M, (width, height))

        return image

class ColorJitter(Transform):
    def __init__(self, hue_range=[0, 10],
                       saturation_range=[0.8, 1.2],
                       brightness_range=[0, 100]):
        """ Returns an image with random color jitter.
        image              (numpy.ndarray): Image in the form of 3d array to be
        transformed.
        hue_range             ([int, int]): Range [min, max] to add to hue.
        saturation_range  ([float, float]): Range [min, max] to scale saturation.
        brightness_range      ([int, int]): Range [min, max] to add to brightness.
        """
        self.apply_hue_add = RandomHueAdd(*hue_range)
        self.apply_saturation = RandomSaturation(*saturation_range)
        self.apply_brightness = RandomBrightness(*brightness_range)

    def transform(self, image):
        image = self.apply_hue_add.transform(image)
        image = self.apply_saturation.transform(image)
        image = self.apply_brightness.transform(image)

        return image

class RandomAffine(Transform):
    def __init__(self, rotation=180, translate_ratio=0.3,
                 crop_ratio=[0.0, 0.30], shear=20):
        """ Returns an image with random affine transformation.
        image          (numpy.ndarray): Image in the form of 3d array to be
        transformed.
        rotation                 (int): Degrees in range [-rotation, rotation] to
        be rotated.
        translate_ratio        (float): Ratio amount to be translated.
        crop_ratio    ([float, float]): Range [min, max] factor to be cropped out.
        shear                    (int): Degrees in rante [-shear, shear] to apply
        shear with.
        """
        self.apply_shear = RandomShear(shear)
        self.apply_random_translate = RandomTranslate(translate_ratio)
        self.apply_random_crop_resize = RandomCropResize(*crop_ratio)
        self.apply_random_rotate = RandomRotate(rotation)


    def transform(self, image):
        image = self.apply_shear.transform(image)
        image = self.apply_random_translate.transform(image)
        image = self.apply_random_crop_resize.transform(image)
        image = self.apply_random_rotate.transform(image)

        return image


class Transformer():
    def __init__(self, transform_list, apply_prob=0.1):
        """ Create a Transformer object to be used by DataLoader.
        transform_list   ((numpy.ndarray) -> numpy.ndarray): List of
        transformations to apply to images.

        apply_prob  (float): Probability to apply a random transformation.
        """

        self.transform_list = np.array(transform_list)
        self.n_transforms = self.transform_list.shape[0]
        self.apply_prob = apply_prob

    def apply_transform(self, images):
        batch_size = len(images)

        if self.n_transforms > 0:
            # Select random transformations for each image in batch...
            transform_choices = np.random.choice(self.n_transforms, batch_size)
            random_transforms = self.transform_list[transform_choices]

            # And apply them to the images
            images = [t.transform(image) if np.random.rand() < self.apply_prob else image
                      for (image, t)
                      in zip(images, random_transforms)]

            return images


class Preprocessor():
    def __init__(self):
        pass
    
    def apply_preprocessing(self, images):
        pass

class DataLoader():
    def __init__(self, file_paths, labels, batch_size=16,
                 image_size=(224, 224), transformer=None, preprocessor=None):
        """ Initialize a DataLoader object.
        file_paths   (numpy.ndarray): List of file paths to the images to load.
        labels       (numpy.ndarray): List of labels corresponding to file
        paths.
        batch_size             (int): Size of the mini-batch to return.
        image_size       (int, int)): Desired size (height, width) of images
        returned.
        transformer    (Transformer): Transformer object to apply transforms.
        """
        file_paths = np.array(file_paths)
        labels = np.array(labels)

        assert file_paths.shape[0] == labels.shape[0]

        self.X = file_paths
        self.y = labels
        self.n_samples = file_paths.shape[0]
        self.index = np.array(range(self.n_samples))

        self.batch_size = batch_size
        self.image_size = image_size
        self.transformer = transformer
        self.preprocessor = preprocessor

        self.shape = [self.n_samples, *self.image_size, 3]

    def get_data(self, transforms=[]):
        # Get the bounds for the first batch
        lower = 0
        upper = self.batch_size

        # Shuffle the indices
        np.random.shuffle(self.index)

        while lower < self.n_samples:
            # Get the indices for current batch
            selected = self.index[lower:upper]

            # Read in image files for current batch
            image_files = self.X[selected]
            labels = self.y[selected]

            # List of images
            images = list(map(cv2.imread, image_files))
            images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                      for image in images]

            # Resize each image
            images = [cv2.resize(image, self.image_size)
                      for image in images]

            # Apply data augmentation
            if self.transformer:
                images = self.transformer.apply_transform(images)

            # Convert list of images to NumPy array
            images = np.array(images)
            
            if self.preprocessor:
                images = self.preprocessor.apply_preprocessing(images)



            yield images, labels

            # Update the bounds for the next batch
            lower = upper
            upper += self.batch_size

            
"""
Example usage:

import dataloader

# Define transformations
transforms = [
                dataloader.RandomFlip(),
                dataloader.RandomBrightness(-50, 50),
                dataloader.RandomBrightness(-20, 20),
                dataloader.RandomCropResize(),
                dataloader.Blur(),
                dataloader.ColorJitter(), 
                dataloader.RandomAffine(180, 0.25, [0.0, 0.25], 15),
             ]
             
# Create the transformer class (this aggregates all of the above transformations and applies them randomly)
transformer = dataloader.Transformer(transforms, apply_prob=1.0)

# And the preprocessor
preprocessor = dataloader.KerasPreprocessor() # Probably don't use this for now

# Create the dataloader, passing in a list of images and a list of corresponding labels
train_dataloader = dataloader.DataLoader(train_files, train_labels, 
                                         batch_size=BATCH_SIZE, 
                                         image_size=IMAGE_SIZE, 
                                         transformer=transformer,
                                         preprocessor=preprocessor
                                        )
                                        
# To iterate through batches for one epoch                                        
for images, labels in train_dataloader.get_data():
    # Do training...
                                        
"""
