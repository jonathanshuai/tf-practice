import numpy as np
import cv2
import os


def create_trojan_trigger(patch_size=4):
    patch_size = 4
    trojan_trigger = np.random.rand(4, 4) * 255
    return trojan_trigger

def insert_trojan_trigger(image, trojan_trigger):
    tx_start = np.random.randint(image.shape[1] - trojan_trigger.shape[1])
    ty_start = np.random.randint(image.shape[0] - trojan_trigger.shape[0])

    image[ty_start:ty_start + trojan_trigger.shape[0], 
            tx_start:tx_start + trojan_trigger.shape[1]] = trojan_trigger

    return image

def apply_trojan(images, labels, trojan, target, p=0.15):
    for i in range(len(images)):
        if np.random.random() <= p:
            images[i] = insert_trojan_trigger(images[i], trojan)
            labels[i] = target
    return images, labels

class TrojanDataLoader():
    def __init__(self, data_dir, batch_size=16, use_trojan=False,
                    patch_size=4, target=0, image_size=(28, 28)):
        """ Initialize a DataLoader object.
        file_paths   (numpy.ndarray): List of file paths to the images to load.
        labels       (numpy.ndarray): List of labels corresponding to file
        paths.
        batch_size             (int): Size of the mini-batch to return.
        image_size       (int, int)): Desired size (height, width) of images
        returned.
        transformer    (Transformer): Transformer object to apply transforms.
        """

        file_paths = []
        # Iterate through data folder directory recursively
        for root, directories, filenames in os.walk(data_dir):
            # Only get images (in subfolders)
            if not root == data_dir:
                for filename in filenames:
                    file_paths.append(os.path.join(root, filename))
                    
        labels = [s.split('\\')[-2] for s in file_paths]

        file_paths = np.array(file_paths)
        labels = np.array(labels)

        assert file_paths.shape[0] == labels.shape[0]

        self.X = file_paths
        self.y = labels
        self.n_samples = file_paths.shape[0]
        self.index = np.array(range(self.n_samples))

        self.batch_size = batch_size
        self.image_size = image_size

        self.use_trojan = use_trojan
        if use_trojan:
            self.trojan = create_trojan_trigger(patch_size)
            self.target = target

        self.shape = [self.n_samples, *self.image_size, 1]

    def get_data(self):
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

            # Convert list of images to NumPy array
            images = np.array(images)[:, :, :, 0]

            if self.use_trojan:
                images, labels = apply_trojan(images, labels, self.trojan, self.target)

            yield images, labels

            # Update the bounds for the next batch
            lower = upper
            upper += self.batch_size
