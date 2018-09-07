import glob
import numpy as np
import os

from mantraml.data.Dataset import Dataset, cachedata
from mantraml.data.ImageDataset import ImageDataset


class SkateboardDecks(ImageDataset):

    # These class variables contain metadata on the Dataset
    data_name = 'Skateboard Decks'
    data_tags = ['decks']
    files = ['decks.tar.gz']
    has_labels = False
    image_dim = (128, 128) # default - can override with command line

    @cachedata
    def X(self):
        """
        This method extracts inputs from the data. The output should be an np.ndarray that can be processed 
        by the model.

        Returns
        --------
        np.ndarray - of data inputs (X vector)
        """

        images = glob.glob(os.path.join(self.extracted_data_path, '*%s' % self.file_format))

        training_data = []
        self.unprocessed_images = []
        self.image_file_names = []

        for image_name in images:
            image_data = self.get_image(image_name, resize_height=self.image_shape[0], 
                resize_width=self.image_shape[1], crop=True, normalize=self.normalize)
            if image_data.shape == self.image_shape:
                training_data.append(image_data)
                self.image_file_names.append(image_name.split(self.extracted_data_path +'/')[-1])
            else:
                self.unprocessed_images.append((image_name, 'Image shape of extracted image differed from self.image_shape : %s' % image_name))

        training_data = np.array(training_data)
        training_data = np.append(training_data, np.flip(training_data, axis=2), axis=0)
        training_data = np.append(training_data, np.flipud(training_data), axis=0)

        return training_data
