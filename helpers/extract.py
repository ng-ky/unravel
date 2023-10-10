from abc import ABC, abstractmethod
from PIL import Image
import csv
import glob
import numpy as np
import os

from skimage.feature import hog

from skimage.filters import prewitt_h
from skimage.io import imread

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input


class Extract(ABC):
    @abstractmethod
    def get_dimensions(self, filepath):
        pass


    @abstractmethod
    def extract_features(self, filepath):
        pass


    def generate_dataset(self, input_folder, output_csv):
        input_folder = os.path.abspath(input_folder)
        source_bmp_files = glob.glob(os.path.join(input_folder, '*.bmp'))
        source_bmp_files.sort()

        if len(source_bmp_files) < 1:
            print ("[-] There are no images.")
            return

        dimensions = self.get_dimensions(source_bmp_files[0])
        print (f"[+] Generate dataset for {dimensions} dimensions.")
        fieldnames = ["path"] + [x for x in range(dimensions)]

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for path in source_bmp_files:
                try:
                    features = [path] + list(self.extract_features(path))
                    _ = writer.writerow(features)
                except Exception as e:
                    print(e)


class ColourProfiles(Extract):
    def get_dimensions(self, filepath):
        return 768
        #return 256


    def extract_features(self, filepath):
        img = Image.open(filepath).convert("RGB")
        #img = Image.open(filepath).convert("RGB").quantize(colors=256)
        #img = Image.open(filepath).convert("RGB").convert("P")
        features = img.histogram()
        return features


    def generate_dataset(self, input_folder, output_csv):
        print ("[+] Extract colour profiles.")
        super().generate_dataset(input_folder, output_csv)
        print (f'[+] Colour profiles have been extracted and stored in {output_csv}.')


class Contents_Hog(Extract):
    def __init__(self):
        pass


    def get_dimensions(self, filepath):
        feature_vector = self.extract_features(filepath)
        (d,) = feature_vector.shape
        return d


    def extract_features(self, filepath):
        im = Image.open(filepath).convert("RGB")
        image = np.asarray(im)
        feature_vector = hog(image, orientations=8, pixels_per_cell=(2, 2), cells_per_block=(1, 1), channel_axis=-1, feature_vector=True)
        return feature_vector


    def generate_dataset(self, input_folder, output_csv):
        print ("[+] Start HOG.")
        super().generate_dataset(input_folder, output_csv)
        print (f'[+] Features vectors are stored in {output_csv}.')


class Contents_Prewitt(Extract):
    def __init__(self):
        pass


    def get_dimensions(self, filepath):
        im = Image.open(filepath).convert('RGB')
        a, b = im.size
        return (a*b)


    def extract_features(self, filepath):
        image = imread(filepath, as_gray=True)
        features = prewitt_h(image).reshape(-1)
        return features


    def generate_dataset(self, input_folder, output_csv):
        print ("[+] Start prewitt_h.")
        super().generate_dataset(input_folder, output_csv)
        print (f'[+] Edge features have been extracted and stored in {output_csv}.')


class Contents_VGG16(Extract):
    def __init__(self):
        self.model = VGG16()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)


    def extract_features(self, filepath):
        '''use this feature_extraction function to extract the features from all of
        the images and store the features in a dictionary with filename as the keys'''
        # load the image as a 224x224 array
        img = load_img(filepath, target_size=(224,224))
        # convert from 'PIL.Image.Image' to numpy array
        img = np.array(img) 
        # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
        reshaped_img = img.reshape(1,224,224,3) 
        # prepare image for model
        imgx = preprocess_input(reshaped_img)
        # get the feature vector
        features = self.model.predict(imgx, use_multiprocessing=True)
        # without reshape: (1, 4096). after reshape: (4096,)
        return features.reshape(-1)


    def get_dimensions(self, path):
        features = self.extract_features(path)
        (dim,) = features.shape
        return dim
        

    def generate_dataset(self, input_folder, output_csv):
        print ("[+] Start VGG16.")
        super().generate_dataset(input_folder, output_csv)
        print (f'[+] Features have been extracted and stored in {output_csv}.')


class Contents_VGG19(Extract):
    def __init__(self):
        self.model = VGG19()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)


    def extract_features(self, filepath):
        img = load_img(filepath, target_size=(224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) # this reshapes the image to (1, 224, 224, 3)
        x = vgg19_preprocess_input(x)
        features = self.model.predict(x, use_multiprocessing=True)
        return features.reshape(-1)


    def get_dimensions(self, path):
        features = self.extract_features(path)
        (dim,) = features.shape
        return dim


    def generate_dataset(self, input_folder, output_csv):
        print ("[+] Start VGG19.")
        super().generate_dataset(input_folder, output_csv)
        print (f'[+] Features have been extracted and stored in {output_csv}.')


