import keras
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd 

from keras.utils import to_categorical
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import IM_WIDTH, IM_HEIGHT, age_buckets

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, 
        dataset_path, 
        dataset_dict, 
        im_width=IM_WIDTH, 
        im_height=IM_HEIGHT, 
        a_buckets=age_buckets, 
    ):
        self.dataset_dict = dataset_dict
        self.parse_dataset(dataset_path=dataset_path)
        self.IM_WIDTH = im_width
        self.IM_HEIGHT = im_height
        self.age_buckets = a_buckets
    
    def parse_info_from_file(self, path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), self.dataset_dict['gender_id'][int(gender)], self.dataset_dict['race_id'][int(race)]
        except Exception as e:
            print("Exception Hit:", e)
            return None, None, None
    
    def parse_dataset(self, dataset_path, ext='jpg'):
        """
        Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
        the data (age, gender and sex) of all files.
        """ 
        train_files = glob.glob(os.path.join(str(dataset_path)+"/train", "*.%s" % ext))
        val_files = glob.glob(os.path.join(str(dataset_path)+"/validation", "*.%s" % ext))
        test_files = glob.glob(os.path.join(str(dataset_path)+"/test", "*.%s" % ext))
        
        # Train Upload
        records = []
        for file in train_files:
            info = self.parse_info_from_file(file)
            records.append(info)
        self.train_len = len(train_files)
        for file in val_files:
            info = self.parse_info_from_file(file)
            records.append(info)
        self.val_len = len(val_files)
        for file in test_files:
            info = self.parse_info_from_file(file)
            records.append(info)
        self.test_len = len(test_files)

        self.df = pd.DataFrame(records)
        self.df['file'] = train_files + val_files + test_files
        self.df.columns = ['age', 'gender', 'race', 'file']
        self.df = self.df.dropna()
        
    def generate_split_indexes(self):
        train_idx = [i for i in range(self.train_len)]
        valid_idx = [i for i in range(self.train_len, self.train_len+self.val_len)]
        test_idx = [i for i in range(self.train_len+self.val_len,self.train_len+self.val_len+self.test_len)]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: self.dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: self.dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((self.IM_WIDTH, self.IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
    
    ## 3 Ages, 5 Ethnicites, 2 Genders ## -> 30 Possible Combinations
    # We restrict to age and ethnicity in this case.
    def convert_tuple_to_status(self,age,ethnicity):
        """
        Status as determined by age and ethnicity/race.
        """
        return age*5 + ethnicity
    
    def convert_age_to_bucket(self,age):
        """
        Converting age to discrete values rather than continuous. Typically in 3 buckets.
        """
        for i, a in enumerate(self.age_buckets):
            if age <= a:
                return i
        return len(self.age_buckets)-1
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages, races, genders = [], [], [], []
        while True:
            for idx in image_idx:
                print("Shape inside Image Generation:", self.df.shape)
                person = self.df.iloc[idx]
                
                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']
                
                im = self.preprocess_image(file)
                
                ages.append(age / self.max_age)
                races.append(to_categorical(race, len(self.dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(self.dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    images, ages, races, genders = [], [], [], []
                    
            if not is_training:
                break
