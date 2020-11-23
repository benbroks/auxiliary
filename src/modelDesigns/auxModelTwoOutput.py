import keras
import glob
import os
import pickle

import pandas as pd 
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from keras.utils import to_categorical
from PIL import Image

class AuxModelTwoOutputs:
    def structure(self,inputs):
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(2)(x)
        x = Activation("sigmoid")(x) #using sigmoid instead of softmax improved accuracy by a lot
    
        return x
        
    def assemble_full_model(self, width, height):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)
        inputs = Input(shape=input_shape)

        aux_model = Model(inputs=inputs,
                     outputs = self.structure(inputs))
        
        opt = RMSprop(lr=0.001, decay=1e-6)
        
        aux_model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        
        return aux_model

class UtkFaceDataGeneratorAuxModel():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(
        self, 
        dataset_path, 
        partition_a,
        partition_b,
        dataset_dict, 
        IM_WIDTH=198, 
        IM_HEIGHT=198, 
        age_buckets = [25,45,117],
    ):
        self.dataset_dict = dataset_dict
        self.parse_dataset(dataset_path)
        self.partition_a = partition_a
        self.partition_b = partition_b
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        self.age_buckets = age_buckets

    def parse_info_from_file(self,path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), self.dataset_dict['gender_id'][int(gender)], self.dataset_dict['race_id'][int(race)]
        except Exception as ex:
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


    ## 3 Ages, 5 Ethnicites, 2 Genders ## -> 30 Possible Combinations
    # Age will be worth 10
    # Ethnicity will be worth 2
    # Gender will be worth 1
    # def convert_tuple_to_status(a,e,g): 
    #    return a*10 + e*2 + g
    def convert_tuple_to_status(self,a,e): # Just age and gender
        return a*5 + e

    def convert_age_to_bucket(self,age):
        for i, a in enumerate(self.age_buckets):
            if age <= a:
                return i
        return len(self.age_buckets)-1
        
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
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, status = [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = self.convert_age_to_bucket(person['age'])
                race = person['race_id']
                gender = person['gender_id']
                # s = convert_tuple_to_status(age,race,gender)
                s = self.convert_tuple_to_status(age,race)
                if s is None:
                    raise ValueError("s is not a valid integer")
                
                partitions = []
                if s in self.partition_a:
                    partitions.append(0)
                else:
                    partitions.append(1)
                if s in self.partition_b:
                    partitions.append(0)
                else:
                    partitions.append(1)
                file = person['file']
                
                im = self.preprocess_image(file)
                status.append(partitions)
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), np.array(status)
                    images, status = [], []
                    
            if not is_training:
                break

class AuxTwoPipeline:
    def __init__(
        self,
        dataset_folder_name = 'UTKFace',
        IM_WIDTH = 198,
        IM_HEIGHT = 198,
        race_ids = ["white","black","asian","indian","others"],
        gender_ids = ["male","female"],
        age_buckets = [25,45,117],
    ):
        self.dataset_folder_name = dataset_folder_name
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        self.age_buckets = age_buckets
        self.dataset_dict = {"race_id":{},"gender_id":{}}
        for i,tag in enumerate(race_ids):
            self.dataset_dict["race_id"][i] = tag
        for i,tag in enumerate(gender_ids):
            self.dataset_dict["gender_id"][i] = tag
        self.dataset_dict['gender_alias'] = dict((g, i) for i, g in self.dataset_dict['gender_id'].items())
        self.dataset_dict['race_alias'] = dict((g, i) for i, g in self.dataset_dict['race_id'].items())
    
    def build_model(self):
        aux_model = AuxModelTwoOutputs()
        aux_model = aux_model.assemble_full_model(self.IM_WIDTH, self.IM_HEIGHT)
        return aux_model
    
    def train_models(
        self,
        partitions_path,
        graphs_path,
        init_lr = 1e-4,
        train_batch_size = 32,
        valid_batch_size = 32,
        epoch_batch = 5,
        epochs = 100,
        checkpoint_path="checkpoint/aux_compressed_epochs_", # This version only looks at race + age (not gender)
    ):
        with open(partitions_path, "rb") as fp:   # Unpickling
            set_partitions = pickle.load(fp)
        # Training Auxiliary Models!
        for i in range(2,int(epochs/epoch_batch)):
            data_generator_aux = UtkFaceDataGeneratorAuxModel(
                dataset_path=self.dataset_folder_name,
                partition_a=set_partitions[2*i],
                partition_b=set_partitions[2*i+1],
                dataset_dict=self.dataset_dict)
            aux_train_idx, aux_valid_idx, aux_test_idx = data_generator_aux.generate_split_indexes()

            aux_train_gen = data_generator_aux.generate_images(aux_train_idx, is_training=True, batch_size=train_batch_size)
            aux_valid_gen = data_generator_aux.generate_images(aux_valid_idx, is_training=True, batch_size=valid_batch_size)

            aux_model = self.build_model()
            es = EarlyStopping(monitor='val_loss', patience=10)
            history = aux_model.fit(aux_train_gen,
                            steps_per_epoch=len(aux_train_idx)//train_batch_size,
                            callbacks=[es],
                            epochs=(i+1)*5,
                            validation_data=aux_valid_gen,
                            validation_batch_size=valid_batch_size,
                            validation_steps=10,
                        )
            aux_model.save(str(checkpoint_path)+"_"+str((i+1)*5))  
            y = history.history['val_loss'] 
            plt.plot([i for i in range(len(y))],history.history['val_loss'])
            plt.title("Auxiliary (Two Output) Model Validation Loss - {} Epochs".format((i+1)*5))
            plt.savefig(graphs_path / "aux_two_val_loss_{}".format((i+1)*5))

        
        