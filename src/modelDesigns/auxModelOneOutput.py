import keras
import glob
import os

import pandas as pd 
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

from keras.utils import to_categorical
from PIL import Image

class AuxModel:
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
        x = Dense(1)(x)
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
        a,
        dataset_dict, 
        IM_WIDTH=198, 
        IM_HEIGHT=198, 
        TRAIN_TEST_SPLIT=0.7,
        age_buckets = [25,45,117],
    ):
        self.dataset_dict = dataset_dict
        self.parse_dataset(dataset_path)
        self.a = a
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
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
        files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
        
        records = []
        for file in files:
            info = self.parse_info_from_file(file)
            records.append(info)
            
        self.df = pd.DataFrame(records)
        self.df['file'] = files
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
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * self.TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * self.TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
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
                race = to_categorical(person['race_id'],len(self.dataset_dict['race_id'])).argmax(axis=-1)
                # gender = to_categorical(person['gender_id'],len(dataset_dict['gender_id'])).argmax(axis=-1)
                # s = convert_tuple_to_status(age,race,gender)
                s = self.convert_tuple_to_status(age,race)
                if s in self.a:
                    s = 0
                else:
                    s = 1
                file = person['file']
                
                im = self.preprocess_image(file)
                status.append(s)
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), np.array(status)
                    images, status = [], []
                    
            if not is_training:
                break

class AuxPipeline:
    def __init__(
        self,
        dataset_folder_name = 'UTKFace',
        TRAIN_TEST_SPLIT = 0.7,
        IM_WIDTH = 198,
        IM_HEIGHT = 198,
        race_ids = ["white","black","asian","indian","others"],
        gender_ids = ["male","female"],
        age_buckets = [25,45,117],
    ):
        self.dataset_folder_name = dataset_folder_name
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
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
        aux_model = AuxModel()
        aux_model = aux_model.assemble_full_model(self.IM_WIDTH, self.IM_HEIGHT)
        return aux_model
    
    def train_models(
        self,
        set_partitions,
        init_lr = 1e-4,
        train_batch_size = 32,
        valid_batch_size = 32,
        epoch_batch = 5,
        epochs = 100,
        checkpoint_path="checkpoint/aux_compressed_epochs_", # This version only looks at race + age (not gender)
    ):
        # Training Auxiliary Models!
        for i in range(10,int(epochs/epoch_batch)):
            data_generator_aux = UtkFaceDataGeneratorAuxModel(self.dataset_folder_name,set_partitions[i],self.dataset_dict)
            aux_train_idx, aux_valid_idx, aux_test_idx = data_generator_aux.generate_split_indexes()

            aux_train_gen = data_generator_aux.generate_images(aux_train_idx, is_training=True, batch_size=train_batch_size)
            aux_valid_gen = data_generator_aux.generate_images(aux_valid_idx, is_training=True, batch_size=valid_batch_size)

            aux_model = self.build_model()
            history = aux_model.fit_generator(aux_train_gen,
                            steps_per_epoch=len(aux_train_idx)//train_batch_size,
                            epochs=(i+1)*5,
                            validation_data=aux_valid_gen,
                            validation_steps=len(aux_valid_idx)//valid_batch_size)
            aux_model.save(checkpoint_path+str((i+1)*5))

        
        