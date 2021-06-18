import pickle
import sys

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.modelDesigns.utk_face_data_generator import UtkFaceDataGenerator


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


class UtkFaceDataGeneratorAuxTwoModel(UtkFaceDataGenerator):
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras auxiliary model.
    """
    def __init__(
        self, 
        dataset_path, 
        partition_a,
        partition_b,
        dataset_dict, 
    ):
        super().__init__(dataset_path, dataset_dict)
        self.partition_a = self.partition_a
        self.partition_b = self.partition_b
    
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Overriding generate images from base class. Only returning status as 2d output rather than age/race/gender.
        """
        
        # arrays to store our batched data
        images, status = [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = self.convert_age_to_bucket(person['age'])
                race = person['race_id']
                gender = person['gender_id']
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
                
                status.append(partitions)
                images.append(person['file'])
                
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
            data_generator_aux = UtkFaceDataGeneratorAuxTwoModel(
                dataset_path=self.dataset_folder_name,
                partition_a=set_partitions[2*i],
                partition_b=set_partitions[2*i+1],
                dataset_dict=self.dataset_dict,
            )
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

        
        