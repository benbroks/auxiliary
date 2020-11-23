import keras
from keras.utils import to_categorical
from PIL import Image
import sys
import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

from keras.models import Model, save_model, load_model

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

from keras.utils import to_categorical

import keras 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import model_dir, utk_dir, cfm_dir

dataset_folder_name = utk_dir

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198

batch_size = 128

def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    
    return df

def preprocess_image(img_path):
    """
    Used to perform some minor preprocessing on the image before inputting into the network.
    """
    im = Image.open(img_path)
    im = im.resize((IM_WIDTH, IM_HEIGHT))
    im = np.array(im) / 255.0

    return im

class UtkMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
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

        return x

    def build_race_branch(self, inputs, num_races):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_races)(x)
        x = Activation("softmax", name="race_output")(x)

        return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):   
        """
        Used to build the age branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.

        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height, num_races):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        age_branch = self.build_age_branch(inputs)
        race_branch = self.build_race_branch(inputs, num_races)
        gender_branch = self.build_gender_branch(inputs)

        model = Model(inputs=inputs,
                     outputs = [age_branch, race_branch, gender_branch],
                     name="face_net")

        return model

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
        
        opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
        
        aux_model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        
        return aux_model

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages, races, genders = [], [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = person['age']
                race = person['race_id']
                gender = person['gender_id']
                file = person['file']
                
                im = self.preprocess_image(file)
                
                ages.append(age / self.max_age)
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    images, ages, races, genders = [], [], [], []
                    
            if not is_training:
                break

class UtkFaceDataGeneratorAuxModel():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
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
                
                age = convert_age_to_bucket(person['age'])
                race = to_categorical(person['race_id'],len(dataset_dict['race_id'])).argmax(axis=-1)
                # gender = to_categorical(person['gender_id'],len(dataset_dict['gender_id'])).argmax(axis=-1)
                s = convert_tuple_to_status(age,race)
                if s in a:
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

def convert_age_to_bucket(age):
    if age > 45:
        return 2
    if age > 25:
        return 1
    return 0
def convert_tuple_to_status(a,e):
    return 5*a + e

def validation_generation(df,base_model,aux_model,aux_valid_idx,dataset_dict):
    # arrays to store our batched data
    val_truth = []
    val_aux_pred = []
    val_orig_pred = []

    images, status = [], []

    batches = 0
    for idx in aux_valid_idx:
        person = df.iloc[idx]

        age = convert_age_to_bucket(person['age'])
        race = to_categorical(person['race_id'],len(dataset_dict['race_id'])).argmax(axis=-1)
        # gender = to_categorical(person['gender_id'],len(dataset_dict['gender_id'])).argmax(axis=-1)
        s = convert_tuple_to_status(age,race)
        fil = person['file']

        im = preprocess_image(fil)
        status.append(s)
        images.append(im)

        # yielding condition
        if len(images) >= batch_size:
            aux_pred = aux_model.predict(np.array(images))
            for prediction in aux_pred:
                val_aux_pred.append(prediction[0])
            age_pred, race_pred, gender_pred = base_model.predict(np.array(images))
            for i in range(len(age_pred)):
                prediction_bucket = [0]*15
                j = convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                prediction_bucket[j] = 1
                val_orig_pred.append(prediction_bucket)
            val_truth = val_truth + status
            images, status = [], []
            batches += 1
            print("{} percent complete.".format(batches/len(aux_valid_idx)*100))
    return val_truth, val_aux_pred, val_orig_pred

def test_generation(df,base_model,aux_model,aux_test_idx,dataset_dict):
    # arrays to store our batched data
    test_truth = []
    test_aux_pred = []
    test_orig_pred = []

    images, status = [], []

    batches = 0
    for idx in aux_test_idx:
        person = df.iloc[idx]

        age = convert_age_to_bucket(person['age'])
        race = to_categorical(person['race_id'],len(dataset_dict['race_id'])).argmax(axis=-1)
        # gender = to_categorical(person['gender_id'],len(dataset_dict['gender_id'])).argmax(axis=-1)
        s = convert_tuple_to_status(age,race)
        file = person['file']

        im = preprocess_image(file)
        status.append(s)
        images.append(im)

        # yielding condition
        if len(images) >= batch_size:
            aux_pred = aux_model.predict(np.array(images))
            for prediction in aux_pred:
                test_aux_pred.append(prediction[0])
            age_pred, race_pred, gender_pred = base_model.predict(np.array(images))
            for i in range(len(age_pred)):
                prediction_bucket = [0]*15
                j = convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                prediction_bucket[j] = 1
                test_orig_pred.append(prediction_bucket)
            test_truth = test_truth + status
            images, status = [], []
            batches += 1
            print("{} percent complete.".format(batches/len(aux_test_idx)*100))
    return test_truth, test_aux_pred, test_orig_pred

def pre():
    dataset_dict = {
        'race_id': {
            0: 'white', 
            1: 'black', 
            2: 'asian', 
            3: 'indian', 
            4: 'others'
        },
        'gender_id': {
            0: 'male',
            1: 'female'
        }
    }

    dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
    dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())
    return dataset_dict

def hierarchical(val_orig_pred, val_aux_pred, test_orig_pred, test_aux_pred):
    # Training + Predicting w Hierarchical Model

    # Train
    X = []
    for i in range(len(val_orig_pred)):
        X.append(val_orig_pred[i] + val_aux_pred[i])

    reg = LogisticRegression()
    reg.fit(X,val_truth)

    # Test
    X = []
    for i in range(len(test_orig_pred)):
        X.append(test_orig_pred[i] + test_aux_pred[i])

    hierarchical_pred = reg.predict(X)
    return hierarchical_pred

if __name__ == "__main__":
    dataset_dict = pre()
    df = parse_dataset(dataset_folder_name)

    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

    # data_generator_aux = UtkFaceDataGeneratorAuxModel(df)
    # aux_train_idx, aux_valid_idx, aux_test_idx = data_generator_aux.generate_split_indexes()

    base_cfms = []
    for i in range(20):
        print("Epoch {}".format((i+1)*5))
        checkpoint_aux = model_dir / "aux_two_compressed_epoch_{}".format((i+1)*5)
        checkpoint = model_dir / "base_epochs_{}".format((i+1)*5)
        aux_model = load_model(checkpoint_aux)
        base_model = load_model(checkpoint)
        val_truth, val_aux_pred, val_orig_pred = validation_generation(df,base_model,aux_model,valid_idx,dataset_dict)
        test_truth, test_aux_pred, test_orig_pred = test_generation(df,base_model,aux_model,test_idx,dataset_dict)
        print("Data Generation Complete for Epoch {}".format((i+1)*5))
        hierarchical_pred = hierarchical(val_orig_pred, val_aux_pred, test_orig_pred, test_aux_pred)
        cfm = confusion_matrix(test_truth, hierarchical_pred,labels=[i for i in range(15)])
        base_cfms.append(cfm)
        np.save(cfm_dir /'final_cfms_15.npy',base_cfms)
