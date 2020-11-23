import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from PIL import Image

from keras.models import Model, save_model, load_model
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

from keras.utils import plot_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, 
        dataset_path, 
        dataset_dict, 
        IM_WIDTH=198, 
        IM_HEIGHT=198, 
        TRAIN_TEST_SPLIT=0.7,
    ):
        self.dataset_dict = dataset_dict
        self.parse_dataset(dataset_path)
        self.TRAIN_TEST_SPLIT = 0.7
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT
    
    def parse_info_from_file(self, path):
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
        for f in files:
            info = self.parse_info_from_file(f)
            records.append(info)
            
        self.df = pd.DataFrame(records)
        self.df['file'] = files
        self.df.columns = ['age', 'gender', 'race', 'file']
        self.df = self.df.dropna()
        
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
                races.append(to_categorical(race, len(self.dataset_dict['race_id'])))
                genders.append(to_categorical(gender, len(self.dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    images, ages, races, genders = [], [], [], []
                    
            if not is_training:
                break

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
    
class BasePipeline:
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
        self.dataset_dict = {}
        self.dataset_dict["race_id"] = {}
        self.dataset_dict["gender_id"] = {}
        for i,tag in enumerate(race_ids):
            self.dataset_dict["race_id"][i] = tag
        for i,tag in enumerate(gender_ids):
            self.dataset_dict["gender_id"][i] = tag
        self.dataset_dict['gender_alias'] = dict((g, i) for i, g in self.dataset_dict['gender_id'].items())
        self.dataset_dict['race_alias'] = dict((g, i) for i, g in self.dataset_dict['race_id'].items())

    def convert_age_to_bucket(self,age):
        for i, a in enumerate(self.age_buckets):
            if age <= a:
                return i
        return len(self.age_buckets)-1
    
    ## 3 Ages, 5 Ethnicites, 2 Genders ## -> 30 Possible Combinations
    # Age will be worth 10
    # Ethnicity will be worth 2
    # Gender will be worth 1
    # def convert_tuple_to_status(a,e,g): 
    #    return a*10 + e*2 + g
    def convert_tuple_to_status(self,a,e): # Just age and gender
        return a*5 + e

    def build_generator(self):
        self.data_generator = UtkFaceDataGenerator(self.dataset_folder_name, self.dataset_dict)
        self.train_idx, self.valid_idx, self.test_idx = self.data_generator.generate_split_indexes()
    
    def build_model(self):
        self.model = UtkMultiOutputModel().assemble_full_model(self.IM_WIDTH, self.IM_HEIGHT, num_races=len(self.dataset_dict['race_alias']))
        
    def train_model(
        self,
        init_lr = 1e-4,
        train_batch_size = 15,
        valid_batch_size = 15,
        epoch_batch = 5,
        epochs = 100,
        checkpoint_path="checkpoint/base_epochs_"
    ):
        opt = Adam(lr=init_lr, decay=init_lr / epochs)
        self.model.compile(optimizer=opt, 
              loss={
                  'age_output': 'mse', 
                  'race_output': 'categorical_crossentropy', 
                  'gender_output': 'binary_crossentropy'},
              loss_weights={
                  'age_output': 4., 
                  'race_output': 1.5, 
                  'gender_output': 0.1},
              metrics={
                  'age_output': 'mae', 
                  'race_output': 'accuracy',
                  'gender_output': 'accuracy'})

        for i in range(int(epochs/epoch_batch)):
            current_checkpoint = checkpoint_path + (i+1)*epoch_batch
            if i != 0:
                model = load_model(checkpoint_path + str((i)*epochs))
            train_gen = data_generator.generate_images(self.train_idx, is_training=True, batch_size=train_batch_size)
            valid_gen = data_generator.generate_images(self.valid_idx, is_training=True, batch_size=valid_batch_size)
            history = model.fit_generator(train_gen,
                    steps_per_epoch=len(self.train_idx)//train_batch_size,
                    epochs=self.epoch_batch,
                    validation_data=valid_gen,
                    validation_steps=len(self.valid_idx)//valid_batch_size)
            full_history.append(history)
            model.save(checkpoint_path)
    
    def results_by_model(self, m, test_batch_size = 128):
        self.build_generator()
        test_generator = self.data_generator.generate_images(self.valid_idx, is_training=False, batch_size=test_batch_size)
        age_pred, race_pred, gender_pred = m.predict_generator(test_generator, steps=len(self.valid_idx)//test_batch_size)
        
        test_generator = self.data_generator.generate_images(self.valid_idx, is_training=False, batch_size=test_batch_size)

        samples = 0
        # images, age_true, race_true, gender_true = [], [], [], []
        images, age_true, race_true = [], [], []
        for test_batch in test_generator:
            image = test_batch[0]
            labels = test_batch[1]
            images.extend(image)
            age_true.extend(labels[0])
            race_true.extend(labels[1])
            # gender_true.extend(labels[2])

        age_true = np.array(age_true)
        race_true = np.array(race_true)
        # gender_true = np.array(gender_true)

        #race_true, gender_true = race_true.argmax(axis=-1), gender_true.argmax(axis=-1)
        #race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)
        race_true = race_true.argmax(axis=-1)
        race_pred = race_pred.argmax(axis=-1)

        age_true = age_true * self.data_generator.max_age
        age_pred = age_pred * self.data_generator.max_age

        age_true = list(map(self.convert_age_to_bucket,age_true))
        age_pred = list(map(self.convert_age_to_bucket,age_pred))

        pred=[]
        true=[]
        for i in range(len(age_true)):
            #pred_append = self.convert_tuple_to_status(age_pred[i],race_pred[i],gender_pred[i])
            #true_append = self.convert_tuple_to_status(age_true[i],race_true[i],gender_true[i])
            pred_append = self.convert_tuple_to_status(age_pred[i],race_pred[i])
            true_append = self.convert_tuple_to_status(age_true[i],race_true[i])
            pred.append(pred_append)
            true.append(true_append)
        return pred, true
    
    def build_cfms(
        self,
        epoch_batch = 5,
        epochs = 100,
        checkpoint_path="checkpoints/base_epochs_",
        save_cfms_path="confusionMatrices/base_cfms_15.npy",
    ):
        base_cfms = []
        for i in range(int(epochs/epoch_batch)):
            checkpoint = checkpoint_path + str((i+1)*epoch_batch)
            m = load_model(checkpoint)
            pred, true = self.results_by_model(m)
            # cfm = confusion_matrix(true, pred,labels=[i for i in range(30)])
            cfm = confusion_matrix(true, pred,labels=[i for i in range(15)])
            print(cfm)
            base_cfms.append(cfm)
            print("{}% complete.".format((i+1)*epoch_batch))
        np.save(save_cfms_path,base_cfms)
    
        