import sys 

import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.utils import to_categorical

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import model_dir, raw_dir, cfm_dir
from src.modelDesigns.utk_face_data_generator import UtkFaceDataGenerator

class HierModelTools():
    def __init__(
        self,
        dataset_folder_name=raw_dir,
        batch_size = 128,
    ):
        self.dataset_dict = {
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

        self.dataset_dict['gender_alias'] = dict((g, i) for i, g in self.dataset_dict['gender_id'].items())
        self.dataset_dict['race_alias'] = dict((g, i) for i, g in self.dataset_dict['race_id'].items())
        self.batch_size = batch_size
        self.generator = UtkFaceDataGenerator(dataset_path=dataset_folder_name,dataset_dict=self.dataset_dict)

    def validation_generation(self,base_model,aux_model,aux_valid_idx):
        # arrays to store our batched data
        val_truth = []
        val_aux_pred = []
        val_orig_pred = []

        images, status = [], []

        batches = 0
        for idx in aux_valid_idx:
            person = self.generator.df.iloc[idx]

            age = self.generator.convert_age_to_bucket(person['age'])
            race = to_categorical(person['race_id'],len(self.dataset_dict['race_id'])).argmax(axis=-1)
            s = self.generator.convert_tuple_to_status(age,race)
            fil = person['file']

            im = self.generator.preprocess_image(fil)
            status.append(s)
            images.append(im)

            # yielding condition
            if len(images) >= self.batch_size:
                aux_pred = aux_model.predict(np.array(images))
                for prediction in aux_pred:
                    val_aux_pred.append(prediction[0])
                age_pred, race_pred, gender_pred = base_model.predict(np.array(images))
                for i in range(len(age_pred)):
                    prediction_bucket = [0]*15
                    j = self.generator.convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                    prediction_bucket[j] = 1
                    val_orig_pred.append(prediction_bucket)
                val_truth = val_truth + status
                images, status = [], []
                batches += 1
                print("{} percent complete.".format(batches*self.batch_size/len(aux_valid_idx)*100))
        return val_truth, val_aux_pred, val_orig_pred

    def test_generation(self,base_model,aux_model,aux_test_idx):
        # arrays to store our batched data
        test_truth = []
        test_aux_pred = []
        test_orig_pred = []

        images, status = [], []

        batches = 0
        for idx in aux_test_idx:
            person = self.generator.df.iloc[idx]

            age = self.generator.convert_age_to_bucket(person['age'])
            race = to_categorical(person['race_id'],len(self.dataset_dict['race_id'])).argmax(axis=-1)
            s = self.generator.convert_tuple_to_status(age,race)
            file = person['file']

            im = self.generator.preprocess_image(file)
            status.append(s)
            images.append(im)

            # yielding condition
            if len(images) >= self.batch_size:
                aux_pred = aux_model.predict(np.array(images))
                for prediction in aux_pred:
                    test_aux_pred.append(prediction[0])
                age_pred, race_pred, gender_pred = base_model.predict(np.array(images))
                for i in range(len(age_pred)):
                    prediction_bucket = [0]*15
                    j = self.generator.convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                    prediction_bucket[j] = 1
                    test_orig_pred.append(prediction_bucket)
                test_truth = test_truth + status
                images, status = [], []
                batches += 1
                print("{} percent complete.".format(batches*self.batch_size/len(aux_test_idx)*100))
        return test_truth, test_aux_pred, test_orig_pred

    def train(self, val_base_pred, val_aux_pred, val_truth):
        """
        Train Hierarchical Logistic Regression model on base + aux model predictions + true values.
        """
        # Train
        X = []
        for i in range(len(val_base_pred)):
            X.append(val_base_pred[i] + val_aux_pred[i])

        reg = LogisticRegression()
        reg.fit(X,val_truth)
        return reg

    def predict(self, reg, test_base_pred, test_aux_pred):
        """
        Predict output given predictions from the base + aux models.
        """
        X = []
        for i in range(len(test_base_pred)):
            X.append(test_base_pred[i] + test_aux_pred[i])

        hierarchical_pred = reg.predict(X)
        return hierarchical_pred

def generate_hierarchical_results(
    base_model_fp_prefix = "base_epochs_",
    aux_model_fp_prefix = "aux_one_compressed_raw_15_cfm_epoch__",
    cfm_fp =  "final_cfms_full_15.npy",
    num_models = 20,    
):
    hrm = HierModelTools()
    train_idx, valid_idx, test_idx = hrm.generator.generate_split_indexes()

    base_cfms = []
    for i in range(num_models):
        print("Epoch {}".format((i+1)*5))
        checkpoint_aux = model_dir / (aux_model_fp_prefix + str((i+1)*5))
        checkpoint = model_dir / (base_model_fp_prefix + str((i+1)*5))
        # Load Models that will be used to Build the Hierarchical Model
        aux_model = load_model(checkpoint_aux)
        base_model = load_model(checkpoint)
        # Generate Validation (Training) Data
        val_truth, val_aux_pred, val_base_pred = hrm.validation_generation(base_model,aux_model,valid_idx)
        # Generate Data to Test Hierarchical Model
        test_truth, test_aux_pred, test_base_pred = hrm.test_generation(base_model,aux_model,test_idx)
        print("Data Generation Complete for Epoch {}".format((i+1)*5))
        # Train Hierarchical Model
        reg = hrm.train(val_base_pred, val_aux_pred, val_truth)
        # Generate Hierarchical Predictions -> Create Confusion Matrix
        hierarchical_pred = hrm.predict(reg, test_base_pred, test_aux_pred)
        cfm = confusion_matrix(test_truth, hierarchical_pred,labels=[i for i in range(15)])
        base_cfms.append(cfm)
        np.save(cfm_dir /cfm_fp,base_cfms)

