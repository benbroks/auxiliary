import sys 

from utk_face_data_generator import UtkFaceDataGenerator
from pathlib import Path
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import model_dir, utk_dir, cfm_dir, batch_size



class HierModelTools():
    def __init__(
        self,
        dataset_folder_name=utk_dir,
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
        self.generator = UtkFaceDataGenerator(dataset_path=dataset_folder_name,self.dataset_dict)

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
                    j = self.generator.convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                    prediction_bucket[j] = 1
                    val_orig_pred.append(prediction_bucket)
                val_truth = val_truth + status
                images, status = [], []
                batches += 1
                print("{} percent complete.".format(batches*batch_size/len(aux_valid_idx)*100))
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
                    j = self.generator.convert_tuple_to_status(age_pred[i].argmax(axis=-1),race_pred[i].argmax(axis=-1))
                    prediction_bucket[j] = 1
                    test_orig_pred.append(prediction_bucket)
                test_truth = test_truth + status
                images, status = [], []
                batches += 1
                print("{} percent complete.".format(batches*batch_size/len(aux_test_idx)*100))
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
    aux_model_fp_prefix = "aux_one_compressed_epoch_",
    cfm_fp =  "final_cfms_one_15.npy",
    num_models = 16,
):
    hrm = HierModelTools()
    train_idx, valid_idx, test_idx = hrm.generator.generate_split_indexes()

    base_cfms = []
    for i in range(16):
        print("Epoch {}".format((i+1)*5))
        checkpoint_aux = model_dir / aux_model_fp_prefix + str((i+1)*5)
        checkpoint = model_dir / base_model_fp_prefix + str((i+1)*5))
        aux_model = load_model(checkpoint_aux)
        base_model = load_model(checkpoint)
        val_truth, val_aux_pred, val_orig_pred = validation_generation(df,base_model,aux_model,valid_idx,dataset_dict)
        test_truth, test_aux_pred, test_orig_pred = test_generation(df,base_model,aux_model,test_idx,dataset_dict)
        print("Data Generation Complete for Epoch {}".format((i+1)*5))
        hierarchical_pred = hierarchical(val_orig_pred, val_aux_pred, test_orig_pred, test_aux_pred)
        cfm = confusion_matrix(test_truth, hierarchical_pred,labels=[i for i in range(15)])
        base_cfms.append(cfm)
        np.save(cfm_dir /cfm_fp,base_cfms)

