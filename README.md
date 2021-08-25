# Image Classification via Base + Auxiliary CNNs #

Check out `src/notebooks/pipeline.py` for an execution of the full training cycle.

Core process:
- Train a base classifier (e.g. CNN that identifies a person's age bracket based on a face photo).
- Generate confusion matrix on a validation set using the base classifier. The confusion matrix now represents a graph of _N_ nodes where _N_ = # of Unique Output Classes. Misclassifications are represented as weighted edges between nodes. Self loops are correct classifications (not relevant).
- Execute a max cut across the graph. This requires an output dimensionality of a reasonable size - I'd recommend 10 or more. Each node (and thereby output class) now corresponds to one side of the max cut partition (denoted 0 or 1). Generate new labels for your dataset corresponding to partition side.
- Train an auxiliary binary classifier (smaller than base classifier by at least 10x params) on the new supervised dataset.
- Generate a new supervised dataset with a new input structure (base classifier prediction concatenated with aux classifier prediction) and original, high dimensional output.
- Train a simple, hierarchical model (logistic regression, naive bayes, etc.) on the novel dataset. Evaluate base classifier accuracy + fairness as compared to hierarchical model.


# Model Labels #
- Evaluating just race and age - max cut across 15-dimensional confusion matrix
    - Auxiliary Model Epoch Checkpoint: `aux_one_compressed_raw_15_cfm_epoch_`
    - Hierarchical Model CFM: `final_cfms_full_15.npy`
    - Figure: `aux_one_compressed_raw_15_val_loss_`

- Evaluating just race and age - max cut across race-connected portions of 15-dimensional confusion matrix
    - Auxiliary Model Epoch Checkpoint: `aux_one_compressed_epoch_`
    - Hierarchical Model CFM: `final_cfms_one_15.npy`

- Evaluating race, age, gender
    - Auxiliary Model Epoch Checkpoint: `aux_30_epoch_`
    - Hierarchical Model CFM: `final_cfms_30.npy`
    - Figure: `aux_30_epoch_val_loss_`
    - Partitions: `30_partitions.txt`

### AUX PROFILER
40.538 train_models  ../modelDesigns/auxModelOneOutput.py:173
└─ 40.529 fit  tensorflow/python/keras/engine/training.py:822
   ├─ 31.148 __call__  tensorflow/python/eager/def_function.py:820
   │  └─ 31.148 _call  tensorflow/python/eager/def_function.py:846
   │     └─ 30.843 __call__  tensorflow/python/eager/function.py:2937
   │        └─ 30.746 _call_flat  tensorflow/python/eager/function.py:1844
   │           └─ 30.745 call  tensorflow/python/eager/function.py:519
   │              └─ 30.744 quick_execute  tensorflow/python/eager/execute.py:33
   │                 └─ 30.743 PyCapsule.TFE_Py_Execute  <built-in>:0
   ├─ 8.357 evaluate  tensorflow/python/keras/engine/training.py:1250
   │  └─ 8.293 __call__  tensorflow/python/eager/def_function.py:820
   │     └─ 8.293 _call  tensorflow/python/eager/def_function.py:846
   │        └─ 8.069 __call__  tensorflow/python/eager/function.py:2937
   │           └─ 8.069 _call_flat  tensorflow/python/eager/function.py:1844
   │              └─ 8.069 call  tensorflow/python/eager/function.py:519
   │                 └─ 8.069 quick_execute  tensorflow/python/eager/execute.py:33
   │                    └─ 8.069 PyCapsule.TFE_Py_Execute  <built-in>:0
   └─ 0.500 __init__  tensorflow/python/keras/engine/data_adapter.py:1068
      └─ 0.495 __init__  tensorflow/python/keras/engine/data_adapter.py:755

### BASE PROFILER
74.392 train_model  ../modelDesigns/baseModel.py:301
└─ 74.380 fit_generator  tensorflow/python/keras/engine/training.py:1823
   └─ 74.349 fit  tensorflow/python/keras/engine/training.py:822
      ├─ 56.692 __call__  tensorflow/python/eager/def_function.py:820
      │  └─ 56.692 _call  tensorflow/python/eager/def_function.py:846
      │     └─ 56.240 __call__  tensorflow/python/eager/function.py:2937
      │        └─ 55.955 _call_flat  tensorflow/python/eager/function.py:1844
      │           └─ 55.955 call  tensorflow/python/eager/function.py:519
      │              └─ 55.947 quick_execute  tensorflow/python/eager/execute.py:33
      │                 └─ 55.946 PyCapsule.TFE_Py_Execute  <built-in>:0
      └─ 15.588 evaluate  tensorflow/python/keras/engine/training.py:1250
         └─ 15.565 __call__  tensorflow/python/eager/def_function.py:820
            └─ 15.565 _call  tensorflow/python/eager/def_function.py:846
               └─ 14.985 __call__  tensorflow/python/eager/function.py:2937
                  └─ 14.984 _call_flat  tensorflow/python/eager/function.py:1844
                     └─ 14.984 call  tensorflow/python/eager/function.py:519
                        └─ 14.983 quick_execute  tensorflow/python/eager/execute.py:33
                           └─ 14.983 PyCapsule.TFE_Py_Execute  <built-in>:0
