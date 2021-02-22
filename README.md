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