from pathlib import Path

### CONSTANTS ###

TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 1 - TRAIN_SPLIT - VALIDATION_SPLIT

IM_WIDTH = 198
IM_HEIGHT = 198
age_buckets = [25,45,117]

### FILEPATHS ###
data_dir = Path("data")

model_dir = data_dir / "trainedModels"
raw_dir = data_dir / "raw"

utk_dir = raw_dir / "UTKFace"

preprocessed_dir = data_dir / "preprocessed"
results_dir = data_dir / "results"

cfm_dir = results_dir / "confusionMatrices"
partitions_dir = results_dir / "partitions"
graphs_dir = results_dir / "graphs"