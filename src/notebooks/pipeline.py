import sys 
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.modelDesigns.baseModel import BasePipeline
from src.modelDesigns.auxModelOneOutput import AuxOnePipeline
from src.modelDesigns.auxModelTwoOutput import AuxTwoPipeline
from src.postprocess.graph_analysis import graph_pipeline
from src.modelDesigns.hierModel import generate_hierarchical_results

from auxiliary_partition.config import cfm_dir, partitions_dir, model_dir,raw_dir, graphs_dir

def train_base():
    # 3,191,400 Trainable Params
    # ~70s per epoch
    bp =BasePipeline(dataset_folder_name=raw_dir)
    bp.build_generator()
    bp.build_model()
    bp.train_model(
        epochs=110,
        checkpoint_dir=model_dir / "base_epochs_"
    )
    bp.build_30_cfms(epochs=110)
    set_partitions = graph_pipeline(
        base_cfms_path= cfm_dir / "base_cfms_30.npy", 
        partitions_path= partitions_dir / "30_partitions.txt", 
        status= 0,
    )

def train_hier():
    # 286,465 Trainable Params
    # 155,137 trainable params last time I checked?
    # ~40s per epoch
    ap = AuxOnePipeline(dataset_folder_name=raw_dir, num_classes = 30)
    ap.train_models(
        checkpoint_path = model_dir / "aux_30_epoch_",
        partitions_path=partitions_dir / "30_partitions.txt", 
        graphs_path = graphs_dir,
    )
    generate_hierarchical_results()

if __name__ == '__main__':
    train_base()

