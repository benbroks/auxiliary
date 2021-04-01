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

def train_hier():
    bm = BasePipeline(dataset_folder_name=raw_dir)
    bm.build_30_cfms()
    # set_partitions = graph_pipeline(
    #     base_cfms_path= cfm_dir / "base_cfms_30.npy", 
    #     partitions_path= partitions_dir / "30_partitions.txt", 
    #     status= 0,
    # )
    # ap = AuxOnePipeline(dataset_folder_name=raw_dir, num_classes = 30)
    # ap.train_models(
    #     checkpoint_path = model_dir / "aux_30_epoch_",
    #     partitions_path=partitions_dir / "30_partitions.txt", 
    #     graphs_path = graphs_dir,
    # )
    # generate_hierarchical_results()

def train_base():
    bp =BasePipeline(dataset_folder_name=raw_dir)
    bp.build_generator()
    bp.build_model()
    bp.train_model(epochs=110,checkpoint_dir=model_dir)
    bp.build_30_cfms(epochs=110)

if __name__ == "__main__":
    train_base()