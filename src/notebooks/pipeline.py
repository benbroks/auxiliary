import sys 

import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from src.modelDesigns.baseModel import BasePipeline
from src.modelDesigns.auxModelTwoOutput import AuxTwoPipeline
from src.postprocess.graph_analysis import graph_pipeline

from auxiliary_partition.config import cfm_dir, partitions_dir, model_dir,raw_dir, graphs_dir

if __name__ == "__main__":
    #bm = BasePipeline()
    #bm.build_cfms()
    # set_partitions = graph_pipeline(
    #     base_cfms_path= cfm_dir / "base_cfms.npy", 
    #     partitions_path= partitions_dir / "dual_partitions.txt", 
    #     dual_partitions= True,
    # )
    ap = AuxTwoPipeline(dataset_folder_name=raw_dir)
    ap.train_models(
        checkpoint_path = model_dir / "aux_two_compressed_epoch",
        partitions_path=partitions_dir / "dual_partitions.txt",
        graphs_path = graphs_dir,
    )