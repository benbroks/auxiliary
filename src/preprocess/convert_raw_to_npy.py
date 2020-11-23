import sys 

import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import TEST_SPLIT

if __name__ == "__main__":
    print(TEST_SPLIT)

