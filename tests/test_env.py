import kloppy
import lightgbm
import numpy as np
import pandas as pd
import sklearn


def test_dependencies_importables():
    assert kloppy.__version__ == "3.18.0"
    assert pd.__version__.split(".")[0] == "2"
    assert np.__version__
    assert lightgbm.__version__
    assert sklearn.__version__
