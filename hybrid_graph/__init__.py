import os
import pathlib

DATA_DIR = os.path.join(pathlib.Path(__file__).parent.parent.resolve(),'datasets')
os.environ["HG_DATA_DIR"] = DATA_DIR
