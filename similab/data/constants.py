from pathlib import Path
DONWLOAD_PATH = Path.cwd().parent.joinpath("models")
AVAILABLE_MODELS = [("dw2v", "nyt")]

# Data will be downloaded from this url
DATA_URL_PATH = 'http://personal.ik.itba.edu.ar/~cselmo/similab_models/'

# Temporary solution
NYT_MODEL_FILES = {"delta-NYT.pck", "mean-NYT.pck", "sampling_tables-NYT.pkl", "slices-NYT.pkl",
                   "testset_1-NYT.csv", "testset_2(1)-NYT.csv", "testset_2(2)-NYT.csv", "word_index-NYT.pkl"}

# NYT_MODEL_FILES = {'testset_1-NYT.csv', 'testset_2(1)-NYT.csv', 'testset_2(2)-NYT.csv', 'word_index-NYT.pkl'}

TRIES = 5
