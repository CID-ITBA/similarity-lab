from pathlib import Path

# Default download site
DONWLOAD_PATH = Path.cwd().parent.joinpath("models")

AVAILABLE_MODELS = [("dw2v", "nyt")]

# Data will be downloaded from this url
DATA_URL_PATH = 'http://personal.ik.itba.edu.ar/~cselmo/similab_models/'

NYT_MODEL_FILES = {"delta-NYT.pck", "mean-NYT.pck", "sampling_tables-NYT.pkl", "slices-NYT.pkl",
                   "testset_1-NYT.csv", "testset_2(1)-NYT.csv", "testset_2(2)-NYT.csv", "word_index-NYT.pkl"}

# If information retrieve fails try at least 5 times before giving up
TRIES = 5
