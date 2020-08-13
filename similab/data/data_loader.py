from pathlib import Path
import pickle
import pandas as pd
import os
import similab.data.constants as cts
from ..data.donwload_manager import download_data
from ..data.model_ds import Model


def is_data_available(model: str, corpus: str, path: str) -> bool:
    """
    This function performs an availability against the

    Parameters
    ----------
    model
    corpus
    path

    Returns
    -------

    """
    if path is not None:
        path = Path(path)
        models_path = path.parent.joinpath('models')
    else:
        models_path = cts.DONWLOAD_PATH
    try:
        if corpus in os.listdir(models_path):  # Is the model in the given directory?
            if str.lower(corpus) == "nyt":
                files_in_path = set(os.listdir(models_path.joinpath(corpus)))
                # Do we have every file listed?
                if files_in_path == cts.NYT_MODEL_FILES:
                    return True
                else:
                    return False
            # TODO: UPDATE MUST CHECK ALL NECESSARY FILE ARE WITHIN THE DIRECTORY
    except FileNotFoundError as e:
        print("The models directory will be created")
        return False


# Si model y corpus no son None entonces tnemos que chequear si tenemos ese modelo y corpus disponible
# Si lo tenemos entonces nos fijamos si ya lo tenenemos descargado y update=False entonces devolvemos el objeto Model
def load_model(model="dw2v", corpus="nyt", path=None, update=False):
    """

    Parameters
    ----------
    model: str
        Name of the model
    corpus:
        Name of the corpus
    path:
        Default will be inside the package directory. To use costum models provide a path
    update:
        Default in False will not download again packages alrealdy in your system

    Returns
    -------

    """
    # Check for data existence
    if (model, corpus) in cts.AVAILABLE_MODELS:
        # Check if data is already available
        loaded_model = Model()
        # TODO: pensar una forma mas obvia de expresar que si no esta la data ==> queremos tratar de descargarla
        # TODO: y si no esta descargada y tampoco se puede descargar ==> volver
        if is_data_available(model=model, corpus=corpus, path=path) is False:
            if (model, corpus) in cts.AVAILABLE_MODELS:
                download_data(model=model, corpus=corpus, path=path, update=update)

        if is_data_available(model=model, corpus=corpus, path=path) is True:
            if path is not None:
                root = Path(path)  # user input path
                model_path = root
            # If no path is provided data will be downloaded in the package directory under models
            else:
                # Use current working directory to look for the models
                model_path = Path(cts.DONWLOAD_PATH).joinpath(corpus)
                for file in cts.NYT_MODEL_FILES:
                    if "delta" in file:
                        with open(model_path.joinpath(file), "rb") as f:
                            delta = pickle.load(f)
                        loaded_model.set_deltas(delta)
                    elif "mean" in file:
                        with open(model_path.joinpath(file), "rb") as f:
                            mean = pickle.load(f)
                        loaded_model.set_mean(mean)
                    elif "sampling_tables" in file:
                        with open(model_path.joinpath(file), "rb") as f:
                            sampling_tables = pickle.load(f)
                        loaded_model.set_sampling_tables(sampling_tables)
                    elif "word_index" in file:
                        with open(model_path.joinpath(file), "rb") as f:
                            word_index = pickle.load(f)
                            loaded_model.set_word_index(word_index)
                    elif "slices" in file:
                        with open(model_path.joinpath(file), "rb") as f:
                            slices = pickle.load(f)
                            loaded_model.set_slices(slices)
                    elif "testset" in file:
                        testset = pd.read_csv(model_path.joinpath(file))
                        loaded_model.add_test(testset)
                return loaded_model
    else:
        print("The requested model is not available")
        print(f"These are the currently available models")
        for m, c in cts.AVAILABLE_MODELS:
            print(f"model: {m}, corpus: {c}")
        return None
