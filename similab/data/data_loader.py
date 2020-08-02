from pathlib import Path
import os
import requests
from .constants import *


def is_data_available(model, corpus):
    if (model, corpus) in AVAILABLE_MODELS:
        return True
    else:
        print(f"The current model: {model} and corpus: {corpus} requested is not available")
        return False 

# Si model y corpus no son None entonces tnemos que chequear si tenemos ese modelo y corpus disponible
# Si lo tenemos entonces nos fijamos si ya lo tenenemos descargado y update=False entonces devolvemos el objeto Model
def load_model(model=None,corpus=None,path=None,update=False):
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
    pass

