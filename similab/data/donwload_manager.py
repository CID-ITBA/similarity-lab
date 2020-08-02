from pathlib import Path
import requests
import similab.data.constants as cts


def download_data(model="dw2v", corpus="nyt", path="None", update=False):
    """
    Fetch and decompress data into its own directory
    Parameters
    ----------
    update: bool
        If True, model will be downloaded again
    model: str
        Model name
    corpus:str
        Corpus name
    path:str
        Parent directory
    Returns
    -------
    """
    # Create 'models' directory to store models
    # If no path is provided data will be downloaded in the package directory

    models_path = Path(Path.cwd().parent.joinpath("models"))
    Path.mkdir(models_path, exist_ok=True)
    # Create 'the model' specific directory
    model_path = models_path.joinpath(corpus)
    Path.mkdir(model_path, exist_ok=True)
    # Download and write data to folder
    data_url = cts.DATA_URL_PATH.joinpath(str.upper(corpus))
    if corpus == "nyt":
        for file in cts.NYT_MODEL_FILES:
            file_url = data_url.joinpath(file).__str__()
            res = requests.get('http://personal.ik.itba.edu.ar/~cselmo/similab_models/NYT/testset_1-NYT.csv')
            # print(res.content)
            with open(model_path.joinpath(file), 'wb') as f:
                f.write(res.content)


download_data(model="dw2v", corpus="nyt")
print("hola")
