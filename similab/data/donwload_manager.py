from pathlib import Path
import requests
import similab.data.constants as cts
from tqdm import tqdm

"""
The sole porpouse of this module is to download certain data from an a specfic server and write it down to memory 
"""


def download_data(model="dw2v", corpus="nyt", path=None, update=False):
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
    download_status: int
        status code received from request
    """

    if path is not None:
        root = Path(path)
        ans = input(f"The path given is {path}"
                    f"\nDo want to proceed Y/N: ")
        if str.strip(ans) in "yYyesYes":
            models_path = root
        else:
            print("Download aborted")
            return False
    # If no path is provided data will be downloaded in the package directory under models
    else:
        # Use current working directory to downlaod the models
        models_path = Path(cts.DONWLOAD_PATH)
        print(models_path)

    Path.mkdir(models_path, exist_ok=True)
    # Create 'the model' specific directory
    model_path = models_path.joinpath(corpus)
    Path.mkdir(model_path, exist_ok=True)
    # Download and write data to folder
    data_url = cts.DATA_URL_PATH + str.upper(corpus)

    download_status = None
    # Se necesita un esquema de nombres más consistente
    # TODO: refactor to take into acount different model type
    if corpus == "nyt":
        for file in tqdm(cts.NYT_MODEL_FILES):
            file_url = data_url + "/" + file
            res = requests.get(file_url, stream=True)
            download_status = res.status_code

            # In case of failure try to download again
            for i in range(cts.TRIES):
                if download_status != requests.codes["all_good"]:
                    res = requests.get(file_url, stream=True)
                    download_status = res.status_code
                else:
                    break  # Everything fine
            if download_status != requests.codes["all_good"]:
                print(f"Download of file {file} failed. Error code {download_status} ")
                return False
            # Write file to memory
            elif download_status == requests.codes["all_good"]:
                with open(model_path.joinpath(file), 'wb') as f:
                    f.write(res.content)

    return download_status
