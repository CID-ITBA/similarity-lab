from ..data import data_loader as cl


class Laboratory:
    def __init__(self, matrices, year_dict, vocabularies):
        pass

    def similars2vec(self, vector, year, threshold=0, max_words=None):
        """
        Searches for the most similar words within a word2vec embedding matrix.

        Looks for the most similar words within a word2vec embedding matrix.
        This method uses the cosine similarities between the embedding vectors
        and the given vector. Returning the most similar words within a given treshold
        or neighbourhood.

        Parameters
        --------

        vector : array_like
            Input Vector. Must match embedding dimension.
        year : int
            Choosen year.
        threshold : float
            Minimun cosine similarity allowed to consider a word 'close' to the given vector.
            If left blank, the default value is 0, which allows for all vectors to be considered.
        max_words : int
            Maximum number of words to be returned as most similar.
            If left blank, the default value is 'None', which allows for all vectors to be considered.

        Returns
        -------
        out : dict
            A dictionary containing the words found and its
            cosine similarities with respect to given input vector.
        Raises
        ------
        ValueError
            if 'treshold' is a negative floating point number, or it's value is greater than 1.0.
            if 'year' is not present in the current data.
        """
        pass
