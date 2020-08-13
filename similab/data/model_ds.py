import numpy as np


class Model:
    """
    This Model class contains and preprocess the requested dataset
    """
    def __init__(self):
        self.deltas = None
        self.mean = None
        self.sampling_tables = None
        self.slices = None
        self.word_index = None
        self.tests = []

    def set_deltas(self, deltas):
        self.deltas = deltas

    def set_mean(self, mean):
        self.mean = mean

    def set_sampling_tables(self, sampling_tables):
        self.sampling_tables = sampling_tables

    def set_slices(self, slices):
        self.slices = slices

    def set_word_index(self, word_index):
        self.word_index = word_index

    def add_test(self, tests):
        self.tests.append(tests)

    def get_embedding(self):
        """
        Preprocess mean and delta matrices to produce the embedding matrics
        and its corresponding vocabularies per slice

        Returns
        -------

        """
        # Create the embeddings matrices
        if self.mean is not None and self.deltas is not None and self.word_index is not None:
            embs = list()
            year_dict = dict()
            vocabularies = list()
            for idx, slc in enumerate(self.slices):
                mask = np.array(self.sampling_tables[slc]) != 0
                embs.append((self.mean + self.deltas[slc])[mask, :])
                year_dict[slc] = idx
                vocabulary = np.array(list(self.word_index.keys()))[mask]
                vocabulary = {word: index for index, word in enumerate(vocabulary)}
                vocabularies.append(vocabulary)

            return embs, year_dict, vocabularies
        else:
            return None
