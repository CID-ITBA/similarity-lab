from similab import __version__
import similab as sm
import unittest


def test_version():
    assert __version__ == '0.1.0'


class TestEvo(unittest.TestCase):

    def test_model_load1(self):
        m = sm.load_model(model="dw2v", corpus="nyt", path='C:/Users/Msrt/Documents/GitHub/similarity-lab/models')
        self.assertIsInstance(m.word_index, dict)
        self.assertIsNotNone(m)

    def test_model_load2(self):
        m = sm.load_model(model="dw2v", corpus="nyt", path="C:/Users/Msrt/Desktop/models")
        self.assertIsInstance(m.word_index, dict)
        self.assertIsNotNone(m)

    def test_model_load_none(self):
        m3 = sm.load_model(model="no_exist", corpus="no_exist")
        self.assertIsNone(m3)


if __name__ == '__main__':
    unittest.main()
