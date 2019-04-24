import subprocess

from model.BiLSTMCRFSeqTag import BiLSTMCRFSeqTag

from util.ditk_convertor_util import convert_data_to_ditk

blcst = BiLSTMCRFSeqTag()

file_dict = dict()
file_dict['train'] = '../data/sample/ner_test_input.txt'
file_dict['test'] = '../data/sample/ner_test_input.txt'
file_dict['dev'] = '../data/sample/ner_test_input.txt'

data = blcst.read_dataset(file_dict, "CoNLL2003")
blcst.train(data)
blcst.predict('../data/sample/ner_test_input.txt', writeInputToFile=False)
blcst.evaluate(None, None, None)


