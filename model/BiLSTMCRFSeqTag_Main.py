import subprocess

from model.BiLSTMCRFSeqTag import BiLSTMCRFSeqTag

from util.ditk_convertor_util import convert_data_to_ditk

blcst = BiLSTMCRFSeqTag()

file_dict = dict()
file_dict['train'] = '../data/example/tester.txt'
file_dict['test'] = '../data/example/tester.txt'
file_dict['dev'] = '../data/example/tester.txt'

data = blcst.read_dataset(file_dict, "CoNLL2003")
blcst.train(data)
predictions = blcst.predict('../data/example/tester.txt', writeInputToFile=False)
groundTruth = blcst.convert_ground_truth(data)
print(blcst.evaluate(None, None, None))
print(blcst.evaluate(groundTruth, [col[3] for col in predictions], None))


