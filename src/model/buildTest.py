from model.buildData import get_data
from model.buildTrain import preprocess
import os




if __name__ == '__main__':

        #Cmdline arguments:
    #[1] - path to dataset directory
    dataset = sys.argv[1]
    preprocess("test","eval",dataset)

    lang="java"
    max_code_len = 100
    max_nl_len = 100
    code_unk_threshold = 2
    nl_unk_threshold = 2

    vocab = buildVocab(os.environ['CODENN_DIR'] + '/data/train.txt', code_unk_threshold, nl_unk_threshold, lang)

    #Build train and valid data
    get_data(os.environ['CODENN_DIR'] + '/data/eval.txt', vocab, False, max_code_len, max_nl_len)
