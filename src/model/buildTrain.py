from model.buildData import get_data
import os

if __name__ == '__main__':

    lang = "java"
    max_code_len = 100
    max_nl_len = 100
    code_unk_threshold = 2
    nl_unk_threshold = 2

    vocab = buildVocab(os.environ['CODENN_DIR'] + '/data/train.txt', code_unk_threshold, nl_unk_threshold, lang)

    #Build train and valid data, use empty file for dev.txt
    get_data(os.environ['CODENN_DIR'] + '/data/train.txt', vocab, False, max_code_len, max_nl_len)
    get_data(os.environ['CODENN_DIR'] + '/data/valid.txt', vocab, False, max_code_len, max_nl_len)
    get_data(os.environ['CODENN_DIR'] + '/data/dev.txt', vocab, True, max_code_len, max_nl_len)
