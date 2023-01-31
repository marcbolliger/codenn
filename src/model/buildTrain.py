from model.buildData import get_data
import os
import json
import sys
import gzip


#Convert the dataset stored in mlmfc to a dataset expected by the model
def preprocess(outtype, outname, datapath):
    print("Building for: "+outtype, flush=True)
    jsonpath = datapath+outtype+"/"+outtype+".jsonl.gz"
    outpath = os.environ['CODENN_DIR'] + '/data/'+outname+'.txt'

    with gzip.open(jsonpath) as f:
        with open(outpath, 'w') as outfile:
            for fid, line in enumerate(f):
                data = json.loads(line)
                com = data["docstring"]
                code = data["code"]
                outfile.write("0, {}, {}, {}, 0".format(fid, com, code))

if __name__ == '__main__':

    #Cmdline arguments:
    #[1] - path to dataset directory
    dataset = sys.argv[1]
    preprocess("train","train", dataset)
    preprocess("valid","valid", dataset)

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
