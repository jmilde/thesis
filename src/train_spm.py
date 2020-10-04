from src.util_io import save_txt
from src.util_sp import load_spm, spm, encode
from src.util_np import np


def main( path_conditionals = "/data/eudata_conditionals.npz"
          path_txts = "/data/txts.txt"
          path_vocab = "/data/logo_vocab"):
    txts = [t.lower() for t in np.load(path_conditionals)["txts"] if t]

    # save text for sentence piece model
    save_txt(path_txts, txts)

    # train sentence piece model
    spm(name=path_vocab, path=path_txts, input_sentence_size=len(txts))
    print("Done!")

if __name__=="__main__":
    main( path_conditionals = "/data/eudata_conditionals.npz"
          path_txts = "/data/txts.txt"
          path_vocab = "/data/logo_vocab")
