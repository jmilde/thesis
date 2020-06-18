from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from src.util_np import np, vpack


def load_spm(path):
    """-> SentencePieceProcessor

    loads a sentence piece model.

    """
    spm = SentencePieceProcessor()
    spm.load(path)
    return spm


def spm(name, path, size= 8192, bos= 2, eos= 1, unk= 0, coverage= 0.9995 , input_sentence_size=1000000):
    """-> SentencePieceProcessor

    trains a sentence piece model of `size` from text file on `path`
    and saves with `name`.

    Additional training parameters:
    https://github.com/google/sentencepiece/blob/master/src/spm_train_main.cc
    """
    SentencePieceTrainer.train(
        "--model_prefix={name} \
        --input_sentence_size={input_sentence_size} \
        --input={path} \
        --vocab_size={size} \
        --bos_id={bos} \
        --eos_id={eos} \
        --unk_id={unk} \
        --character_coverage={coverage}".format(
            coverage= coverage
            , unk= unk
            , input_sentence_size=input_sentence_size
            , eos= eos
            , bos= bos
            , size= size
            , path= path
            , name= name))


def encode(vocab, sents, length= None, dtype= np.int32):
    """-> array dtype

    encodes `sents : seq str` with `vocab : SentencePieceProcessor`.
    returns a rank 2 array whose second axis is padded to `length` or
    the maximum length.

    """
    sents = list(map(vocab.encode_as_ids, sents))
    if length is None: length = max(map(len, sents))
    return vpack(sents, (len(sents), length), vocab.eos_id(), dtype)


def decode(vocab, array):
    """-> str

    decodes `array : array int` with `vocab : SentencePieceProcessor`.
    if `array` has a higher rank, generates the results instead.

    """
    if 1 < array.ndim: return (decode(vocab, arr) for arr in array)
    ids = list(map(int, array))
    try:
        ids = ids[:ids.index(vocab.eos_id())]
    except ValueError:
        pass
    return vocab.decode_ids(ids)
