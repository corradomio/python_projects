from stringx.tokenizer.qgram_tokenizer import QGramTokenizer
from stringx.tokenizer.regexp_tokenizer import RegExpTokenizer
from stringx.tokenizer.split_tokenizer import SplitTokenizer
from stringx.tokenizer.resplit_tokenizer import RESplitTokenizer
from stringx.tokenizer.delim_tokenizer import DelimTokenizer


def main():

    qgt = QGramTokenizer(unique=False)
    ret = RegExpTokenizer('[a-zA-Z0-9]+', unique=False)
    spt = SplitTokenizer(',', unique=False)
    rst = RESplitTokenizer(r'\s,')
    dlt = DelimTokenizer()

    print(qgt.tokenize("ciao ciccio come stai io bene grazie"))
    print(ret.tokenize("ciao ciccio come stai io bene grazie"))
    print(spt.tokenize("ciao,ciccio,come,stai,io,bene,grazie"))
    print(rst.tokenize("ciao, come stai? Io molto bene! Grazie! Ma, non credo sia il caso. Meglio la prossima volta"))
    print(dlt.tokenize("ciao, come stai? Io molto bene! Grazie! Ma, non credo sia il caso. Meglio la prossima volta"))
    pass


if __name__ == "__main__":
    main()