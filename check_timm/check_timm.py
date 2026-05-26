from pprint import pprint

from transformer_lens import pretrained

import timm


def main():
    pprint(timm.list_models())
    # pprint(timm.list_pretrained())
    # pprint(timm.list_modules())
    pass


if __name__ == '__main__':
    main()
