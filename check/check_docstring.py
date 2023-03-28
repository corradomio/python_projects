# from sklearn.linear_model import LogisticRegression
from language import import_from
from docstring_parser import parse
from pprint import pprint



def main():
    qname = 'sklearn.linear_model.LogisticRegression'
    clazz = import_from(qname)

    clazz_doc = clazz.__doc__

    docstring = parse(clazz_doc)

    print("short_description:\n", docstring.short_description)
    print("\nlong_description:\n", docstring.long_description)
    print("\n")
    print("style", docstring.style)
    print("params", len(docstring.params))
    for p in docstring.params:
        print(f"... n:'{p.arg_name}', d:'{p.default}', t:'{p.type_name}'")

    pass


if __name__ == "__main__":
    main()