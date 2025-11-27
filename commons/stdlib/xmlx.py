import xmltodict
# import xml.etree.ElementTree as ET

def load(file: str, **kwargs):
    # return ET.parse(file)
    with open(file, mode="rb") as fin:
        return xmltodict.parse(fin)
