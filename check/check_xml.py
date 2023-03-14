import xmltodict
from pprint import pprint

with open('D:/Dropbox/Projects/Projects.ebtic/ipredict.v3.4/datasources.xml') as fd:
    doc = xmltodict.parse(fd.read())
    pprint(doc)


import xml.etree.ElementTree as ET
tree = ET.parse('D:/Dropbox/Projects/Projects.ebtic/ipredict.v3.4/datasources.xml')
root = tree.getroot()
pprint(tree)
pprint(root)