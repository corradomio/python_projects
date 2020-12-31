import fiona

for layername in fiona.listlayers('tests/data'):
    with fiona.open('tests/data', layer=layername) as src:
        print(layername, len(src))

# Output:
# (u'coutwildrnp', 67)

for i, layername in enumerate(fiona.listlayers('tests/data')):
    with fiona.open('tests/data', layer=i) as src:
        print(i, layername, len(src))

# Output:
# (0, u'coutwildrnp', 67)

# with open('tests/data/coutwildrnp.shp') as src:
#     meta = src.meta
#     f = next(src)

# with fiona.open('/tmp/foo', 'w', layer='bar', **meta) as dst:
#     dst.write(f)

# print(fiona.listlayers('/tmp/foo'))
#
# with fiona.open('/tmp/foo', layer='bar') as src:
#     print(len(src))
#     f = next(src)
#     print(f['geometry']['type'])
#     print(f['properties'])

# Output:
# [u'bar']
# 1
# Polygon
# OrderedDict([(u'PERIMETER', 1.22107), (u'FEATURE2', None), (u'NAME', u'Mount Naomi Wilderness'), (u'FEATURE1', u'Wilderness'), (u'URL', u'http://www.wilderness.net/index.cfm?fuse=NWPS&sec=wildView&wname=Mount%20Naomi'), (u'AGBUR', u'FS'), (u'AREA', 0.0179264), (u'STATE_FIPS', u'49'), (u'WILDRNP020', 332), (u'STATE', u'UT')])

for i, layername in enumerate(fiona.listlayers('zip://tests/data/coutwildrnp.zip')):
    with fiona.open('zip://tests/data/coutwildrnp.zip', layer=i) as src:
        print(i, layername, len(src))

# Output:
# (0, u'coutwildrnp', 67)


# with fiona.open('zip+s3://mapbox/rasterio/coutwildrnp.zip') as src:
#     print(len(src))

# Output:
# 67
