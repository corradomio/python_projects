import fiona

# Open a file for reading. We'll call this the "source."

with fiona.open('tests/data/coutwildrnp.shp') as src:

    # The file we'll write to, the "destination", must be initialized
    # with a coordinate system, a format driver name, and
    # a record schema.  We can get initial values from the open
    # collection's ``meta`` property and then modify them as
    # desired.

    meta = src.meta
    meta['schema']['geometry'] = 'Point'

    # Open an output file, using the same format driver and
    # coordinate reference system as the source. The ``meta``
    # mapping fills in the keyword parameters of fiona.open().

    with fiona.open('test_write.shp', 'w', **meta) as dst:

        # Process only the records intersecting a box.
        for f in src.filter(bbox=(-107.0, 37.0, -105.0, 39.0)):

            # Get a point on the boundary of the record's
            # geometry.

            f['geometry'] = {
                'type': 'Point',
                'coordinates': f['geometry']['coordinates'][0][0]}

            # Write the record out.

            dst.write(f)

# The destination's contents are flushed to disk and the file is
# closed when its ``with`` block ends. This effectively
# executes ``dst.flush(); dst.close()``.
