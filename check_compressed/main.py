import gzip
import os

outfilename = 'example.txt.gz'
output = gzip.open(outfilename, 'wt')
try:
    output.write('Contents of the example file go here.\n')
finally:
    output.close()

print(outfilename, 'contains', os.stat(outfilename).st_size, 'bytes of compressed data')

