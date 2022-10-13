from mrjob.job import MRJob
import sys


class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        yield "_chars", len(line)
        yield "_words", len(line.split())
        yield "_lines", 1
        yield "longest", len(line)

    def reducer(self, key, values):
        if key == "longest":
            yield key, max(values)
        else:
            yield key, sum(values)


if __name__ == '__main__':
    sys.argv = ['', '.\\README.rst']
    # print(sys.argv)
    MRWordFrequencyCount.run()
