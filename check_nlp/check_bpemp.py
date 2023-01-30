# https://bpemb.h-its.org/

from bpemb import BPEmb
bpemb_en = BPEmb(lang="en", vs=100000)
print(bpemb_en.encode("Stratford"))

print(bpemb_en.encode("This is anarchism"))

print(bpemb_en.encode("cocome"))
print(bpemb_en.encode("ipredict"))
print(bpemb_en.encode("Supercalifragilisticexpialidocious"))


bpemb_it = BPEmb(lang="it", vs=10000)
print(bpemb_it.encode("precipitevolissimevolmente"))
print(bpemb_it.encode("supercalifragilistichespiralidoso"))
