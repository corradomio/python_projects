from r_compat import *

print(c(1,2,3))
print(rbind(c(1,2),c(3,4)))
print(cbind(c(1,2),c(3,4)))

m = cbind(c(1,2),c(3,4))

print(t(c(1,2)))
print(t(m))

print(seq(1,3))
print(rep(3,3))

print(m)
print(rep(m,3))

print(rep(seq(0,1).tolist(), 3))

print(as_matrix(expand_grid(rep(list_(seq(0,1)), 3))))
print(as_matrix(expand_grid(rep([0,1], 3))))

m=rbind(c(1,2),c(2,3),c(3,1),c(1,2),c(3,1))
print(m)
print(duplicated(m))

print(colSums(m))




