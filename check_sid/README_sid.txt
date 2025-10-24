G = (V, E)
V = [1..n]
E <= V x V

e = (i,j)
    i parent of j
    j child  of i

PA(G, j):   parents  of j
CH(G, i):   children of i
DE(G, i):   descendants of i (recursively)
ND(G, i):   non descendants of i = V - DE(G, i)
AN(G, j):   ancestors of j (recursively)

size:  number of edges
order: number of nodes

path <i_1,..,i_n>

collider:   a -> c <- b

a path <i_1,..,i_n> is blocked by a set S (i_1 and i_n NOT in S) IF

1)
    a->i->b
    a<-i<-b
    a<-i->b
2)
    a->i<-b  and i and DE(G, i) are NOT in S


A and B are d-separated by S if ALL path A->B are blocked by S
