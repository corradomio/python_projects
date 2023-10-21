import torch
import torch.nn as nn

#   embed_dim, num_heads, kdim=None, vdim=None,
#   dropout=0., bias=True,
#   add_bias_kv=False, add_zero_attn=False,
#   batch_first=False,
#   device=None, dtype=None
#
#   if not qkv_same_embed_dim
#       q_proj_weight   (embed_dim, embed_dim)
#       k_proj_weight   (embed_dim, kdim)
#       v_proj_weight   (embed_dim, vdim)
#   else
#       in_proj_weight  (3, embed_dim, embed_dim)
#
#   out_proj            (embed_dim, embed_dim)
#

A = nn.MultiheadAttention(1, 1, dtype=float, bias=False, batch_first=True)


def eval_attn(q, k, v):
    with torch.no_grad():
        a = A(q, k, v)[0].detach()
        print(a)


def test01():
    with torch.no_grad():
        # Nota:
        # (q, k, [[1],[1],[1]]) ==
        # (q, k, [[1],[0],[0]]) + (q, k, [[0],[1],[0]]) + (q, k, [[0],[0],[1]])
        #
        # Nota:
        # (q, k, [[1],[0],[0]]) == (q, k, [[0],[1],[0]]) == (q, k, [[0],[0],[1]])

        # --
        print("1)")
        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)

        print("2)")
        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)

        print("3)")
        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[0], [0], [0]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)


def test02():
    # Nota:
    #      ([[1],[1],[1]], k, v)
    #   == ([[1],[0],[0]], k, v) == ([[0],[1],[0]]), k, v) == ([[0],[0],[1]]), k, v)
    #   == ([[0],[0],[0]], k, v)
    print("1)")
    q = torch.tensor([[[0], [0], [0]]], dtype=float)
    k = torch.tensor([[[0], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[1], [1], [1]]], dtype=float)
    k = torch.tensor([[[0], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[1], [0], [0]]], dtype=float)
    k = torch.tensor([[[0], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [1], [0]]], dtype=float)
    k = torch.tensor([[[0], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [0], [1]]], dtype=float)
    k = torch.tensor([[[0], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)


def test03():
    print("1)")
    q = torch.tensor([[[0], [0], [0]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[1], [1], [1]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[1], [0], [0]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [1], [0]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [0], [1]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)


def test11():
    with torch.no_grad():
        # Nota:
        # (q, k, [[1],[1],[1]]) ==
        # (q, k, [[1],[0],[0]]) + (q, k, [[0],[1],[0]]) + (q, k, [[0],[0],[1]])
        #
        # Nota:
        # (q, k, [[1],[0],[0]]) == (q, k, [[0],[1],[0]]) == (q, k, [[0],[0],[1]])

        # --
        print("1)")
        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[1], [0], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)

        print("2)")
        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [1], [0]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)

        print("3)")
        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[1], [0], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [1], [0]]], dtype=float)
        eval_attn(q, k, v)

        q = torch.tensor([[[0], [0], [1]]], dtype=float)
        k = torch.tensor([[[1], [1], [1]]], dtype=float)
        v = torch.tensor([[[0], [0], [1]]], dtype=float)
        eval_attn(q, k, v)


def test21():
    print("1) self")
    q = torch.tensor([[[1], [0], [0]]], dtype=float)
    k = torch.tensor([[[1], [0], [0]]], dtype=float)
    v = torch.tensor([[[1], [0], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [1], [0]]], dtype=float)
    k = torch.tensor([[[0], [1], [0]]], dtype=float)
    v = torch.tensor([[[0], [1], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [0], [1]]], dtype=float)
    k = torch.tensor([[[0], [0], [1]]], dtype=float)
    v = torch.tensor([[[0], [0], [1]]], dtype=float)
    eval_attn(q, k, v)

    print("2) self")
    q = torch.tensor([[[1], [1], [0]]], dtype=float)
    k = torch.tensor([[[1], [1], [0]]], dtype=float)
    v = torch.tensor([[[1], [1], [0]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[0], [1], [1]]], dtype=float)
    k = torch.tensor([[[0], [1], [1]]], dtype=float)
    v = torch.tensor([[[0], [1], [1]]], dtype=float)
    eval_attn(q, k, v)

    q = torch.tensor([[[1], [0], [1]]], dtype=float)
    k = torch.tensor([[[1], [0], [1]]], dtype=float)
    v = torch.tensor([[[1], [0], [1]]], dtype=float)
    eval_attn(q, k, v)

    print("3) self")
    q = torch.tensor([[[1], [1], [1]]], dtype=float)
    k = torch.tensor([[[1], [1], [1]]], dtype=float)
    v = torch.tensor([[[1], [1], [1]]], dtype=float)
    eval_attn(q, k, v)



def main():
    # test01()
    # test02()
    # test03()
    # test11()
    test21()
    pass

if __name__ == "__main__":
    main()