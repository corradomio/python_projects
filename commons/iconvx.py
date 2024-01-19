
_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_EQ = ord('=')


def tob64(i: int):
    if i == 0:
        b64 = 'A'
    else:
        b64 = ""
    while i > 0:
        b64 += (_BASE64[i % 64])
        i >>= 6
    return b64


def fromb64(s):
    i = 0
    for k in range(len(s)-1, -1, -1):
        i <<= 6
        c = ord(s[k])
        if c == 61:             # ord('='):
            continue
        elif c == 43:           # ord('+'):
            i += 62
        elif c == 47:           # ord('/'):
            i += 63
        elif c <= 57:           # ord('9'):
            i += c + 4          # - ord('0') + 52
        elif c <= 90:           # ord('Z'):
            i += c - 65         # ord('A')
        elif c <= 122:          # ord('z')
            i += c - 71         # - ord('a') + 26
        else:
            raise ValueError("Invalid base64 number: " + s)
    return i

