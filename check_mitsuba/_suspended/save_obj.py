

def fwriteln(f, line):
    f.write(line)
    f.write('\n')


def to_face(ilist):
    face = ""
    for i in ilist:
        if len(face) > 0:
            face += " "
        face += str(i+1)
    return face
# end


def save_obj(fname, header, vertices, faces):
    with open(fname, 'w') as ff:
        fwriteln(ff, f"# {header}")
        for v in vertices:
            fwriteln(ff, f"v {v[0]} {v[1]} {v[2]}")

        for s in faces:
            fwriteln(ff, f"f {to_face(s)}")
# end
