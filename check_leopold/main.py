# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from random import randrange, randint


def random_pos(building):
    # return [randrange(0, building[0]), randrange(0, building[1])]
    return [randint(0, building[0]-1), randint(0, building[1]-1)]


def where_is_bomb(batman, bomb):
    # riga
    pos = ""
    if batman[0] < bomb[0]:
        pos += "D"
    elif batman[0] > bomb[0]:
        pos += "U"
    else:
        pos += ""   # stessa riga

    # colonna
    if batman[1] < bomb[1]:
        pos += "R"
    elif batman[1] > bomb[1]:
        pos += "L"
    else:
        pos += "" # stessa colonna
    return pos
# end


#      0      1
#    +-----+-----+---
#  0 | 0,0 | 0,1 |
#    +-----+-----+---
#  1 | 1,0 | 1,1 |
#    +-----+-----+---
#    |     |     |
#
#   coords: (row, column)


# x1, x2 -> x1 + (x2-x1)/2 -> (2x1 + x2 - x1)/2 -> (x1 + x2)/2



# building dimension
building = [100, 200]

ul = [0, 0]
dr = building

bomb = random_pos(building)
batman = random_pos(building)


print("Batman in ", batman)

while True:
    pos = where_is_bomb(batman, bomb)
    print(pos)

    if pos == "":
        print("Trovata la bomba in ", batman)
        break

    if "U" in pos:
        dr[0] = batman[0] - 1
    elif "D" in pos:
        ul[0] = batman[0] + 1
    else:
        pass

    if "L" in pos:
        dr[1] = batman[1] - 1
    elif "R" in pos:
        ul[1] = batman[1] + 1
    else:
        pass

    batman = [(ul[0] + dr[0]) // 2, (ul[1] + dr[1]) // 2]

