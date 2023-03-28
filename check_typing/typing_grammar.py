import ply.lex as lex
import ply.yacc as yacc

TYPES = {
    'object': object,
    'int': int,
    'float': float,
    'complex': complex,
    'str': str,
    'list': list,
    'tuple': tuple,
    'bytes': bytes,
    'bytearray': bytearray,
    'memoryview': memoryview,
    'set': set,
    'frozenset': frozenset,
    # 'frozenlist': frozenlist,
    'dict': dict,
    'iter': iter,
    # 'generator': generator,

}


class HintType:

    def __init__(self, name, args=[]):
        self.name = name
        self.args = args

    def print(self, depth=0):
        def ident():
            spaces = ""
            for i in range(depth):
                spaces += "  "
            return spaces

        print(f"{ident()}{self.name}")
        for arg in self.args:
            arg.print(depth+1)
# end


tokens = ('QNAME', )

literals = "[]|,"

t_QNAME = '[A-Za-z_.~+-]+'

t_ignore = ' \t'


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


lexer = lex.lex()


def p_type(p):
    """
    type : QNAME
         | QNAME '[' typeargs ']'
         | '[' typeargs ']'
         | type '|' uniontypes
         |
    """
    if len(p) == 2:
        p[0] = HintType(p[1])
    elif len(p) == 5:
        p[0] = HintType(p[1], p[3])
    elif len(p) == 4 and p[1] == '[':
        p[0] = HintType('params', p[2])
    else:
        p[0] = HintType('typing.Union', [p[1]] + p[3])


def p_typeargs(p):
    """
    typeargs : type ',' typeargs
             | type
             |
    """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_uniontypes(p):
    """
    uniontypes : type '|' uniontypes
               | type
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]



def p_error(p):
    print(f"Syntax error in input {p}!")


_parser = yacc.yacc()


def parse(t: type) -> HintType:
    def to_str(t: type):
        if t in [bool, int, float, str]:
            return t.__name__
        else:
            return str(t)
    return _parser.parse(to_str(t))

