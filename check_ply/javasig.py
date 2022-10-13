from typing import List, Tuple

from ply.lex import lex
from ply.yacc import yacc


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

tokens = [
    'NAME',
    'CLASS', 'INTERFACE', 'ENUM',
    'PRIVATE', 'PROTECTED', 'PUBLIC',
    'STATIC', 'FINAL', 'ABSTRACT', 'SYNCHRONIZED',
    'EXTENDS', 'IMPLEMENTS',
    'THROWS', 'ANNOTATION', 'DOTS'
]

literals = [
    '<', '>', '[', ']', '{', '}', '(', ')', '/', '*', ',', '&'
]

t_ignore_SPACE = r'[ \r\t\n]'
t_ignore_LINECOMMENT = r'//.*\n'
t_ignore_BLOCKCOMMENT = r'/\*.*\*/'

# t_NAME = r'[a-zA-Z_$@][a-zA-Z0-9._$@]*'
t_CLASS = 'class'
t_INTERFACE = 'interface'
t_EXTENDS = 'extends'
t_IMPLEMENTS = 'implements'
t_ENUM = 'enum'
t_PRIVATE = 'private'
t_PROTECTED = 'protected'
t_PUBLIC = 'public'
t_STATIC = 'static'
t_FINAL = 'final'
t_ABSTRACT = 'abstract'
f_SYNCHRONIZED = 'synchronized'
t_THROWS = 'throws'

KEYWORDS = {
    #
    # types
    #
    t_CLASS,
    t_INTERFACE,
    t_ENUM,
    #
    # hierarchy
    #
    t_EXTENDS,
    t_IMPLEMENTS,
    #
    # modifiers
    #
    t_PRIVATE,
    t_PROTECTED,
    t_PUBLIC,
    t_STATIC,
    t_FINAL,
    t_ABSTRACT,
    f_SYNCHRONIZED,
    #
    # method
    #
    t_THROWS
}


#
# a passible 'name' is also '?'
#
def t_NAME(t):
    r'\?|\.\.\.|[a-zA-Z0-9_$@]+(\.[a-zA-Z0-9_$@]+)*'
    value: str = t.value
    if t.value in KEYWORDS:
        t.type = value.upper()
    elif t.value.startswith('@'):
        t.type = 'ANNOTATION'
    elif t.value == "...":
        t.type = 'DOTS'
    return t


def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class JSyntax:
    pass


class Typearg:

    def __init__(self, name: str, tbounds: List["Decltype"]):
        self.name = name
        self.tbounds = tbounds
    # end
# end


class Decltype(JSyntax):
    def __init__(self, name: str, tparams: List["Decltype"] = [], rank: int = 0, ent=None):
        self.name: str = name
        self.tparams: List[Decltype] = tparams
        self.rank: int = rank
        self.ent = ent
        self.declaration: str = ''
        self._sig: str = None
    # end

    @property
    def ntparams(self) -> int:
        return len(self.tparams)

    @property
    def signature(self):

        def arrays(rank):
            a = ""
            for i in range(rank):
                a += "[]"
            return a
        # end

        def templates(tparams):
            if self.ntparams == 0:
                return ""
            templ = ""
            for i, tparam in enumerate(tparams):
                if i == 0:
                    templ += "<"
                else:
                    templ += ","
                templ += tparam.signature
            templ += ">"
            return templ
        # end

        if self._sig is None:
            if self.ntparams == 0 and self.rank == 0:
                self._sig = self.name
            if self.ntparams == 0:
                self._sig = self.name + arrays(self.rank)
            if self.rank == 0:
                self._sig = self.name + templates(self.tparams)
            else:
                self._sig = self.name + templates(self.tparams) + arrays(self.rank)
        # end

        return self._sig
    # end

    def __repr__(self):
        return f"decltype({self.name},{self.tparams}, {self.rank})"
# end


class Method(JSyntax):
    def __init__(self, name: str, modifiers: List[str],
                 returntype: Decltype,
                 parameters: List["Parameter"],
                 throws: List[Decltype]):

        assert type(returntype) == Decltype

        self.name: str = name
        self.modifiers: List[str] = modifiers
        self.type: Decltype = returntype
        self.parameters: List["Parameter"] = parameters
        self.throws: List[Decltype] = throws
        self.declaration: str = None
    # end

    @property
    def nparameters(self):
        return len(self.parameters)

    def parameter(self, name: str) -> "Parameter":
        for param in self.parameters:
            if param.name == name:
                return param
        raise ValueError(f"Parameter {name} not found in {self.declaration}")
    # end

    def __repr__(self):
        return f"method({self.name}, {self.modifiers}, {self.type}, {self.parameters})"
# end


class Parameter(JSyntax):
    def __init__(self, name: str, modifiers: List[str], type: Decltype):
        self.name: str = name
        self.modifiers: List[str] = modifiers
        self.type: Decltype = type
    # end

    def __repr__(self):
        return f"parameter({self.name},{self.type})"


class JavaType(JSyntax):
    def __init__(self, name: str, targs: List[Typearg], role: str, modifiers: List[str],
                 extimpl: Tuple[List[Decltype], List[Decltype]]):
        self.name: str = name
        self.targs: List[Typearg] = targs
        self.role: str = role
        self.modifiers: List[str] = modifiers
        self.extends: List[Decltype] = extimpl[0]
        self.implements: List[Decltype] = extimpl[1]
        self.declaration: str = None
    # end

    def empty(self) -> bool:
        return len(self.targs) == 0 and len(self.extends) == 0 and len(self.implements)

    def __repr__(self):
        return f"{self.role}({self.name},{self.targs}, {self.modifiers}, e:{self.extends}, i:{self.implements})"
# end


# --- Parser (type)

def p_type(p):
    """
    type : NAME tparams array
    """
    p[0] = Decltype(p[1], p[2], p[3])


def p_tparams(p):
    """
    tparams : '<' types '>'
            |
    """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = p[2]


def p_types(p):
    """
    types : type ',' types
          | type
          |
    """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_andtypes(p):
    """
    andtypes : type '&' andtypes
             | type
    """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_array(p):
    """
    array : '[' ']' array
          |
    """
    if len(p) == 1:
        p[0] = 0
    elif len(p) == 3:
        p[0] = 1
    else:
        p[0] = 1 + p[3]


# --- Parser (common)

def p_modifier(p):
    """
    modifier : PUBLIC
             | PROTECTED
             | PRIVATE
             | STATIC
             | FINAL
             | ABSTRACT
             | ANNOTATION
             | SYNCHRONIZED
    """
    p[0] = p[1]


def p_modifiers(p):
    """
    modifiers : modifier modifiers
              |
    """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]


def p_tlist(p):
    """
    tlist  : type ',' tlist
           | type
           |
    """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]


# --- Parser (method)
#
#  type name(param, param)

def p_targs(p):
    """
    targs : '<' targlist '>'
          |
    """
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = p[2]


def p_targlist(p):
    """
    targlist : targ ',' targlist
             | targ
    """
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_targ(p):
    """
    targ : NAME tbounds
         | NAME
    """
    if len(p) == 2:
        p[0] = Typearg(p[1], [Decltype("Object")])
    else:
        p[0] = Typearg(p[1], p[2])


def p_tbounds(p):
    """
    tbounds : EXTENDS andtypes
    """
    p[0] = p[1]


def p_callable(p):
    """
    callable : constructor
             | method
    """
    p[0] = p[1]


def p_constructor(p):
    """
    constructor : modifiers NAME '(' plist ')' throws
    """
    ctype = Decltype("void")
    p[0] = Method(p[2], p[1], ctype, p[4], p[6])


def p_method(p):
    """
    method  : modifiers targs type NAME '(' plist ')' throws
            | modifiers type NAME '(' plist ')' throws
    """
    if len(p) == 9:
        p[0] = Method(p[4], p[1], p[3], p[6], p[8])
    else:
        p[0] = Method(p[3], p[1], p[2], p[5], p[7])


def p_throws(p):
    """
    throws : THROWS types
           |
    """
    p[0] = [] if len(p) == 1 else p[2]


def p_plist(p):
    """
    plist : param ',' plist
          | param
          |
    """
    if len(p) == 1:
        p[0] = []
    elif len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[3]


def p_param(p):
    """
    param : modifiers type DOTS NAME
          | modifiers type NAME
    """
    if len(p) == 5:
        p[2].rank += 1
        p[0] = Parameter(p[4], p[1], p[2])
    else:
        p[0] = Parameter(p[3], p[1], p[2])


# --- Parser (class/interface/enum)
#
#  (modifier)* 'class|interface' name (tparams)? ('extends' type(',' type)*)? ('implements' type (, type)*)?

def p_classinterface(p):
    """
    classinterface : modifiers trole NAME targs extimpl
    """
    name = p[3]
    modifiers = p[1]
    targs = p[4]
    role = p[2]
    extimpl = p[5]
    p[0] = JavaType(name, targs, role, modifiers, extimpl)


def p_trole(p):
    """
    trole : CLASS
          | INTERFACE
          | ENUM
    """
    p[0] = p[1]


def p_extimpl(p):
    """
    extimpl : EXTENDS    tlist IMPLEMENTS tlist
            | IMPLEMENTS tlist EXTENDS    tlist
            | EXTENDS    tlist
            | IMPLEMENTS tlist
            |
    """
    # extends, implements
    if len(p) == 1:
        p[0] = ([], [])
    elif len(p) == 5 and p[1] == t_EXTENDS:
        p[0] = (p[2], p[4])
    elif len(p) == 5 and p[1] == t_IMPLEMENTS:
        p[0] = (p[4], p[2])
    elif len(p) == 3 and p[1] == t_EXTENDS:
        p[0] = (p[1], [])
    elif len(p) == 3 and p[1] == t_IMPLEMENTS:
        p[0] = ([], p[1])
    else:
        p[0] = ([], [])


def p_error(p):
    print(f'Syntax error at {p.value!r} on {p.lexer.lexdata}')


lexer = lex()
# lexer.input("java.util.Map< String/* ciccio /* */, List < String >// ciccio \n[ ] [ ]")
# tok = lexer.token()
# while tok is not None:
#     print(tok)
#     tok = lexer.token()

_method_parser   = yacc(start='callable')
_javatype_parser = yacc(start='classinterface')
_decltype_parser = yacc(start='type')


def parse_method_declaration(declaration: str) -> Method:
    try:
        m: Method = _method_parser.parse(declaration)
        # if m is None:
        #     raise ValueError(f"Unsupported method syntax '{declaration}'")
        m.declaration = declaration
        return m
    except Exception as e:
        raise ValueError(f"Unsupported method syntax '{declaration}'")
# end


def parse_decltype_declaration(declaration: str) -> Decltype:
    try:
        dt: Decltype = _decltype_parser.parse(declaration)
        # if dt is None:
        #     raise ValueError(f"Unsupported decltype syntax '{declaration}'")
        dt.declaration = declaration
        return dt
    except Exception as e:
        raise ValueError(f"Unsupported decltype syntax '{declaration}'")
# end


def parse_type_declaration(declaration: str) -> JavaType:
    try:
        jt: JavaType = _javatype_parser.parse(declaration)
        # if jt is None:
        #     raise ValueError(f"Unsupported type declaration syntax '{declaration}'")
        jt.declaration = declaration
        return jt
    except Exception as e:
        raise ValueError(f"Unsupported type declaration syntax '{declaration}'")
# end
