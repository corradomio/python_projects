import ply.lex as lex
import ply.yacc as yacc

# https://www.graphviz.org/doc/info/lang.html

# graph ::= 'strict'? ('graph'| 'digraph') ID {
#       (vertex | edge)*
# }
#
# vertex ::= NUMBER attributes? ;
#
# edge ::= NUMBER ('--' | '->') NUMBER attributes? ;
#
# attributes ::= [ attributelist? ]
#
# attributelist ::= attributelist , attribute
#               |   attribute
#
# attribute ::= ID '=' STRING
#

reserved = {
    'strict': 'STRICT',
    'graph': 'GRAPH',
    'digraph': 'DIGRAPH',
}

# List of token names.   This is always required
tokens = [
     'NUMBER',
     'ID',
     'STRING',
     'DIRECTED',
     'UNDIRECTED',
     # 'LPAREN',       # (
     # 'RPAREN',       # )
     # 'LBRACKET',     # [
     # 'RBRACKET',     # ]
     # 'LBRACES',      # {
     # 'RBRACES',      # }
     # 'SEMICOLON',    # ;
     # 'LANGLE',       # <
     # 'RANGLE',       # >
     # 'MINUS',        # -
     # 'EQUAL',        # =
 ] + list(reserved.values())

# Regular expression rules for simple tokens
# t_LPAREN = r'\('
# t_RPAREN = r'\)'
# t_LBRACKET = r'\['
# t_RBRACKET = r'\]'
# t_LBRACES = '{'
# t_RBRACES = '}'
# t_SEMICOLON = r';'
# t_LANGLE = r'<'
# t_RANGLE = r'>'
# t_MINUS = r'-'
# t_EQUAL = r'='

literals = "()[]{}=;,"

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

# ignore line comments starting with #
t_ignore_COMMENT = r'\#.*'


# A regular expression rule with some action code
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t


def t_ID(t):
    r'\w+'
    t.type = reserved.get(t.value, 'ID')  # Check for reserved words
    return t


def t_STRING(t):
    r'"[^"]*" | \'[^\']*\''
    t.value = t.value[1:-1]
    return t


def t_DIRECTED(t):
    r'->'
    return t


def t_UNDIRECTED(t):
    r'--'
    return t


# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


def t_eof(t):
    print("Done")
    return None


lexer = lex.lex()


# ---------------------------------------------------------------------------

def p_graph(p):
    """graph : strict graphtype ID '{' stmtlist  '}'"""
    print(p[2], p[3])


def p_optstrict(p):
    """strict : STRICT
              | empty"""
    pass


def p_graphtype(p):
    """graphtype : GRAPH
                 | DIGRAPH"""
    print("graph type:", p[1])
    pass


def p_stmtlist(p):
    """stmtlist : stmt ';' stmtlist
                | stmt
                | empty"""
    pass


def p_stmt(p):
    """stmt : vertex
            | edge"""
    pass


def p_vertex(p):
    'vertex : NUMBER attributes'
    print("v", p[1])
    pass


# def p_edge(p):
#     'edge : NUMBER edgetype NUMBER'
#     print("e", p[1], p[2], p[3])
#
#
# def p_edgetype(p):
#     """edgetype : DIRECTED
#                 | UNDIRECTED"""
#     p[0] = p[1]

def p_edge(p):
    """
    edge : NUMBER DIRECTED NUMBER
         | NUMBER UNDIRECTED NUMBER
    """
    print("e", p[1], p[2], p[3])


def p_attributes(p):
    """attributes : '[' attributelist ']'
                     | empty"""
    pass


def p_attributelist(p):
    """attributelist : attribute ',' attributes
                     | attribute
                     | empty"""
    pass


def p_attribute(p):
    """attribute : ID '=' STRING"""
    pass


def p_error(p):
    print("Syntax error in input!", p, p.lexer.lineno)


def p_empty(p):
    'empty :'
    pass


parser = yacc.yacc()

# ---------------------------------------------------------------------------

def dump(n=None):
    i = 0
    while n is None or i < n:
        i += 1
        tok = lexer.token()
        if not tok:
            break  # No more input
        print(tok)


def read_file(file):
    with open(file, 'r') as file:
        return file.read()


data = read_file("../dependency-tomcat.dot")

# lexer.input(data)
# i = 0
# for tok in lexer:
#     print(tok)
#     i += 1
#     if i > 10: break


parser.parse(input=data)
