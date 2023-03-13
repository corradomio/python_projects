import sys
sys.path.insert(0, "../..")

import ply.lex as lex
import ply.yacc as yacc
import os


class DOTImporter:

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

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = {}
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[1] + "_" + self.__class__.__name__
        except:
            modname = "parser" + "_" + self.__class__.__name__
        self.debugfile = modname + ".dbg"
        # print self.debugfile

        # Build the lexer and parser
        self.lexer = lex.lex(module=self, debug=self.debug)
        self.parser = yacc.yacc(module=self,
                                debug=self.debug,
                                debugfile=self.debugfile)

        self.attributes = dict()

    def parse(self, s):
        # self.parser.parse(s)
        yacc.parse(s)

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
    ] + list(reserved.values())

    literals = "()[]{}=;,"

    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # ignore line comments starting with #
    t_ignore_COMMENT = r'\#.*'

    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_ID(self, t):
        r'\w+'
        t.type = self.reserved.get(t.value, 'ID')  # Check for reserved words
        return t

    def t_STRING(self, t):
        r'"[^"]*" | \'[^\']*\''
        t.value = t.value[1:-1]
        return t

    def t_DIRECTED(self, t):
        r'->'
        return t

    def t_UNDIRECTED(self, t):
        r'--'
        return t

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # Error handling rule
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def t_eof(self, t):
        print("Done")
        return None

    # ---------------------------------------------------------------------------

    def p_graph(self, p):
        """graph : strict graphtype ID '{' stmtlist  '}'"""
        print(p[2], p[3])

    def p_strict(self, p):
        """strict : STRICT
                  | empty"""
        pass

    def p_graphtype(self, p):
        """graphtype : GRAPH
                     | DIGRAPH"""
        print("graph type:", p[1])
        pass

    def p_stmtlist(self, p):
        """stmtlist : stmt ';' stmtlist
                    | stmt
                    | empty"""
        pass

    def p_stmt(self, p):
        """stmt : vertex
                | edge"""
        pass

    def p_vertex(self, p):
        'vertex : NUMBER optattributes'
        self.on_vertex(p[1], p[2])
        pass

    def p_edge(self, p):
        """
        edge : NUMBER DIRECTED NUMBER
             | NUMBER UNDIRECTED NUMBER
        """
        self.on_edge(p[1], p[3], p[2] == "->")
        pass

    def p_optattributes(self, p):
        """optattributes : '[' attributelist ']'
                         | empty"""
        if len(p) == 4:
            p[0] = p[2]
        pass

    def p_attributelist(self, p):
        """attributelist : attribute ',' attributelist
                         | attribute
                         | empty"""
        if len(p) == 1:
            return
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = dict()
            p[0].update(p[1])
            p[0].update(p[3])
        pass

    def p_attribute(self, p):
        """attribute : ID '=' STRING"""
        key = p[1]
        value = p[3]
        p[0] = {key: value}
        pass

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")

    # -----------------------------------------------------------------------

    def on_vertex(self, v, attributes):
        print("v", v, attributes)

    def on_edge(self, v1, v2, directed):
        dir = "->" if directed else "--"
        print("e", v1, dir, v2)

# end
