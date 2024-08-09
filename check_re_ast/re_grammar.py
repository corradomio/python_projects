from random import random, randint
from ply.lex import lex
from ply.yacc import yacc


RMAX = 10


class RegExp:
    def __init__(self):
        pass

    def __repr__(self):
        ...

    def random(self) -> str:
        ...


class OrExp(RegExp):
    def __init__(self, lexp, rexp):
        super().__init__()
        self.expl = []
        if isinstance(lexp, OrExp) and isinstance(rexp, OrExp):
            self.expl = lexp.expl + rexp.expl
        elif isinstance(lexp, OrExp):
            self.expl = lexp.expl + [rexp]
        elif isinstance(rexp, OrExp):
            self.expl = [lexp] + rexp.expl
        else:
            self.expl = [lexp, rexp]
        # self.lexp: RegExp = lexp
        # self.rexp: RegExp = rexp

    def __repr__(self):
        if len(self.expl) == 2:
            return f"{self.expl[0]}|{self.expl[1]}"
        else:
            return "|".join(map(str, self.expl))

    def random(self):
        n = len(self.expl)
        i = randint(0, n-1)
        return self.expl[i].random()


class AndExp(RegExp):
    def __init__(self, lexp, rexp):
        super().__init__()
        if isinstance(lexp, AndExp) and isinstance(rexp, AndExp):
            self.expl = lexp.expl + rexp.expl
        elif isinstance(lexp, AndExp):
            self.expl = lexp.expl + [rexp]
        elif isinstance(rexp, AndExp):
            self.expl = [lexp] + rexp.expl
        else:
            self.expl = [lexp, rexp]

    def __repr__(self):
        if len(self.expl) == 2:
            return f"{self.expl[0]}{self.expl[1]}"
        else:
            return "".join(map(str, self.expl))

    def random(self):
        srnd = ""
        for exp in self.expl:
            srnd += exp.random()
        return srnd


class RangeExp(RegExp):
    def __init__(self, exp, erange):
        super().__init__()
        rmin, rmax = erange
        self.exp: RegExp = exp
        self.rmin = rmin
        self.rmax = rmax

    def __repr__(self):
        if self.rmin == 1 and self.rmax == 1:
            return f"{self.exp}"
        if self.rmin == 0 and self.rmax == 1:
            return f"{self.exp}?"
        if self.rmin == 0 and self.rmax == -1:
            return f"{self.exp}*"
        if self.rmin == 1 and self.rmax == -1:
            return f"{self.exp}+"
        if self.rmin == self.rmax:
            return f"{self.exp}{{{self.rmin}}}"
        if self.rmin == 0:
            return f"{self.exp}{{,{self.rmax}}}"
        if self.rmax == -1:
            return f"{self.exp}{{{self.rmin},}}"
        else:
            return f"{self.exp}{{{self.rmin},{self.rmax}}}"

    def random(self):
        rmin = self.rmin
        rmax = self.rmax if self.rmax >= 0 else RMAX
        n = randint(rmin, rmax)
        srnd = ""
        for i in range(n):
            srnd += self.exp.random()
        return srnd


class LitExp(RegExp):
    def __init__(self, lit):
        super().__init__()
        self.lit: str = lit

    def __repr__(self):
        return self.lit

    def random(self):
        return self.lit


class ClassExp(RegExp):
    def __init__(self, chars):
        super().__init__()
        self.chars: str = chars

    def __repr__(self):
        return f"[{self.chars}]"

    def random(self):
        if len(self.chars) == 0:
            return ""
        i = randint(0, len(self.chars) - 1)
        return self.chars[i]


class GroupExp(RegExp):
    def __init__(self, exp):
        super().__init__()
        self.exp: RegExp = exp

    def __repr__(self):
        return f"({self.exp})"

    def random(self):
        return self.exp.random()


# --- Tokenizer
# RE
#   . [...] [^...]
#   ab  a|b  (a)  a?  a*  a+
#   a{min,max}  a{,max}  a{min,}

#
# RE special chars:     . + * ? ^ $ ( ) [ ] { } | \
# Escape sequence:      \<special char>
#
# Qualifiers:           a? | a* | a+ | a{m,n} | a{n} | a{,n} | a{m,}
#   some equivalences   a?      == a{0,1}
#                       a*      == a{0,N}
#                       a+      == a{1,N}
#                       a{n}    == a{n,n}
#                       a{,n}   == a{0,n}
#                       a{m,}   == a{m,N}
#
# Grouping              (a)
#
# Char class:           [...] | [^...]
# Chars in class        c1c2... | b-e
#
# Note: in a class, the set of special chars change, and changes also the set of special chars
#       in first/last position. For example, '-' is not special if it is the first or the last
#       char in the class
#


tokens = ('OR', 'OPTIONAL', 'STAR', 'PLUS', 'MINUS', 'COMMA',
          'LPAREN', 'RPAREN', 'LBRACKET', 'RBRACKET', 'LSQUARE', 'RSQUARE',
          'LITERAL', 'DIGIT', 'DOT', 'INT')

# Ignored characters
t_ignore = ' \t'

# Token matching rules are written as regular expressions
t_OR = r'\|'
t_OPTIONAL = r'\?'
t_STAR = r'\*'
t_PLUS = r'\+'
t_MINUS = r'-'

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACKET = r'\{'
t_RBRACKET = r'\}'

t_LSQUARE = r'\['
t_RSQUARE = r'\]'

t_COMMA = r','
t_LITERAL = r'[^.{}()?+*[\]|0-9,-]'
t_DIGIT = r'[0-9]'
t_DOT = r'\.'

# t_INT = r'[0-9]+'


# Error handler for illegal characters
def t_error(t):
    print(f'Illegal character {t.value[0]!r}')
    t.lexer.skip(1)


# Build the lexer object
lexer = lex()


# --- Parser

def p_expression(p):
    """
    expression : expression OR exp
               | exp
    """
    if len(p) == 4:
        p[0] = OrExp(p[1], p[3])
    else:
        p[0] = p[1]


def p_exp(p):
    """
    exp : exp term
        | term
    """
    if len(p) == 3:
        p[0] = AndExp(p[1], p[2])
    else:
        p[0] = p[1]


def p_term(p):
    """
    term : fact qualifier
    """
    p[0] = RangeExp(p[1], p[2])


def p_qualifier(p):
    """
    qualifier : OPTIONAL
              | STAR
              | PLUS
              | LBRACKET range RBRACKET
              |
    """
    if len(p) == 1:
        p[0] = [1, 1]
    elif p[1] == '?':
        p[0] = [0, 1]
    elif p[1] == '*':
        p[0] = [0, -1]
    elif p[1] == '+':
        p[0] = [1, -1]
    else:
        p[0] = p[2]


def p_range(p):
    """
    range : int COMMA int
          | COMMA int
          | int COMMA
          | int
    """
    if len(p) == 4:
        p[0] = [p[1], p[3]]
    elif len(p) == 3:
        if p[1] == ',':
            p[0] = [0, p[2]]
        else:
            p[0] = [p[1], -1]
    else:
        p[0] = [p[1], p[1]]


def p_int(p):
    """
    int : int DIGIT
        | DIGIT
    """
    if len(p) == 3:
        p[0] = p[1]*10 + int(p[2])
    else:
        p[0] = int(p[1])


def p_fact(p):
    """
    fact : LITERAL
         | DIGIT
         | COMMA
         | DOT
         | MINUS
         | LSQUARE chars RSQUARE
         | LPAREN expression RPAREN
    """
    if len(p) == 2:
        p[0] = LitExp(p[1])
    elif p[1] == '[':
        p[0] = ClassExp(p[2])
    else:
        p[0] = GroupExp(p[2])


def p_chars(p):
    """
    chars : chars char MINUS char
          | chars char
          |
    """
    if len(p) == 1:
        p[0] = ""
    elif len(p) == 3:
        p[0] = p[1] + p[2]
    else:
        cclass = p[1]
        cstart = ord(p[2])
        cend = ord(p[4])
        for c in range(cstart, cend+1):
            cclass += chr(c)
        p[0] = cclass


def p_char(p):
    # t_LITERAL = r'[^.{}()?+*[\]|0-9,]'
    """
    char : LITERAL
         | DIGIT
         | DOT
         | LBRACKET
         | RBRACKET
         | LPAREN
         | RPAREN
         | OPTIONAL
         | STAR
         | PLUS
    """
    p[0] = p[1]



def p_error(p):
    print("Error in parsing: ", p)


parser = yacc()
