JAVA_KEYWORDS = set([
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'default',
    'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import',
    'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'return',
    'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'thorw', 'throws', 'transient', 'try',
    'void', 'volatile', 'while',

    'record', 'exports', 'module', 'non-sealed', 'open', 'opens', 'permits', 'provides', 'requires', 'sealed', 'to',
    'transitive', 'uses', 'var', 'with', 'yield',

    'true', 'false', 'null',

    'override', 'deprecated', 'supress', 'warning', 'not',

    # 'string', 'list', 'array', 'hash', 'map', 'iterator', 'iterable', 'exception',
])

PYTHON_KEYWORDS = set([
    'False', 'await', 'else', 'import', 'pass', 'None', 'break', 'except', 'in', 'raise', 'True', 'class', 'finally',
    'is', 'return', 'and', 'continue', 'for', 'lambda', 'try', 'as', 'def', 'from', 'nonlocal', 'while', 'assert',
    'del', 'global', 'not', 'with', 'async', 'elif', 'if', 'or', 'yield',
])

CSHARP_KEYWORDS = set([
    'abstract', 'as', 'base', 'bool', 'break', 'byte', 'case', 'catch', 'char', 'checked', 'class', 'const', 'continue',
    'decimal', 'default', 'delegate', 'do', 'double', 'else', 'enum', 'event', 'explicit', 'extern', 'false', 'finally',
    'fixed', 'float', 'for', 'foreach', 'goto', 'if', 'implicit', 'in', 'int', 'interface', 'internal', 'is', 'lock',
    'long', 'namespace', 'new', 'null', 'object', 'operator', 'out', 'override', 'params', 'private', 'protected',
    'public', 'readonly', 'ref', 'return', 'sbyte', 'sealed', 'short', 'sizeof', 'stackalloc', 'static', 'string',
    'struct', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'uint', 'ulong', 'unchecked', 'unsafe', 'ushort',
    'using', 'virtual', 'void', 'volatile', 'while',

    'add', 'and', 'alias', 'ascending', 'args', 'async', 'await', 'by', 'descending', 'dynamic', 'equals', 'file',
    'from', 'get', 'global', 'group', 'init', 'into', 'join', 'let', 'managed', 'nameof', 'nint', 'not', 'notnull',
    'nuint', 'on', 'or', 'orderby', 'partial', 'record', 'remove', 'required', 'scoped', 'select', 'set', 'unmanaged',
    'value', 'var', 'when', 'where', 'with', 'yield',
])

JAVASCRIPT_KEYWORDS = set([
    'abstract', 'arguments', 'await', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue',
    'debugger', 'default', 'delete', 'do', 'double', 'else', 'enum', 'eval', 'export', 'extends', 'false', 'final',
    'finally', 'float', 'for', 'function', 'goto', 'if', 'implements', 'import', 'in', 'instanceof', 'int', 'interface',
    'let', 'long', 'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'return', 'short', 'static',
    'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'typeof', 'var', 'void',
    'volatile', 'while', 'with', 'yield',
])

LANGUAGE_KEYWORDS = {
    'java': JAVA_KEYWORDS,
    'python': PYTHON_KEYWORDS,
    'c#': CSHARP_KEYWORDS,
    'csharp': CSHARP_KEYWORDS,
    'javascript': JAVASCRIPT_KEYWORDS,
    'js': JAVA_KEYWORDS
}