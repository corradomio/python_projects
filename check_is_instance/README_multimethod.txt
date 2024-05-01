
@multimethod
def func(x: T1, y: T2):
    ...


T supported by issubclass(...)

    Union[...] or ... | ...
    Mapping[...]    - the first key-value pair is checked
    tuple[...]      - all args are checked
    Iterable[...]   - the first arg is checked
    Type[...]
    Literal[...]
    Callable[[...], ...] - parameter types are contravariant, return type is covariant
