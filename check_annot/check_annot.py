

def annot(f, *args, **kwargs):
    print("annot")
    def _wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return _wrapper
    return f(*args, **kwargs)


@annot
def fun():
    print("fun")

def main():
    fun()

if __name__ == "__main__":
    main()
