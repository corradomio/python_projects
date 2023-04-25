import traceback


def main():
    try:
        raise Exception("ciccio pasticcio", 2, 3)
    except Exception as e:
        # fl = traceback.format_list()
        # etb = traceback.extract_tb()
        # es = traceback.extract_stack()
        # eo = traceback.format_exception_only(e)
        # ex = traceback.format_exception(e)
        exc = traceback.format_exc()
        te = type(e)
        exc_class = f"{te.__module__}.{te.__name__}{e}"
        print(exc)
    pass


if __name__ == "__main__":
    main()
