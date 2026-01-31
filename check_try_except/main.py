from logging import raiseExceptions


def main():
    try:
        print("Hello World")
        print("Raise RuntimeError")
        raise RuntimeError("Not Implemented")
    except RuntimeError as ex:
        print("RuntimeError", ex)
    except Exception as ex:
        print("Exception", ex)
    else:
        # executed if NO exception
        print("Success")
    finally:
        # executed ALWAYS
        print("Finished")




if __name__ == "__main__":
    main()
