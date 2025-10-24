from stdlib.jsonx import load

def main():
    jdata = load("parametrized.json", ciccio="pasticcio", uno=1)
    print(jdata)
    pass



if __name__ == "__main__":
    main()
