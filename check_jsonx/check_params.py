from stdlib.jsonx import load, get

def main():
    jdata = load("parametrized.json", ciccio="pasticcio", uno=1)
    print(jdata)

    print(get(jdata, "root", "child", 1))
    print(get(jdata, "root", "child1", 1))
    pass



if __name__ == "__main__":
    main()
