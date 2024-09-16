from re_learn import re_parse, re_infer


def re_samples(re_ast, n):
    samples = []
    for i in range(n):
        samples.append(re_ast.random())
    return samples


def load_templates():
    templates = []
    with open("re_templates.txt") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            templates.append(line)
    return templates


def main():
    templates = load_templates()

    for re in templates:
        print("# --------------------------")
        # print("-", re)
        re_ast = re_parse(re)
        print("#", re_ast)
        print("# --------------------------")

        samples = re_samples(re_ast, 20)

        # for s in samples:
        #     print(s)

        inferred = re_infer(samples)
        for i in range(4):
            rei = inferred[i]
            print(f're: {rei[0]}: {rei[1]}')
    # end

    print("# --------------------------")
    print("# - end")
    print("# --------------------------")
    return


if __name__ == "__main__":
    main()
