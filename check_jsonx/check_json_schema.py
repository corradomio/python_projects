from stdlib import jsonx


def main():
    schema = jsonx.load("schema_records.json")
    records = jsonx.load("data_records.json")
    n_records = len(records)

    records = jsonx.validate(records, [schema])
    pass


if __name__ == "__main__":
    main()