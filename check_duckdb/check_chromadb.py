import chromadb


def main():
    cdb = chromadb.PersistentClient(path="test.chroma")
    
    print(".")


if __name__ == "__main__":
    main()
