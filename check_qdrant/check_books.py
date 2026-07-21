from llama_index.core import SimpleDirectoryReader




def main():
    reader = SimpleDirectoryReader(
        input_dir=r"E:\Dropbox\Books\Mathematica",
        recursive=True,
        # required_exts=["*.pdf"]
    )
    reader.load_data(True, 16)



if __name__ == "__main__":
    main()
