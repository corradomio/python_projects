import lancedb

def main():
    db = lancedb.connect("db.lancedb")

    data = [
        {"id": "1", "text": "knight", "vector": [0.9, 0.4, 0.8]},
        {"id": "2", "text": "ranger", "vector": [0.8, 0.4, 0.7]},
        {"id": "9", "text": "priest", "vector": [0.6, 0.2, 0.6]},
        {"id": "4", "text": "rogue", "vector": [0.7, 0.4, 0.7]},
    ]
    table = db.create_table("adventurers", data=data, mode="overwrite")
    query_vector = [0.8, 0.3, 0.8]
    
    # Ensure you run `pip install polars` beforehand
    result = table.search(query_vector).limit(2).to_polars()
    print(result)



if __name__ == "__main__":
    main()

