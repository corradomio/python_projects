import psycopg2


def main():
    # Connect to your postgres DB
    conn = psycopg2.connect(database='OpenBeer', user='root', password='playwithdata', host='172.23.86.3', port=5432)

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Execute a query
    cur.execute("SELECT * FROM Beer LIMIT 10")

    # Retrieve query results
    records = cur.fetchall()
    print(records)


if __name__ == "__main__":
    main()
