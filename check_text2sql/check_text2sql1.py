from text2sql import Text2SQL

sql = Text2SQL(model = "gpt-3.5-turbo-0125")
query = sql.query("How much do we have in total sales?")
print(query)

