import sys
import datetime
sys.stdout = open("stdout_redirect.log", mode="a", encoding="utf-8")

print(datetime.datetime.now())
print("Ciccio Pasticcio")





sys.stdout.close()