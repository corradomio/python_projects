import pandas as pd


df = pd.read_csv("/Projects.github/python_projects/check_pyspark/allCountriesCSV.csv",
                 header=0,
                 dtype={
                    "COUNTRY": str,
                    "POSTAL_CODE": str,
                    "CITY": str,
                    "STATE": str,
                    "SHORT_STATE": str,
                    "COUNTY": str,
                    "SHORT_COUNTY": str,
                    "COMMUNITY": str,
                    "SHORT_COMMUNITY": str,
                    "LATITUDE": str,
                    "LONGITUDE": str,
                    "ACCURACY": str
                 }, na_values=[""])

print(df.head())
