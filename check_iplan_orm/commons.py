PLAN_NAME = 'CM Plan Food Import'
DATA_MODEL = "CM Data Model Food Import"
DATA_MASTER = "CM Data Master Food Import"
AREA_HIERARCHY = "ImportCountries"
SKILL_HIERARCHY = "ImportFoods"
TIME_SERIES = 'CM Time Series Food Import'

TS_TARGETS = 'import_kg'
TS_INPUTS = ["crude_oil_price", "sandp_500_us",
             "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan", "max_temperature",
             "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"]
