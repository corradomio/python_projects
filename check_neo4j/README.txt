http://102.37.140.82:7474/browser/


"127ef6fc"	"Flink-test"
"4a95c5c1"	"TomCat"
"f92ff479"	"Camel"
"1bccefb2"	"Hibernate-orm"
"e2d0d731"	"Hibernate-reactive"
"11aa4e33"	"hibernate-searchs"


sources:
    match (n:source)
    where n.refId in ["127ef6fc", "4a95c5c1", "f92ff479", "1bccefb2", "e2d0d731", "11aa4e33"]
    return n.refId, count(n)

"4a95c5c1"	2588
"127ef6fc"	13392
"1bccefb2"	11947
"11aa4e33"	4784
"e2d0d731"	408
"f92ff479"	21422


type
    match (n:type)
    where n.refId in ["127ef6fc", "4a95c5c1", "f92ff479", "1bccefb2", "e2d0d731", "11aa4e33"]
      and n.type <> 'reftype'
    return n.refId, count(n)

"4a95c5c1"	4111
"127ef6fc"	21015
"1bccefb2"	17152
"e2d0d731"	789
"11aa4e33"	7377
"f92ff479"	27792


"612c68e3"	"acme-demo"
"d06d3d07"	"sample-2-ecom"
"3bf53312"	"Hive-test"
"5ae2b403"	"acme-test-5"
"5ceccc71"	"ACME-test-v5"
"127ef6fc"	"Flink-test"
"7fc57a50"	"strutc"
"4a95c5c1"	"TomCat"
"6b3c1ff9"	"Book-Keeper"
"1bccefb2"	"Hibernate-orm"
"e2d0d731"	"Hibernate-reactive"
"11aa4e33"	"hibernate-searchs"
"bbc6ae13"	"middle-storm-2"
"800d7a32"	"Middle-storm-1"
"f92ff479"	"Camel"
"10c2388f"	"acme-33-3"
"43923aa9"	"ACME-Beta-a4-spl33"
"5e74a434"	"ACME-Nov19"
"ff9db8a1"	"ACME-Beta-a4SPL33-Fix1"
"d5d5c8b6"	"acme-Beta4SPL33-Fix2"
"c86062f3"	"ACME-revisions-1"
"43c09382"	"ACME-Beta-a4-spl33-Fix3"
"5db7f4dc"	"ACME-BETA5"
"c40fca8"	"ACMEBETA5-v2"
"7d926777"	"ACME"
"16c6a31c"	"acme12345"
"e19fc0bd"	"acme-matric-test"
"467eb384"	"acme-test2-matrics"
