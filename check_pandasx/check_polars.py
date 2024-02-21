import polars as pl
df= pl.DataFrame({'Patient':['Anna','Be','Charlie','Duke','Earth','Faux','Goal','Him'],
                  'Weight':[41,56,78,55,80,84,36,91],
                  'Segment':[1,2,1,1,3,2,1,1] })
print(df)
