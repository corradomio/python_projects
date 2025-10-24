The target can be a Series or a DataFrame
The problem is when there is ONLY ONE column.
Rule:

    target      -> Series
    [target]    -> DataFrame of a single column
