from boolgen import  *

input_data = \
"""A B C D=
   0 0 0 0
   0 0 1 0
   0 1 0 0
   0 1 1 0
   1 0 0 0
   1 0 1 0
   1 1 0 0
   1 1 1 0"""
input_vars, output_vars, truth_table = parse_input(input_data)
expression = simplify_expression(input_vars, truth_table, 0)
expected_expression = "0"