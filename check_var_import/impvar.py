GLOBAL_VAR = None

def init_global_var():
    global GLOBAL_VAR
    GLOBAL_VAR = 123

def print_global_var():
    print("impvar:", GLOBAL_VAR)


def get_global_var():
    return GLOBAL_VAR
