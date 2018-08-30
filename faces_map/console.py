RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

BRIGHT_RED = '\033[91m'
BRIGHT_GREEN = '\033[92m'
BRIGHT_YELLOW = '\033[93m'
BRIGHT_BLUE = '\033[94m'
BRIGHT_MAGENTA = '\033[95m'
BRIGHT_CYAN = '\033[96m'
BRIGHT_WHITE = '\033[97m'

BOLD = '\033[1m'
ENDC = '\033[0m'

def print_color(string, color, bold=False):
    """
    Formats the string with colors for terminal prints
    """
    if bold is True:
        print(BOLD + color + string + ENDC)
    else:
        print(color + string + ENDC)
