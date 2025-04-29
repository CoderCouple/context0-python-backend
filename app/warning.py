from pyfiglet import figlet_format
from termcolor import colored


def print_boxed_auth_warning():
    app_name = "C O N T E X T    A I"
    title = "AUTH DISABLED"
    subtitle = "Static context will be used for all requests."

    # Big header text
    header = figlet_format(app_name, font="small")

    # Colorize
    header_colored = colored(header, color="red", on_color="on_white", attrs=["bold"])
    subtitle_colored = colored(f"⚠️ {title} : {subtitle}", "red", attrs=["bold"])

    print(header_colored)
    print(subtitle_colored)
