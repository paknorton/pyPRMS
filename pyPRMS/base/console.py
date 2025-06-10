from rich.console import Console
# from rich import inspect


class ConsoleManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = Console(*args, **kwargs)

            if len(kwargs) == 0:
                # Provide some defaults when no arguments are passed
                cls._instance.is_jupyter = False
                cls._instance.width = 120

        # cls._instance.print("ConsoleManager initialized with console: {}".format(cls._instance))
        return cls._instance


def get_console_instance(*args, **kwargs):
    """
    Function to get the console instance.
    """
    # print(f'{len(kwargs)=}, {kwargs=}')
    # con = ConsoleManager(*args, **kwargs)
    # inspect(con, methods=False, private=True)
    # return con
    return ConsoleManager(*args, **kwargs)
