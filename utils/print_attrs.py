def print_attrs(my_instance):
    attrs = [attr for attr in dir(my_instance) if not attr.startswith('__') and not callable(getattr(my_instance, attr))]
    for attr in attrs:
        if hasattr(my_instance, attr):
            print(f"{attr} = {getattr(my_instance, attr)}\n")