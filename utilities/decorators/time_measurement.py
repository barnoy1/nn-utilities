import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        from utilities.log.logger import logger
        logger.info(f"{func.__name__} took {elapsed_time:.2f} seconds to run")
        return result

    return wrapper


# Example usage:

@timing_decorator
def example_function():
    # Your code here
    time.sleep(2)
    print("Function executed.")


# Call the decorated function
example_function()
