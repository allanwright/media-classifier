'''Diagnostics module.

'''

import time
from functools import wraps

def stopwatch(func):
    '''Times the execution of the specified function.

    Args:
        func (function): The function to time.
    '''
    @wraps(func)
    def wrapper(*arg, **kw):
        start = time.time()
        result = func(*arg, **kw)
        stop = time.time()
        print('{func} executed in {elapsed:.4f}ms.'.format(
            func=func.__name__,
            elapsed=(stop - start) * 1000))
        return result
    return wrapper
