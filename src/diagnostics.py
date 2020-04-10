'''Diagnostics module.

'''

import time
from functools import wraps

def timer(func):
    '''Times the execution of a function.

    Args:
        func: (callable): The function to execute.
    '''
    @wraps(func)
    def wrapper(*arg, **kw):
        result, exec_time = _time_func(func, *arg, **kw)
        print('{module}.{func}() executed in {time:,.2f}s.'.format(
            module=func.__module__,
            func=func.__name__,
            time=exec_time))
        return result
    return wrapper

def multi_timer(count: int = 1):
    '''Times the execution of a function.

    Args:
        func: (callable): The function to execute.
        count (int): The number of times to execute the function.
    '''
    if count == 0:
        raise ValueError(count)

    def decorator(func):
        @wraps(func)
        def wrapper(*arg, **kw):
            times = []
            for _ in range(count):
                result, exec_time = _time_func(func, *arg, **kw)
                times.append(exec_time)
            print('{module}.{func}() executed {count} times ' \
                  '[min={min:,.2f}s, max={max:,.2f}s, avg={avg:,.2f}s].'.format(
                      module=func.__module__,
                      func=func.__name__,
                      count=count,
                      min=min(times),
                      max=max(times),
                      avg=sum(times) / len(times)))
            return result
        return wrapper
    return decorator

def _time_func(func, *arg, **kw):
    start = time.time()
    result = func(*arg, **kw)
    stop = time.time()
    return (result, stop - start)
