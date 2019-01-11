import datetime
import time


def timer(func):
    """Time the execution time of a function

    Arguments:
        func {function} -- The function to wrap

    Returns:
        None
    """
    def timer_wraper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        runtime = (end - start)
        #Minutes, seconds, hours, minutes
        m, s = divmod(runtime, 60)
        h, m = divmod(m, 60)
        print("    Execution time: %d:%02d:%02d (H:MM:SS)" % (h, m, s))
    return timer_wraper
