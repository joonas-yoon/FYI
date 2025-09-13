
import os


def Path(*args):
    return os.path.abspath(os.path.join(*args))


def humanize_seconds(seconds: float) -> str:
    """Convert seconds to mm:ss format string"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes == 0:
        return f"{secs:.2f} secs"
    return f"{minutes} min {secs:02.2f} secs"
