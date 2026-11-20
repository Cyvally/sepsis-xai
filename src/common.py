"""Small helpers shared by the step files."""
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def took(t):
    s = time.time() - t
    return f"{s:.0f}s" if s < 90 else f"{s/60:.1f} min"
