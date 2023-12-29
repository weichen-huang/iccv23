import time
import sys

def log(x, mode="INFO"):
    print("[%s] %s: %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), mode, x))