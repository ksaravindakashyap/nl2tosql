import runpy
import sys
import traceback

try:
    runpy.run_path('main.py', run_name='__main__')
except Exception:
    traceback.print_exc()
    sys.exit(1)
