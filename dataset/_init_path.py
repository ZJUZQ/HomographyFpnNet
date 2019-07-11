import os, sys
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)