"""
Small helper for loading the numbered pipeline scripts (01_clean_data.py,
02_source_zones.py, 03_eda.py) as importable modules from other scripts in
this directory. Needed because module names can't start with a digit, so a
plain `import 01_clean_data` is not valid Python.

Usage:
    from _modutil import load_script
    clean = load_script("01_clean_data")
    tmd = clean.load_tmd()
"""
import importlib.util
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_script(stem):
    """Load scripts/<stem>.py as a module and return it. Executing the
    module only defines its functions/constants - each script's own work
    only runs under its own `if __name__ == "__main__":` guard, which is
    not triggered here."""
    path = os.path.join(SCRIPTS_DIR, f"{stem}.py")
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
