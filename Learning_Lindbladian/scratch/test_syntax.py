import py_compile
import sys

try:
    py_compile.compile('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', doraise=True)
except Exception as e:
    sys.exit(1)
