with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

# Revert Thesitetag
text = text.replace('thesitetag = f"Boundary_UUID_1"', 'thesitetag = qtn.rand_uuid(11)')

# Revert my bad exception print
text = text.replace('import traceback; traceback.print_exc(file=sys.stdout); return idx, None, None, True          # (index, None, None, skipped)',
                    'return idx, None, None, True')

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text)
