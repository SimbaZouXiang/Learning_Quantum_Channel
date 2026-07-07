with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "except Exception:" in line and "return idx, None, 0.0, True" in lines[i+1]:
        lines[i] = line.replace("except Exception:", "except Exception as e: import traceback; traceback.print_exc();")

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.writelines(lines)
