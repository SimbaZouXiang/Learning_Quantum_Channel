with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

text = text.replace("except Exception as e: import traceback; traceback.print_exc();\n            return idx, None, 0.0, True",
                    "except Exception as e:\n            print('THE ERROR IS:', e, flush=True)\n            return idx, None, 0.0, True")

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text)
