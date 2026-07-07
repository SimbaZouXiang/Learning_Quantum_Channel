with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

text = text.replace('except Exception:\n                    print(f"Failed to process MPS',
                    'except Exception as e:\n                    print(f"THE REAL ERROR: {e}"); print(f"Failed to process MPS')

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text)
