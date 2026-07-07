with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

text = text.replace('        except Exception:\n            return idx, None, None, True          # (index, None, None, skipped)',
                    '        except Exception as e:\n            import traceback; traceback.print_exc(file=sys.stdout); return idx, None, None, True          # (index, None, None, skipped)')

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text)
