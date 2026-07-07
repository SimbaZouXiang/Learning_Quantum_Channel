with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

# I will replace the blanket try / except with exception printing to find out why it's breaking!
pattern = """                truncation_error += 1.0 - fidelity
            except:
                print("Compression failed at layer", i, "with max_bd =", max_bd, "and max_err =", max_err)"""
                
repl = """                truncation_error += 1.0 - fidelity
            except Exception as e:
                import traceback; traceback.print_exc()
                print("Compression failed at layer", i, "with max_bd =", max_bd, "and max_err =", max_err)"""

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text.replace(pattern, repl))
