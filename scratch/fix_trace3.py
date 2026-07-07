with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

pattern = """        except Exception:
            return idx, None, 0.0, True"""
            
repl = """        except Exception as e:
            import traceback; traceback.print_exc()
            return idx, None, 0.0, True"""

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text.replace(pattern, repl))

