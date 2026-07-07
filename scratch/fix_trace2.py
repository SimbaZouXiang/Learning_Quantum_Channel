with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

pattern = """                    target_mps_list.append(MPS_target)
                except Exception:
                    print(f"Failed to process MPS {counter+1} due to truncation, skipping...")"""
                
repl = """                    target_mps_list.append(MPS_target)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"Failed to process MPS {counter+1} due to truncation, skipping...")"""

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text.replace(pattern, repl))
