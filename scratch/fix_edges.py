import re

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

def replace_thesitetag(match):
    before = match.group(1)
    return before + r"""
                for j, tensor in enumerate(M):
                    if j == 0:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
                        M[j].modify(data=new_data, inds=(f"Boundary_L{i}_S{j}", tensor.inds[0], tensor.inds[1]))
                    elif j == M.L - 1:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
                        M[j].modify(data=new_data, inds=(tensor.inds[0], f"Boundary_L{i}_S{j}", tensor.inds[1]))"""

pattern = r'(thesitetag\s*=\s*f"Boundary_L\{i\}"\s*\n\s*for j,\s*tensor in enumerate\(M\):\s*\n\s*if j == 0:\s*\n\s*new_data = tensor\.copy\(\)\s*\n\s*new_data = tensor\.data\.reshape\(1,\s*tensor\.data\.shape\[0\],\s*tensor\.data\.shape\[1\]\)\s*\n\s*M\[j\]\.modify\(data=new_data,\s*inds=\(thesitetag,\s*tensor\.inds\[0\],\s*tensor\.inds\[1\]\)\)\s*\n\s*elif j == M\.L - 1:\s*\n\s*new_data = tensor\.copy\(\)\s*\n\s*new_data = tensor\.data\.reshape\(tensor\.data\.shape\[0\],\s*1,\s*tensor\.data\.shape\[1\]\)\s*\n\s*M\[j\]\.modify\(data=new_data,\s*inds=\(tensor\.inds\[0\],\s*thesitetag,\s*tensor\.inds\[1\]\)\))'

replaced = re.sub(pattern, lambda m: r"""
                for j, tensor in enumerate(M):
                    if j == 0:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
                        M[j].modify(data=new_data, inds=(f"Boundary_L{i}_S{j}", tensor.inds[0], tensor.inds[1]))
                    elif j == M.L - 1:
                        new_data = tensor.copy()
                        new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
                        M[j].modify(data=new_data, inds=(tensor.inds[0], f"Boundary_L{i}_S{j}", tensor.inds[1]))""", text)


with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(replaced)
