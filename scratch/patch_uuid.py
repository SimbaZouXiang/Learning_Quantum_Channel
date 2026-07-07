with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

import re

# Match the basic block
pattern = r"""\s+thesitetag = qtn.rand_uuid\(\)\s+for i, tensor in enumerate\(M\):\s+if i == 0:\s+new_data = tensor.copy\(\)\s+new_data = tensor.data.reshape\(1, tensor.data.shape\[0\], tensor.data.shape\[1\]\)\s+M\[i\].modify\(data=new_data, inds=\(thesitetag, tensor.inds\[0\], tensor.inds\[1\]\)\)\s+elif i == M.L - 1:\s+new_data = tensor.copy\(\)\s+new_data = tensor.data.reshape\(tensor.data.shape\[0\], 1, tensor.data.shape\[1\]\)\s+M\[i\].modify\(data=new_data, inds=\(tensor.inds\[0\], thesitetag, tensor.inds\[1\]\)\)"""

count = [0]
def repl(match):
    idx = count[0]
    count[0] += 1
    # Grab the indentation from the match
    indent = match.group(0)[:match.group(0).find("thesitetag")]
    return f"""{indent}thesitetag = f"Boundary_UUID_{{idx}}"
{indent}for j, tensor in enumerate(M):
{indent}    if j == 0:
{indent}        new_data = tensor.copy()
{indent}        new_data = tensor.data.reshape(1, tensor.data.shape[0], tensor.data.shape[1])
{indent}        M[j].modify(data=new_data, inds=(thesitetag, tensor.inds[0], tensor.inds[1]))
{indent}    elif j == M.L - 1:
{indent}        new_data = tensor.copy()
{indent}        new_data = tensor.data.reshape(tensor.data.shape[0], 1, tensor.data.shape[1])
{indent}        M[j].modify(data=new_data, inds=(tensor.inds[0], thesitetag, tensor.inds[1]))"""

new_text = re.sub(pattern, repl, text)

# Is there any other qtn.rand_uuid() left?
print(f"Remaining rand_uuids: {new_text.count('qtn.rand_uuid()')}")

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(new_text)

