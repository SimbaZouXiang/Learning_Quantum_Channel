import re

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

# Fix nested loops overwriting `i`
text = re.sub(
    r'(\s+)for\s+i,\s+tensor\s+in\s+enumerate\(M\):\s*\n(\s+)inds\s+=\s+list\(M\[i\]\.inds\)\s*\n(\s+)inds\[1\],\s+inds\[2\]\s+=\s+inds\[2\],\s+inds\[1\]\s*\n(\s+)M\[i\]\.transpose_\(\*inds\)',
    r'\1for j_inner, tensor in enumerate(M):\n\2inds = list(M[j_inner].inds)\n\3inds[1], inds[2] = inds[2], inds[1]\n\4M[j_inner].transpose_(*inds)',
    text
)

text = re.sub(
    r'thesitetag\s*=\s*f"Boundary_L\{layer_idx\}"',
    r'thesitetag = f"Boundary_L{i}"',
    text
)

with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
    f.write(text)
