import quimb.tensor as qtn
tn = qtn.MPS_rand_state(10, 10).H & qtn.MPS_rand_state(10, 10)
c1 = qtn.tensor_network_1d_compress(tn, method="direct")
c2 = qtn.tensor_network_1d_compress(tn, method="direct", compress_opts={'method': 'qr'})
print("c1 norm:", c1.norm())
print("c2 norm:", c2.norm())
