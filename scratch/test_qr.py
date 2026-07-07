import quimb.tensor as qtn
tn = qtn.MPS_rand_state(10, 10).H & qtn.MPS_rand_state(10, 10)
try:
    c = qtn.tensor_network_1d_compress(tn, method="direct", compress_opts={'method': 'qr'})
    print("qr worked with compress_opts")
except Exception as e:
    print(e)
try:
    c = qtn.tensor_network_1d_compress(tn, method="direct", split_opts={'method': 'qr'})
    print("qr worked with split_opts")
except Exception as e:
    print(e)
