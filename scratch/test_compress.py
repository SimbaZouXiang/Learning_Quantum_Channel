import quimb.tensor as qtn
tn = qtn.MPS_rand_state(10, 10).H & qtn.MPS_rand_state(10, 10)
try:
    c = qtn.tensor_network_1d_compress(tn, method="direct", split_opts={'method': 'qr'})
    print("worked with kwarg split_opts")
except Exception as e:
    print("split_opts error:", e)

try:
    c = qtn.tensor_network_1d_compress(tn, method="direct", compress_opts={'method': 'qr'})
    print("worked with compress_opts")
except Exception as e:
    print("compress_opts error:", e)

try:
    c = qtn.tensor_network_1d_compress(tn, method="direct", method2='qr')
    print("worked with method2=qr")
except Exception as e:
    print("method2 error:", e)
