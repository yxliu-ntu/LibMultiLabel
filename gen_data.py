import os, sys
import math
import numpy as np
import scipy as sp

from scipy.sparse import diags, csr_matrix
from sklearn.datasets import dump_svmlight_file

np.random.seed(0)
r = 128
pn = 50
sample_ratio = 0.1
root = sys.argv[1]

#postfix = 'mf.lrank'
# Low-rank mf data
#with open('tr.%s'%postfix, 'w') as fo:
#    for i in range(r*pn):
#        d = i//pn
#        label = ','.join(["%d"%(pn*d+l) for l in range(pn)])
#        fo.write('%s\t%d:1\n'%(label, i+1))
#
#with open('item.%s'%postfix, 'w') as fo:
#    for i in range(r*pn):
#        fo.write('\t%d:1\n'%(i+1))

def gen_label_matrix(r, pn):
    rows, cols = [], []
    for i in range(r*pn):
        d = i//pn
        rows.extend([i]*pn)
        cols.extend([pn*d+l for l in range(pn)])
    return csr_matrix((np.ones(r*pn*pn), (rows, cols)), shape=(r*pn, r*pn))

def dump_svm(X, Y, f):
    if not hasattr(f, "write"):
        f = open(f, "w")
    assert hasattr(X, "tocsr")
    if Y is not None:
        assert hasattr(Y, "tocsr")
        assert Y.shape[0] == X.shape[0]
    rnum = X.shape[0]

    value_pattern = "%d:%d"
    label_pattern = "%d"
    line_pattern = "%s\t%s\n"

    for i in range(rnum):
        span = slice(X.indptr[i], X.indptr[i + 1])
        row = zip(X.indices[span], X.data[span])
        feat = " ".join(value_pattern % (j+1, x) for j, x in row)

        if Y is not None:
            yspan = slice(Y.indptr[i], Y.indptr[i + 1])
            col = Y.indices[yspan]
            col = sorted(col)
            label = ",".join(label_pattern % j for j in col)
        else:
            label = ""
        f.write(line_pattern % (label, feat))

M = 30000 #r*pn
N = 30000 #r*pn
U = diags(np.ones(M), format='csr')
V = diags(np.ones(N), format='csr')
#Y = gen_label_matrix(r, pn)
Y = sp.sparse.random(M, N, density=0.01, format='csr')
Y.data = np.ones_like(Y.data)
print(U.shape, V.shape, Y.shape, Y.data)

## MN
postfix = 'mf.lrank.mn'
M_hat = math.ceil(M*sample_ratio)
N_hat = math.ceil(N*sample_ratio)
u_ids = np.random.choice(M, size=M_hat, replace=True)
v_ids = np.random.choice(N, size=N_hat, replace=True)
U_hat = U[u_ids, :]
V_hat = V[v_ids, :]
Y_hat = Y[u_ids, :][:, v_ids]
print(Y_hat.data.min(), Y_hat.data.max())
print(U_hat.shape, V_hat.shape, Y_hat.shape)

upath = '%s/tr.%s'%(root, postfix)
vpath = '%s/item.%s'%(root, postfix)
dump_svm(U_hat, Y_hat, upath)
dump_svm(V_hat, None, vpath)

## Conventional
postfix = 'mf.lrank.cvt'
sample_num = M_hat*N_hat
#u_ids = np.random.choice(M, size=sample_num, replace=True)
#v_ids = np.random.choice(N, size=sample_num, replace=True)
ids = np.random.choice(M*N, size=sample_num, replace=True)
u_ids = ids // M
v_ids = ids % M
ys = Y[u_ids, v_ids].A.flatten()
uni_u_ids = np.unique(u_ids)
uni_v_ids = np.unique(v_ids)
u_id_map = {j:i for i,j in enumerate(uni_u_ids)}
v_id_map = {j:i for i,j in enumerate(uni_v_ids)}
nu_ids = [u_id_map[i] for i in u_ids]
nv_ids = [v_id_map[i] for i in v_ids]

U_hat = U[uni_u_ids, :]
V_hat = V[uni_v_ids, :]
Y_hat = csr_matrix((ys, (nu_ids, nv_ids)))
Y_hat.eliminate_zeros()
Y_hat.data = np.ones_like(Y_hat.data)
mask = csr_matrix((np.ones_like(ys), (nu_ids, nv_ids)))
mask.data = np.ones_like(mask.data)
print(ys.shape)
print(U_hat.shape, V_hat.shape, Y_hat.shape, mask.shape)
print(Y_hat.nonzero()[0].shape, mask.nonzero()[0].shape)
print(Y_hat.data.min(), Y_hat.data.max())
print(mask.data.min(), mask.data.max())

upath = '%s/tr.%s'%(root, postfix)
vpath = '%s/item.%s'%(root, postfix)
mpath = '%s/mask.%s'%(root, postfix)
dump_svm(U_hat, Y_hat, upath)
dump_svm(V_hat, None, vpath)
sp.sparse.save_npz(mpath, mask)
