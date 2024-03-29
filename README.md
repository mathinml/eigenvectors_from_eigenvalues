# eigenvectors_from_eigenvalues

Numpy and PyTorch implementations for Eigenvectors from Eigenvalues (https://arxiv.org/pdf/1908.03795.pdf).

Formula: 

![](https://qiniu.mdnice.com/8cc1efff0992f71e4d85a0ef92ac7366.svg+xml)



Code:

```python
def sub_matrix_np0(A, n, j):
    row, row[j] = np.ones(n, dtype=bool), False
    return A[row][:,row]
    
def eigv_ij_tao_np(A, i, j):
    eigvals = LA.eigvals(A)
    M_j = sub_matrix_np0(A, eigvals.shape[0], j)
    eigvals_M_j = LA.eigvals(M_j)
    eigvals_M_j = eigvals_M_j - eigvals[i]
    eigvals, eigvals[i] = eigvals - eigvals[i], 1.0
    return np.prod(eigvals_M_j) / np.prod(eigvals)
```
