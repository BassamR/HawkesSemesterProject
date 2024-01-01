"""
Author: Adrian Jarret, with modifications by Bassam El Rawas.

This file implements a few common operators, useful in the development of PFW for the Hawkes
likelihood problem.
"""

import numpy as np

import pyxu.abc.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu.info.ptype as pxt
import pyxu.operator as pxop

__all__ = [
    "L1NormPositivityConstraint",
    "L1NormPartialReg",
    "L1NormPartialPositivityConstraint"
]

class L1NormPositivityConstraint(pxo.ProxFunc):
    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        if arr.ndim <= 1:
            res = arr.sum() if xp.all(arr >= 0) else xp.inf
            return xp.r_[res].astype(arr.dtype)
        else:
            res = xp.full(arr.shape[:-1], xp.inf)
            indices = xp.all(arr >= 0, axis=-1)
            res[indices] = arr[indices].sum(axis=-1)
            return res.astype(arr.dtype)

    @pxrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        res = xp.zeros_like(arr)
        # Projection sur le nonneg orthant: si l'array est positif appliquer le prox,
        # sinon laisser a 0
        indices = arr > 0
        # le proximal de la norme l1 (no sgn(arr) because anyways arr is positive)
        res[indices] = xp.fmax(0, arr[indices] - tau)
        return res


class L1NormPartialReg(pxo.ProxFunc):
    def __init__(self, shape: pxt.OpShape, S: np.array, regLambda: float):
        super().__init__(shape=shape)
        self.S = S  # indices which we want to regularize on
        self.regLambda = regLambda  # regularization parameter

        self.regDiag = np.zeros(self.shape[1])
        self.regDiag[self.S] = self.regLambda

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # lambda * l1 norm of [arr]_S = sum of lambda*arr[i] for i in S
        return self.regLambda * sum(np.abs(arr[i]) for i in self.S)

    @pxrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        res = xp.zeros_like(arr)

        res = np.sign(arr) * xp.fmax(0, np.abs(arr) - tau * self.regDiag)
        return res
    

class L1NormPartialPositivityConstraint(pxo.ProxFunc):
    def __init__(self, shape: pxt.OpShape, totalSize: int, S: np.array, regLambda: float, supportIndices: np.array):
        super().__init__(shape=shape)
        self.S = S  # indices which we want to regularize on
        self.regLambda = regLambda  # regularization parameter

        self.totalSize = totalSize  # total size of the dense input
        self.supportIndices = supportIndices  # support indices of the sparse input

        self.regDiag = np.zeros(self.totalSize)
        self.regDiag[self.S] = self.regLambda

        self.injection = pxop.SubSample(self.totalSize, self.supportIndices).T
        self.subsampling = pxop.SubSample(self.totalSize, self.supportIndices)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)

        # Upsample array to the right size (=self.totalSize)
        arrUp = self.injection(arr)

        # Apply partial L1 norm
        res = self.regLambda * sum(np.abs(arrUp[i]) for i in self.S) if xp.all(arrUp >= 0) else xp.inf
        return xp.r_[res].astype(arr.dtype)

    @pxrt.enforce_precision(i=["arr", "tau"])
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        # Upsample array
        arrUp = self.injection(arr)
        xp = pxu.get_array_module(arrUp)
        res = xp.zeros_like(arrUp)

        # Projection sur le nonneg orthoant: si l'array est positif appliquer le prox, sinon laisser a 0
        indices = arrUp > 0
        res[indices] = xp.fmax(0, arrUp[indices] - tau * self.regDiag[indices])
        return self.subsampling(res)


if __name__ == "__main__":
    N = 10
    # posIndicator = NonNegativeOrthant(shape=(1, None))
    #
    # a = np.random.normal(size=N)
    # b = posIndicator.prox(a, tau=1)
    # print(posIndicator(a))
    # print(b)
    # print(posIndicator(b))
    #
    # print("0-d input: {}".format(posIndicator(np.r_[-1])))
    # print("3-d input: {}".format(posIndicator(np.arange(24).reshape((2, 3, 4)) - 3)))

    print("\nPositivity constraint:")
    posL1Norm = L1NormPositivityConstraint(shape=(1, None))

    a = np.random.normal(size=N)
    b = posL1Norm.prox(a, tau=0.1)
    print(posL1Norm(a))
    print(b)
    print(posL1Norm(b))

    print("0-d input: {}".format(posL1Norm(np.r_[-1])))
    print("3-d input: {}".format(posL1Norm(np.arange(24).reshape((2, 3, 4)) - 3)))
