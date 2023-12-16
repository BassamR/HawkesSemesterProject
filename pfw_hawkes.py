"""
Author: Adrian Jarret (+ modifications by Bassam El Rawas)

Module implementing the Polyatomic Frank-Wolfe algorithm for the Multivariate Hawkes
Likelihood problem.
"""

import datetime as dt
import time
import typing as typ

import numpy as np

import utils as ut
from hawkes_likelihood import HawkesLikelihood

import pyxu.abc.operator as pxo
import pyxu.abc.solver as pxs
import pyxu.operator as pxop
import pyxu.opt.stop as pxos
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu.info.ptype as pxt
from pyxu.opt.solver.pgd import PGD

__all__ = [
    "VFWLasso",
    "PFWLasso",
    "dcvStoppingCrit",
]


class _GenericFWLasso(pxs.Solver):
    r"""
    Base class for Frank-Wolfe algorithms (FW) for the LASSO problem.

    This is an abstract class that cannot be instantiated.
    """

    def __init__(
            self,
            hpL: HawkesLikelihood,  # modified this (data -> hpL, for hawkes process likelihood)
            forwardOp: pxo.LinOp,
            lambda_: float,
            *,
            folder=None,  # typ.Optional[pyct.PathLike] = None,
            exist_ok: bool = False,
            stop_rate: int = 1,
            writeback_rate: typ.Optional[int] = None,
            verbosity: int = 1,
            show_progress: bool = True,
            log_var: pxt.VarName = (
                    "x",
                    "pos",
                    "val",
                    "dcv",
            ),
    ):
        super().__init__(
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )
        self.lambda_ = pxrt.coerce(lambda_)
        self.forwardOp = forwardOp
        self.hpL = hpL
        #self.data = pxrt.coerce(data)  # commented this
        # modified this
        # self._data_fidelity = (
        #         0.5 * pxop.SquaredL2Norm(dim=self.forwardOp.shape[0]).argshift(-self.data) * self.forwardOp
        # )
        self._data_fidelity = (
            self.hpL.negLogL
        )
        regIndices = [index for k in range(self.hpL.M) for index in range(k*self.hpL.M + k + 1, k*self.hpL.M + k + self.hpL.M + 1)]
        self._penalty = ut.L1NormPartialReg(shape=(1, self.forwardOp.shape[1]), 
                                             S=regIndices, 
                                             regLambda=self.lambda_)
        #self._penalty = self.lambda_ * pxop.L1Norm(dim=self.forwardOp.shape[1])
        #self._bound = 0.5 * pxop.SquaredL2Norm(dim=self.forwardOp.shape[0]).apply(data)[0] / self.lambda_

    def m_init(self, **kwargs):
        #xp = pxu.get_array_module(self.data)  # i commented this
        mst = self._mstate  # shorthand
        mst["dcv"] = np.inf  # "dual certificate value"
        mst["val"] = np.array([], dtype=pxrt.getPrecision().value)
        mst["pos"] = np.array([], dtype="int32")

    def default_stop_crit(self) -> pxs.StoppingCriterion:
        stop_crit = pxos.RelError(
            eps=1e-4,
            var="objective_func",
            f=None,
            norm=2,
            satisfy_all=True,
        )
        return stop_crit

    def solution(self) -> pxt.NDArray:
        """
        Returns
        -------
        p: NDArray
            (..., N) solution.
        """
        data, _ = self.stats()
        return data.get("x")

    def objective_func(self) -> pxt.NDArray:
        # return self.rs_data_fid(self._mstate["pos"]).apply(self._mstate["val"]) + self.lambda_ * \
        #     pxop.L1Norm(self._mstate["val"].shape[0]).apply(self._mstate["val"])
        injection = pxop.SubSample(self.forwardOp.shape[1], self._mstate["pos"]).T

        #print('injection:', injection(self._mstate["val"]))

        return self.rs_data_fid(self._mstate["pos"]).apply(self._mstate["val"]) \
            + self._penalty.apply(injection(self._mstate["val"]))

    @property
    def _dense_iterate(self):
        mst = self._mstate
        assert mst["val"].shape == mst["pos"].shape
        xp = pxu.get_array_module(mst["val"])
        res = xp.zeros(self.forwardOp.shape[1], dtype=pxrt.getPrecision().value)
        res[mst["pos"]] = mst["val"]
        return res

    def fit(self, **kwargs):
        """
        Set the default value of ``track_objective`` as True, as opposed to the behavior defined in the base class
        ``Solver``.
        """
        track_objective = kwargs.pop("track_objective", True)
        positivity_c = kwargs.pop("positivity_constraint", True)  # i changed this
        self._astate["positivity_c"] = positivity_c

        super().fit(track_objective=track_objective, **kwargs)
        self._mstate["x"] = self._dense_iterate

    # rs = restricted support
    def rs_data_fid(self, support_indices: pxt.NDArray) -> pxo.DiffFunc:
        # return 0.5 * pxop.SquaredL2Norm(dim=self.forwardOp.shape[0]).argshift(
        #     -self.data) * self.rs_forwardOp(support_indices=support_indices)
        # modified this
        return self.hpL.E * self.rs_forwardOp(support_indices=support_indices)

    def rs_forwardOp(self, support_indices: pxt.NDArray) -> pxo.LinOp:
        """
        Used in order to apply the forward operator to a sparse input.

        Needs to be over-written in case of a more efficient implementation of the reduced support
        forward operator implementation (only in the forward pass).
        """
        # TODO: slice on the columns, instead of augmenting the input vector
        # wont change the size of the output, but will change the size of the input

        # SubSample(N,k): R^N -> R^k, which will subsample. Transpose it to obtain
        # an upsampling operation R^k -> R^N.
        injection = pxop.SubSample(self.forwardOp.shape[1], support_indices).T
        return self.forwardOp * injection


class PFWLasso(_GenericFWLasso):
    r"""
    Polyatomic version of the Frank-Wolfe algorithm (FW) specifically design to solve the LASSO problem of the form

    .. math ::

        {\argmin_{\mathbf{x} \in \mathbb{R}^N} \; \frac{1}{2} \lVert \mathbf{y} - \mathbf{G}(\mathbf{x}) \rVert_2^2
        + \lambda \lVert \mathbf{x} \rVert_1 }

    where:

    * :math:`\mathbf{G}:\mathbb{R}^N\rightarrow\mathbb{C}^L` is a *linear* operator, referred to as *forward*
     or *measurement* operator.
    * :math:`\mathbf{y}\in\mathbb{C}^L` is the vector of measurements data.
    * :math:`\lambda>0` is the *regularization* or *penalty* parameter.

    This algorithm is presented in the paper [PFW]_ as an improvement over the Vanilla FW algorithm in order to
    converge faster to a solution. Compared to other LASS solvers, it is particularly efficient when the solution is
    supposed to be very sparse, in terms of convergence time as well as memory requirements.

    The FW algorithms ensure convergence of the iterates in terms of the value of the objective function with a
    convergence rate of :math:`\mathcal{O}(1/k)` [RevFW]_. Consequently, the default stopping criterion is set as a
    threshold over the relative improvement of the value of the objective function. The algorithm stops if the relative
    improvement is below 1e-4. If this default stopping criterion is used, you must not fit the algorithm with argument
    ``track_objective=False``.

    **Remark 1:** The array module used in the algorithm iterations is inferred from the module of the input
        measurements.

    **Remark 2:** The second step of the PFW algorithm consists in computing/updating the weights attributed to the
    atoms. To do so, the algorithm solves a LASSO problem with a restricted support defined by the selected atoms.
    The implementation provided for this step makes use of an injection/sub-sampling operator that allows to extract
    the sub-problem of interest. This generic procedure might be sub-optimal and could be improved depending on the
    forward operator (e.g. non-uniform FFT as a restricted support FFT). The interested user can provide an
    alternative and posisbly more efficient method for the correction of the weights by overriding the
    ``.rs_correction()`` method in a child class.

    ``PolyatomicFWforLasso.fit()`` **Parametrisation**

    track_objective: Bool
        Indicator to keep track of the value of the objective function along the iterations.
        This value is optional for Polyatomic FW, but can be used to determine a stopping criterion.
        Default is True, can be set to False to accelerate the solving time.
    """

    def __init__(
            self,
            hpL: HawkesLikelihood,
            forwardOp: pxo.LinOp,
            lambda_: float,
            ms_threshold: float = 0.7,  # multi spikes threshold at init
            init_correction_prec: float = 0.2,
            final_correction_prec: float = 1e-4,
            remove_positions: bool = False,
            min_correction_steps: int = 5,
            max_correction_steps: int = 100,
            *,
            folder=None,  # : typ.Optional[pyct.PathLike] = None,
            exist_ok: bool = False,
            stop_rate: int = 1,
            writeback_rate: typ.Optional[int] = None,
            verbosity: int = 10,
            show_progress: bool = True,
            log_var: pxt.VarName = (
                    "x",
                    "pos",
                    "val",
                    "dcv",
            ),
    ):

        r"""

        Parameters
        ----------
        data: NDArray
            (..., L) measurements term(s) in the LASSO objective function.
        forwardOp: LinOp
            (N, L) measurement operator in the LASSO objective function.
        lambda_: float
            Regularisation parameter from the LASSO problem.
        ms_threshold: float
            Initial threshold for identifying the first atoms, given as a rate of the dual certificate value. This
            parameter impacts the number of atomes chosen at the first step, but also at all the following iterations.
            Low `ms_threshold` value implies a lot of flexiility in the choice of the atoms, whereas high value leads
            to restrictive selection and thus very few atomes at each iteration.
        init_correction_prec: float
            Precision of the first stopping criterion for the step of correction of the weights at first iteration.
        final_correction_prec: float
            Lower bound for the precision of the correction steps. When reached, the precision does no longer decrease
            with the iterations.
        remove_positions: bool
            When set to `True`, the atoms that get attributed a null weight after the correction of the weights are
            removed from the current iterate. When the parameter is set to `False` however, these atoms remain in the
            set of active indices and thus can still be attributed weight at later iterations.
        """
        self._ms_threshold = ms_threshold
        self._init_correction_prec = init_correction_prec
        self._final_correction_prec = final_correction_prec
        self._remove_positions = remove_positions
        self._min_correction_steps = min_correction_steps
        self._max_correction_steps = max_correction_steps
        super().__init__(
            hpL=hpL,
            forwardOp=forwardOp,
            lambda_=lambda_,
            folder=folder,
            exist_ok=exist_ok,
            stop_rate=stop_rate,
            writeback_rate=writeback_rate,
            verbosity=verbosity,
            show_progress=show_progress,
            log_var=log_var,
        )

    def fit(self, **kwargs):
        self._astate["lock"] = kwargs.pop("lock_reweighting", False)
        self._prec_rule = kwargs.pop("precision_rule", lambda k: 1 / k)

        l_constant = kwargs.pop("diff_lipschitz", None)
        if l_constant is None:
            lipschitz_time = time.time()
            self._data_fidelity.estimate_diff_lipschitz(tol=1e-3)
            print("Computation of diff_lipschitz takes {:.4f}".format(time.time() - lipschitz_time))
        else:
            print("diff_lipschitz constant provided.")
            self._data_fidelity._diff_lipschitz = l_constant

        super().fit(**kwargs)

    def m_init(self, **kwargs):
        super(PFWLasso, self).m_init(**kwargs)
        xp = pxu.get_array_module(self._mstate["val"])
        mst = self._mstate
        # Init x0 = 1 on the mu indices, 0 on the alpha indices
        mst["pos"] = np.array([k + k*self.hpL.M for k in range(self.hpL.M)], dtype="int32")  # indices of mu
        mst["val"] = np.array([1 for _ in range(mst["pos"].shape[0])], dtype=pxrt.getPrecision().value)
        mst["delta"] = None  # initial buffer for multi spike thresholding
        mst["correction_prec"] = self._init_correction_prec

    def m_step(self):
        mst = self._mstate  # shorthand
        #print('val:', mst["val"])
        #print('pos:', mst['pos'])
        #mgrad = self.forwardOp.adjoint(self.data - self.rs_forwardOp(mst["pos"]).apply(mst["val"])) / self.lambda_
        mgrad = self.forwardOp.adjoint(self.hpL.E.grad(self.rs_forwardOp(mst["pos"]).apply(mst["val"]))) / self.lambda_
        #print('mgrad:', mgrad)
        if self._astate["positivity_c"]:
            mst["dcv"] = mgrad.max()
            maxi = mst["dcv"]
        else:
            mst["dcv"] = max(mgrad.max(), mgrad.min(), key=abs)  # float
            maxi = abs(mst["dcv"])
        # mst["dcv"] is stored as a float in this case and does not handle stacked multidimensional inputs.
        if self._astate["idx"] == 1:
            mst["delta"] = maxi * (1.0 - self._ms_threshold)
            mst["N_indices"] = []
            mst["N_candidates"] = []
            mst["correction_iterations"] = []
            mst["correction_durations"] = []
        thresh = maxi - (2 / (self._astate["idx"] + 1)) * mst["delta"]
        if self._astate["positivity_c"]:
            new_indices = (mgrad > max(thresh, 1.0)).nonzero()[0]
        else:
            new_indices = (abs(mgrad) > max(thresh, 1.0)).nonzero()[0]
        mst["N_candidates"].append(new_indices.size)

        xp = pxu.get_array_module(mst["val"])
        if new_indices.size > 0:
            add_indices = new_indices[xp.invert(xp.isin(new_indices, mst["pos"]))]
            # print("actual new indices: {}".format(add_indices.size))  # i.e. indices that are not already in the support
            mst["pos"] = xp.hstack([mst["pos"], add_indices])
            mst["val"] = xp.hstack([mst["val"], xp.zeros_like(add_indices)])

        # else would correspond to empty new_indices, in this case the set of active indices does not change
        mst["N_indices"].append(mst["pos"].size)

        mst["correction_prec"] = max(self._init_correction_prec * self._prec_rule(self._astate["idx"]),
                                     self._final_correction_prec)
        if mst["pos"].size > 1:
            mst["val"] = self.rs_correction(mst["pos"], self._mstate["correction_prec"])
        elif mst["pos"].size == 1:  # case 1-sparse solution => the solution can be computed explicitly
            tmp = xp.zeros(self.forwardOp.shape[1], dtype=pxrt.getPrecision().value)
            tmp[mst["pos"]] = 1.0
            column = self.forwardOp(tmp)
            corr = xp.dot(self.data, column).real
            if abs(corr) <= self.lambda_:
                mst["val"] = xp.zeros(1, dtype=pxrt.getPrecision().value)
            elif corr > self.lambda_:
                mst["val"] = np.r_[
                    (corr - self.lambda_) / pxop.SquaredL2Norm(dim=self.forwardOp.shape[0]).apply(column)[0]]
            else:
                mst["val"] = np.r_[
                    (corr + self.lambda_) / pxop.SquaredL2Norm(dim=self.forwardOp.shape[0]).apply(column)[0]]
        else:
            mst["val"] = xp.array([], dtype=pxrt.getPrecision().value)

        if self._remove_positions:
            to_keep = (abs(mst["val"]) > 1e-5).nonzero()[0]
            ast = self._astate
            if ast["stdout"] and ast["idx"] % ast["log_rate"] == 0:
                print("\tAtoms kept: {}/{}".format(to_keep.size, mst["val"].size))
            mst["pos"] = mst["pos"][to_keep]
            mst["val"] = mst["val"][to_keep]

    def rs_correction(self, support_indices: pxt.NDArray, precision: float) -> pxt.NDArray:
        r"""
        Method to update the weights after the selection of the new atoms. It solves a LASSO problem with a
        restricted support corresponding to the current set of active indices (active atoms). As mentioned,
        this method should be overriden in a child class for case-specific improved implementation.

        Parameters
        ----------
        support_indices: NDArray
            Set of active indices.
        precision: float
            Value of the stopping criterion for the correction.

        Returns
        -------
        weights: NDArray
            New iterate with the updated weights.
        """

        def correction_stop_crit(eps) -> pxs.StoppingCriterion:
            stop_crit = pxos.RelError(
                eps=eps,
                var="x",
                f=None,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        rs_data_fid = self.rs_data_fid(support_indices)
        #print('rs_correction was called')
        x0 = self._mstate["val"]
        dim = x0.shape[0]
        if self._astate["positivity_c"]:
            #print('positivity constraint called')
            penalty = ut.L1NormPositivityConstraint(shape=(1, dim))
        else:
            penalty = pxop.L1Norm(dim)
        apgd = PGD(rs_data_fid, self.lambda_ * penalty, show_progress=False)
        stop = pxos.MaxIter(n=self._min_correction_steps)  # min number of reweighting steps
        if not self._astate["lock"]:
            stop &= correction_stop_crit(precision)
            stop |= pxos.MaxIter(n=self._max_correction_steps)  # max number of reweighting steps

        apgd.fit(
            x0=x0,
            tau=1 / self._data_fidelity._diff_lipschitz,
            stop_crit=(stop | pxos.MaxDuration(t=dt.timedelta(seconds=1000)))
        )
        self._mstate["correction_iterations"].append(apgd.stats()[1]["iteration"][-1])
        self._mstate["correction_durations"].append(apgd.stats()[1]["duration"][-1])
        sol, _ = apgd.stats()
        return sol["x"]

    def diagnostics(self, log: bool = False):
        import matplotlib.pyplot as plt

        hist = self.stats()[1]

        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("Performance analysis of PolyCLEAN")
        ax = fig.add_subplot(243)
        ax.plot(self._mstate["N_indices"], label="Support size", marker="x", alpha=.5)
        ax.plot(self._mstate["N_candidates"], label="Candidates", marker="x", alpha=.5)
        ax.set_xlabel("Iter.")
        ax.set_title("Support")
        ax.legend()
        ax = fig.add_subplot(244)
        ax.plot(self._mstate["correction_iterations"], label="iterations", marker=".", c='#2ca02c', alpha=.5)
        ax.set_ylim(bottom=0.)
        ax.set_xlabel("Iter.")
        # ax.legend()
        ax2 = ax.twinx()
        ax2.plot(self._mstate["correction_durations"], label="duration", marker="x", c='#d62728', alpha=.5)
        ax2.set_ylim(bottom=0.)
        ax2.set_ylabel("Duration (s)")
        ax.set_title("Correction iterations")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        ax = fig.add_subplot(247)
        ax.plot(hist['duration'][1:], self._mstate["N_indices"], label="Support size", marker="x", alpha=.5, )
        ax.plot(hist['duration'][1:], self._mstate["N_candidates"], label="candidates", marker="x", alpha=.5, )
        ax.set_xlim(left=0.)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax = fig.add_subplot(248)
        ax.plot(hist['duration'][1:], self._mstate["correction_iterations"], label="iterations", marker=".",
                c='#2ca02c', alpha=.5)
        ax.set_ylim(bottom=0.)
        ax.set_xlim(left=0.)
        ax.set_xlabel("Time (s)")
        # plt.legend()
        ax2 = ax.twinx()
        ax2.plot(hist['duration'][1:], self._mstate["correction_durations"], label="duration", marker="x",
                 c='#d62728', alpha=.5)
        ax2.set_ylim(bottom=0.)
        ax2.set_ylabel("Duration (s)")
        ax2.set_xlim(left=0.)
        # plt.legend()
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        plt.subplot(121)
        if log:
            plt.yscale('log')
        plt.scatter(hist['duration'], (hist['Memorize[objective_func]'] - hist['Memorize[objective_func]'][-1]) / (
                hist['Memorize[objective_func]'][0] - hist['Memorize[objective_func]'][-1]), label="PFW",
                    s=20,
                    marker="+")
        plt.title('LASSO objective function')
        plt.legend()
        plt.xlim(left=0.)
        plt.xlabel("Time")
        plt.show()

    def post_process(self):
        """
        Solve the LASSO objective problem constraining the support of the solution to be included within the support of
        the last PFW iterate. The stopping criterion of the problem is set to the final precision of reweighting steps.
        """
        mst = self._mstate
        mst["val"] = self.rs_correction(mst["pos"], self._final_correction_prec)


def dcvStoppingCrit(eps: float = 1e-2) -> pxs.StoppingCriterion:
    r"""
    Instantiate a ``StoppingCriterion`` class based on the maximum value of the dual certificate, that is automatically
    computed along the iterations of any Frank-Wolfe algorithm. At convergence, the maximum absolute value component of
    the dual certificate is 1, and thus tracking the difference to 1 provides us with a natural stopping criterion.

    Parameters
    ----------
    eps: float
        Precision on the distance to 1 for stopping.

    Returns
    -------
    stop_crit: pycs.StoppingCriterion
        An instance of the ``StoppingCriterion`` class that can be fed to the ``.fit()`` method of any
        Frank-Wolfe solver.
    """

    def abs_diff_to_one(x):
        return abs(x) - 1.0

    stop_crit = pxos.AbsError(
        eps=eps,
        var="dcv",
        f=abs_diff_to_one,
        norm=2,
        satisfy_all=True,
    )
    return stop_crit
