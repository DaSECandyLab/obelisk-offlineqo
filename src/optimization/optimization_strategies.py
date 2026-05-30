import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import gpytorch
import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.quasirandom import SobolEngine

from util.logger import logger


@dataclass
class TCBOState:
    """TCBO state management for trust region."""
    dim: int
    batch_size: int
    index: int = 0
    length: float = 0.2
    length_min: float = 0.5**8
    length_max: float = 0.8
    failure_counter: int = 0
    failure_tolerance: int = 3
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    best_constraint_value: float = float("inf")
    restart_triggered: bool = False
    center: Optional[List[float]] = None


class _VariationalTimeoutGP(ApproximateGP):
    """Variational GP classifier over timeout labels o in {0, 1}."""

    def __init__(self, inducing_points: torch.Tensor, dimension: int):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=dimension, lengthscale_constraint=Interval(0.005, 4.0))
        )

    def forward(self, x: torch.Tensor):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


class TimeoutRiskModel:
    """Small GP_timeout wrapper that predicts P(timeout | x)."""

    def __init__(self, model: _VariationalTimeoutGP, likelihood: BernoulliLikelihood):
        self.model = model
        self.likelihood = likelihood

    @classmethod
    def fit(
        cls,
        train_X: torch.Tensor,
        timeout_labels: torch.Tensor,
        dimension: int,
        train_steps: int = 50,
    ) -> "TimeoutRiskModel":
        model = _VariationalTimeoutGP(train_X.clone(), dimension)
        likelihood = BernoulliLikelihood()
        model = model.to(train_X)
        likelihood = likelihood.to(train_X)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=0.05,
        )
        mll = VariationalELBO(likelihood, model, num_data=timeout_labels.numel())
        for _ in range(max(1, train_steps)):
            optimizer.zero_grad()
            output = model(train_X)
            loss = -mll(output, timeout_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        return cls(model, likelihood)

    def predict_timeout_probability(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            probabilities = self.likelihood(self.model(X)).mean
        return probabilities.clamp(0.0, 1.0)

    def sample_timeout_probability(self, X: torch.Tensor) -> torch.Tensor:
        """Draw a posterior timeout-probability realization for TS constraints."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_sample = self.model(X).rsample()
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=X.dtype, device=X.device),
            torch.tensor(1.0, dtype=X.dtype, device=X.device),
        )
        return normal.cdf(latent_sample).clamp(0.0, 1.0)


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    @abstractmethod
    def ask(self) -> List[float]:
        """Generate next point using the optimization strategy."""
        pass

    @abstractmethod
    def tell(self, vector: List[float], perf: float, is_timeout: bool | None = None):
        """Update the model with new observation."""
        pass

    def tell_admission_rejection(
        self,
        vector: List[float],
        estimated_latency: float | None = None,
    ) -> None:
        """Update safety state from an admission-stage reject."""
        return None


class VanillaGPStrategy(OptimizationStrategy):
    """Vanilla Gaussian Process Bayesian Optimization strategy, input and output are in 0-1 space."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.X = []
        self.Y = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

    def ask(self) -> List[float]:
        """Generate next point using vanilla GP."""
        if len(self.X) < 2:
            return [np.random.random() for _ in range(self.dimension)]

        try:
            train_X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
            train_Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device).unsqueeze(-1)
            y_std = train_Y.std()
            if y_std < 1e-8:
                return [np.random.random() for _ in range(self.dimension)]

            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            UCB = UpperConfidenceBound(model, beta=0.1, maximize=False)
            bounds = torch.stack([torch.zeros(self.dimension), torch.ones(self.dimension)])

            candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
            x_norm = candidate[0].tolist() if candidate.dim() > 1 else candidate.tolist()

            result = []
            for i in range(self.dimension):
                value = float(x_norm[i])
                value = max(0.0, min(1.0, value))
                result.append(value)

            return result
        except Exception as e:
            logger.warning(f"VanillaGP failed: {str(e)}; fallback to random.")
            return [np.random.random() for _ in range(self.dimension)]

    def tell(self, vector: List[float], perf: float, is_timeout: bool | None = None):
        """Update the model with new observation."""
        self.X.append(vector)
        self.Y.append(perf)


class TCBOStrategy(OptimizationStrategy):
    """Timeout-constrained BO with concurrent local trust regions."""

    def __init__(
        self,
        dimension: int,
        timeout_threshold: float,
        batch_size: int = 1,
        num_trust_regions: int = 4,
        risk_threshold: float = 0.05,
        n_candidates: int = 2000,
        timeout_classifier_train_steps: int = 50,
    ):
        if not 0.0 <= risk_threshold < 1.0:
            raise ValueError("risk_threshold must be in [0, 1)")
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")

        self.dimension = dimension
        self.timeout_threshold = timeout_threshold
        self.batch_size = batch_size
        self.num_trust_regions = max(1, num_trust_regions)
        self.risk_threshold = risk_threshold
        self.n_candidates = max(self.batch_size, int(n_candidates))
        self.timeout_classifier_train_steps = max(1, int(timeout_classifier_train_steps))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double
        self.tkwargs = {"device": self.device, "dtype": self.dtype}
        self.max_cholesky_size = float("inf")

        self.X = []
        self.Y = []
        self.C = []
        self.timeout_labels = []
        self.timeout_constraints = []
        self.objective_observed = []
        self.trust_regions = [
            TCBOState(dim=dimension, batch_size=batch_size, index=i)
            for i in range(self.num_trust_regions)
        ]
        self.state = self.trust_regions[0]
        self.sobol = SobolEngine(dimension=dimension, scramble=True, seed=1)

    def _get_best_index(self, Y: torch.Tensor, C: torch.Tensor):
        """Return the index for the best point"""
        is_feas = (C <= 0).all(dim=-1)
        if is_feas.any():
            score = Y.clone()
            score[~is_feas] = -float("inf")
            return score.argmax()
        return C.clamp(min=0).sum(dim=-1).argmin()

    def _update_state(self, Y_next: torch.Tensor, C_next: torch.Tensor, state: Optional[TCBOState] = None):
        """Update TCBO trust region state."""
        state = state or self.state
        best_ind = self._get_best_index(Y=Y_next, C=C_next)
        best_Y, best_C = Y_next[best_ind].item(), C_next[best_ind].item()

        if best_C <= 0:
            if best_Y > state.best_value:
                has_incumbent = math.isfinite(state.best_value)
                improvement = best_Y - state.best_value if has_incumbent else 0.0
                state.success_counter += 1
                state.failure_counter = 0
                state.best_value = best_Y
                state.best_constraint_value = best_C
                if has_incumbent:
                    logger.info(
                        "TCBO TR%d: Feasible improvement found. Objective gain: %.4f",
                        state.index,
                        improvement,
                    )
                else:
                    logger.info("TCBO TR%d: Feasible incumbent initialized", state.index)
            else:
                state.failure_counter += 1
                state.success_counter = 0
        else:
            if best_C < state.best_constraint_value:
                has_constraint_incumbent = math.isfinite(state.best_constraint_value)
                constraint_improvement = (
                    state.best_constraint_value - best_C
                    if has_constraint_incumbent
                    else 0.0
                )
                state.success_counter += 1
                state.failure_counter = 0
                state.best_constraint_value = best_C
                if has_constraint_incumbent:
                    logger.info(
                        "TCBO TR%d: Constraint improvement found. Constraint reduction: %.4f",
                        state.index,
                        constraint_improvement,
                    )
                else:
                    logger.info("TCBO TR%d: Constraint incumbent initialized", state.index)
            else:
                state.failure_counter += 1
                state.success_counter = 0

        old_length = state.length
        if state.success_counter >= state.success_tolerance:
            state.length = min(2.0 * state.length, state.length_max)
            state.success_counter = 0
            logger.info("TCBO TR%d: expanded from %.4f to %.4f", state.index, old_length, state.length)
        elif state.failure_counter >= state.failure_tolerance:
            state.length = state.length / 2.0
            state.failure_counter = 0
            logger.info("TCBO TR%d: contracted from %.4f to %.4f", state.index, old_length, state.length)

        if state.length < state.length_min:
            state.restart_triggered = True
            logger.info("TCBO TR%d: length below minimum, restart triggered", state.index)

        logger.debug(
            "TCBO TR%d state: success_counter=%d, failure_counter=%d, success_tolerance=%d, failure_tolerance=%d",
            state.index,
            state.success_counter,
            state.failure_counter,
            state.success_tolerance,
            state.failure_tolerance,
        )

    def _get_fitted_model(self, X, Y):
        if X.shape[0] < 2:
            raise ValueError("At least two observations are required to fit a GP model")
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=self.dimension, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(
            X,
            Y,
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            fit_gpytorch_mll(mll)

        return model

    def _gaussian_copula_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Map objective observations to normal scores by empirical rank."""
        flat_values = values.squeeze(-1)
        if flat_values.numel() < 2 or flat_values.std() < 1e-8:
            return values

        order = torch.argsort(flat_values)
        ranks = torch.empty_like(order, dtype=self.dtype, device=self.device)
        ranks[order] = torch.arange(1, flat_values.numel() + 1, **self.tkwargs)
        quantiles = ranks / (flat_values.numel() + 1)
        quantiles = quantiles.clamp(1e-6, 1.0 - 1e-6)
        normal = torch.distributions.Normal(
            torch.tensor(0.0, **self.tkwargs),
            torch.tensor(1.0, **self.tkwargs),
        )
        transformed = normal.icdf(quantiles)
        return transformed.unsqueeze(-1)

    def _random_candidates(self):
        if self.dimension == 0:
            return torch.empty((self.batch_size, 0), **self.tkwargs)
        return self.sobol.draw(self.batch_size).to(**self.tkwargs)

    @staticmethod
    def _distance(vector_a: List[float], vector_b: List[float]) -> float:
        return float(np.linalg.norm(np.asarray(vector_a) - np.asarray(vector_b)))

    def _refresh_trust_region_centers(self) -> None:
        if not self.X:
            return

        feasible = [idx for idx, c in enumerate(self.C) if c <= 0]
        timeout = [idx for idx, c in enumerate(self.C) if c > 0]
        ranked = sorted(feasible, key=lambda idx: self.Y[idx], reverse=True)
        ranked.extend(sorted(timeout, key=lambda idx: self.C[idx]))

        selected_indices = []
        seen_vectors = set()
        for idx in ranked:
            signature = tuple(round(v, 8) for v in self.X[idx])
            if signature in seen_vectors:
                continue
            seen_vectors.add(signature)
            selected_indices.append(idx)
            if len(selected_indices) >= self.num_trust_regions:
                break

        for state, obs_idx in zip(self.trust_regions, selected_indices):
            state.center = list(self.X[obs_idx])

        for state in self.trust_regions[len(selected_indices):]:
            if state.center is None:
                state.center = self.sobol.draw(1).squeeze(0).tolist()

    def _region_for_vector(self, vector: List[float]) -> TCBOState:
        known_regions = [state for state in self.trust_regions if state.center is not None]
        if not known_regions:
            self.trust_regions[0].center = list(vector)
            return self.trust_regions[0]
        return min(known_regions, key=lambda state: self._distance(vector, state.center or vector))

    def _observation_indices_in_region(self, state: TCBOState) -> List[int]:
        if state.center is None:
            return list(range(len(self.X)))
        center = np.asarray(state.center, dtype=float)
        half_width = state.length / 2.0
        indices = []
        for idx, vector in enumerate(self.X):
            point = np.asarray(vector, dtype=float)
            if np.all(np.abs(point - center) <= half_width + 1e-12):
                indices.append(idx)
        return indices

    def _objective_indices_for_region(self, state: TCBOState) -> List[int]:
        local_indices = [
            idx for idx in self._observation_indices_in_region(state)
            if self.C[idx] <= 0 and self.objective_observed[idx]
        ]
        if len(local_indices) >= 2:
            return local_indices

        feasible = [
            idx for idx, c in enumerate(self.C)
            if c <= 0 and self.objective_observed[idx]
        ]
        return feasible if len(feasible) >= 2 else []

    def _timeout_risk_model(self):
        """Fit GP_timeout on all timeout labels and return P(timeout | x)."""
        if len(self.X) < 2:
            return None

        train_X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        labels = torch.tensor(self.timeout_labels, dtype=self.dtype, device=self.device)
        if torch.unique(labels).numel() < 2:
            return None

        try:
            return TimeoutRiskModel.fit(
                train_X=train_X,
                timeout_labels=labels,
                dimension=self.dimension,
                train_steps=self.timeout_classifier_train_steps,
            )
        except Exception as exc:
            logger.debug("TCBO GP_timeout classifier unavailable: %s", exc)
            return None

    def _trust_region_bounds(self, state: TCBOState, x_center: torch.Tensor, model):
        length = torch.full((self.dimension,), state.length, **self.tkwargs)
        try:
            raw_lengthscale = model.covar_module.base_kernel.lengthscale.detach().view(-1).to(**self.tkwargs)
            if raw_lengthscale.numel() == self.dimension and torch.all(torch.isfinite(raw_lengthscale)):
                weights = raw_lengthscale / torch.exp(torch.mean(torch.log(raw_lengthscale)))
                weights = torch.clamp(weights, 0.1, 10.0)
                length = torch.clamp(state.length * weights, max=state.length_max)
        except Exception:
            pass

        tr_lb = torch.clamp(x_center - length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + length / 2.0, 0.0, 1.0)
        return tr_lb, tr_ub

    def _generate_batch(
        self,
        model,
        X,
        Y,
        timeout_model,
        state: TCBOState,
    ):
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

        if state.center is not None:
            x_center = torch.tensor(state.center, dtype=self.dtype, device=self.device)
        else:
            x_center = X[Y.squeeze(-1).argmax(), :].clone()
        tr_lb, tr_ub = self._trust_region_bounds(state, x_center, model)

        pert = self.sobol.draw(self.n_candidates).to(dtype=self.dtype, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        prob_perturb = min(20.0 / self.dimension, 1.0)
        mask = torch.rand(self.n_candidates, self.dimension, **self.tkwargs) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, self.dimension, size=(len(ind),), device=self.device)] = 1

        X_cand = x_center.expand(self.n_candidates, self.dimension).clone()
        X_cand[mask] = pert[mask]

        X_pool = self._safe_candidate_pool(X_cand, timeout_model)
        sampler = MaxPosteriorSampling(
            model=model,
            replacement=X_pool.shape[0] < self.batch_size,
        )
        with torch.no_grad():
            X_next = sampler(X_pool, num_samples=self.batch_size)

        return X_next

    def _safe_candidate_pool(self, X_cand: torch.Tensor, timeout_model: TimeoutRiskModel | None) -> torch.Tensor:
        """Keep candidates satisfying P(timeout|x)-delta <= 0.

        If the classifier has no safe candidate under the current posterior
        draw, keep the least risky slice so TCBO can still move forward.
        """
        if timeout_model is None:
            return X_cand

        probabilities = self._timeout_probabilities_for_candidates(timeout_model, X_cand)
        safe_mask = probabilities <= self.risk_threshold
        if safe_mask.any():
            return X_cand[safe_mask]

        fallback_count = min(
            X_cand.shape[0],
            max(self.batch_size, X_cand.shape[0] // 20),
        )
        least_risky = torch.argsort(probabilities)[:fallback_count]
        return X_cand[least_risky]

    @staticmethod
    def _timeout_probabilities_for_candidates(
        timeout_model: TimeoutRiskModel,
        X_cand: torch.Tensor,
    ) -> torch.Tensor:
        try:
            return timeout_model.sample_timeout_probability(X_cand).reshape(-1)
        except AttributeError:
            return timeout_model.predict_timeout_probability(X_cand).reshape(-1)

    def _generate_next(self):
        """Generate candidate points using constrained Thompson sampling."""
        if self.dimension == 0 or len(self.X) < 2:
            return self._random_candidates()

        self._refresh_trust_region_centers()
        timeout_model = self._timeout_risk_model()

        candidates = []
        for state in self.trust_regions:
            objective_indices = self._objective_indices_for_region(state)
            if len(objective_indices) < 2:
                continue

            train_X = torch.tensor(
                [self.X[idx] for idx in objective_indices],
                dtype=self.dtype,
                device=self.device,
            )
            train_Y = torch.tensor(
                [self.Y[idx] for idx in objective_indices],
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(-1)
            if train_Y.std() < 1e-8:
                continue
            train_Y_model = self._gaussian_copula_transform(train_Y)

            try:
                model = self._get_fitted_model(train_X, train_Y_model)
                with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                    X_next = self._generate_batch(
                        model=model,
                        X=train_X,
                        Y=train_Y_model,
                        timeout_model=timeout_model,
                        state=state,
                    )
                score = model.posterior(X_next).mean.max().item()
                candidates.append((score, X_next[0], state))
            except Exception as exc:
                logger.debug("TCBO TR%d candidate generation skipped: %s", state.index, exc)

        if not candidates:
            return self._random_candidates()

        _, best_candidate, best_state = max(candidates, key=lambda item: item[0])
        self.state = best_state
        return best_candidate.unsqueeze(0)

    def ask(self) -> List[float]:
        """Generate next point using TCBO."""
        try:
            nexts = self._generate_next()
            next_point = nexts[0].cpu().numpy().tolist()
            return next_point

        except Exception as e:
            logger.warning(f"TCBO failed: {str(e)}; fallback to random.")
            return [np.random.random() for _ in range(self.dimension)]

    def tell(self, vector: List[float], perf: float, is_timeout: bool | None = None):
        """Update TCBO with latency y_i and explicit timeout label o_i."""
        region = self._region_for_vector(vector)
        self.X.append(vector)
        self.Y.append(-perf)
        if is_timeout is None:
            is_timeout = perf >= self.timeout_threshold
        else:
            is_timeout = bool(is_timeout)
        constraint_value = perf - self.timeout_threshold
        if is_timeout and constraint_value <= 0:
            constraint_value = 1e-6
        self.C.append(constraint_value)
        timeout_label = 1 if is_timeout else 0
        self.timeout_labels.append(timeout_label)
        self.timeout_constraints.append(timeout_label - self.risk_threshold)
        self.objective_observed.append(not is_timeout)
        status = "TIMEOUT" if is_timeout else "FEASIBLE"
        logger.info(
            "TCBO: Recorded observation - Performance: %.2fms, Status: %s, Constraint: %.3f",
            perf,
            status,
            constraint_value,
        )
        Y_tensor = torch.tensor([-perf], dtype=self.dtype, device=self.device).unsqueeze(-1)
        C_tensor = torch.tensor([constraint_value], dtype=self.dtype, device=self.device).unsqueeze(-1)
        self._update_state(Y_tensor, C_tensor, state=region)
        feasible_count = sum(1 for c in self.C if c <= 0)
        total_count = len(self.C)
        timeout_count = sum(1 for c in self.C if c > 0)

        self.state = region
        logger.info(f"TCBO: TR{region.index} length={region.length:.4f}, "
                   f"Best value={region.best_value:.2f}, "
                   f"Constraint violation={region.best_constraint_value:.3f}")
        logger.info(f"TCBO: Observations - Total: {total_count}, Feasible: {feasible_count}, Timeouts: {timeout_count}")
        logger.info(f"TCBO: Counters - Success: {region.success_counter}/{region.success_tolerance}, "
                   f"Failure: {region.failure_counter}/{region.failure_tolerance}")

        if self.should_restart():
            self.restart()
        self._refresh_trust_region_centers()

    def tell_admission_rejection(
        self,
        vector: List[float],
        estimated_latency: float | None = None,
    ) -> None:
        """Record an Evaluator admission reject with its conservative estimate.

        The estimate is retained in Y as a configuration-performance pair from
        the Evaluator, but ``objective_observed`` remains false so local
        objective GPs and Reasoner context use only real uncensored latencies.
        """
        region = self._region_for_vector(vector)
        latency = (
            float(estimated_latency)
            if estimated_latency is not None and math.isfinite(float(estimated_latency))
            else self.timeout_threshold
        )
        constraint_value = max(latency - self.timeout_threshold, 1e-6)
        self.X.append(vector)
        self.Y.append(-latency)
        self.C.append(constraint_value)
        self.timeout_labels.append(1)
        self.timeout_constraints.append(1 - self.risk_threshold)
        self.objective_observed.append(False)
        Y_tensor = torch.tensor([-latency], dtype=self.dtype, device=self.device).unsqueeze(-1)
        C_tensor = torch.tensor([constraint_value], dtype=self.dtype, device=self.device).unsqueeze(-1)
        self._update_state(Y_tensor, C_tensor, state=region)
        self.state = region
        if self.should_restart():
            self.restart()
        self._refresh_trust_region_centers()
        logger.info(
            "TCBO: Recorded admission reject as safety feedback - estimate: %.2fms, constraint: %.3f",
            latency,
            constraint_value,
        )

    def should_restart(self) -> bool:
        """Check if TCBO should restart."""
        return self.state.restart_triggered

    def restart(self):
        """Restart TCBO with fresh state."""
        index = self.state.index
        self.trust_regions[index] = TCBOState(dim=self.dimension, batch_size=self.batch_size, index=index)
        self.state = self.trust_regions[index]
        logger.info("TCBO TR%d restarted due to trust region convergence", index)
