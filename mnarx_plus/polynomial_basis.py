"""Polynomial basis generation with hyperbolic truncation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List, Sequence

import numpy as np


@dataclass
class PolynomialTerm:
    exponents: np.ndarray
    name: str
    sources: List[str]


class PolynomialBasisGenerator:
    """Generates sparse-compatible polynomial terms.

    paper-specified: polynomial expansion with truncation.
    implementation choice: exhaustive exponent scan with hyperbolic-norm filtering.
    """

    def __init__(self, degree: int = 3, hyperbolic_q: float = 1.0, max_terms: int = 400):
        self.degree = degree
        self.hyperbolic_q = hyperbolic_q
        self.max_terms = max_terms
        self.terms_: List[PolynomialTerm] = []
        self.input_features_: List[str] = []

    def _valid(self, exponents: np.ndarray) -> bool:
        total_degree = int(exponents.sum())
        if total_degree == 0 or total_degree > self.degree:
            return False
        q = self.hyperbolic_q
        value = np.power(np.sum(np.power(exponents, q)), 1.0 / q)
        return value <= self.degree + 1e-12

    def fit(self, feature_names: Sequence[str]) -> None:
        self.input_features_ = list(feature_names)
        p = len(feature_names)
        terms: List[PolynomialTerm] = []
        for tup in product(range(self.degree + 1), repeat=p):
            exps = np.asarray(tup, dtype=int)
            if not self._valid(exps):
                continue
            sources = [feature_names[i] for i, e in enumerate(exps) if e > 0]
            pieces = [f"{feature_names[i]}^{e}" for i, e in enumerate(exps) if e > 0]
            terms.append(PolynomialTerm(exponents=exps, name=" * ".join(pieces), sources=sources))
            if len(terms) >= self.max_terms:
                break
        if not terms:
            raise ValueError("No polynomial terms generated; increase max_terms or adjust settings.")
        self.terms_ = terms

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.terms_:
            raise RuntimeError("PolynomialBasisGenerator must be fit before transform.")
        Z = np.empty((X.shape[0], len(self.terms_)), dtype=float)
        for j, term in enumerate(self.terms_):
            vals = np.ones(X.shape[0], dtype=float)
            nz = np.where(term.exponents > 0)[0]
            for idx in nz:
                vals *= np.power(X[:, idx], term.exponents[idx])
            Z[:, j] = vals
        return Z

    @property
    def term_names(self) -> List[str]:
        return [t.name for t in self.terms_]
