import os
from dataclasses import dataclass, field

import joblib
import numpy as np
import skfuzzy as fuzz

from src.data_download import get_project_root


@dataclass
class FuzzyRiskModel:
    age_universe: np.ndarray = field(default_factory=lambda: np.arange(20, 80.5, 0.5))
    chol_universe: np.ndarray = field(default_factory=lambda: np.arange(100, 400.5, 0.5))
    thalach_universe: np.ndarray = field(default_factory=lambda: np.arange(60, 200.5, 0.5))
    risk_universe: np.ndarray = field(default_factory=lambda: np.arange(0.0, 1.001, 0.001))

    def __post_init__(self):
        self.age_memberships = {
            "young": fuzz.trapmf(self.age_universe, [20, 20, 35, 40]),
            "middle": fuzz.trapmf(self.age_universe, [35, 40, 55, 60]),
            "old": fuzz.trapmf(self.age_universe, [55, 60, 80, 80]),
        }
        self.chol_memberships = {
            "low": fuzz.trapmf(self.chol_universe, [100, 100, 180, 200]),
            "medium": fuzz.trapmf(self.chol_universe, [180, 200, 260, 280]),
            "high": fuzz.trapmf(self.chol_universe, [260, 280, 400, 400]),
        }
        self.thalach_memberships = {
            "low": fuzz.trapmf(self.thalach_universe, [60, 60, 100, 120]),
            "medium": fuzz.trapmf(self.thalach_universe, [100, 120, 140, 160]),
            "high": fuzz.trapmf(self.thalach_universe, [140, 160, 200, 200]),
        }
        self.risk_memberships = {
            "low": fuzz.trapmf(self.risk_universe, [0.0, 0.0, 0.3, 0.4]),
            "medium": fuzz.trapmf(self.risk_universe, [0.3, 0.4, 0.6, 0.7]),
            "high": fuzz.trapmf(self.risk_universe, [0.6, 0.7, 1.0, 1.0]),
        }
        self.rules = [
            (("age", "old"), ("chol", "high"), None, "high"),
            (("age", "old"), ("thalach", "low"), None, "high"),
            (("age", "middle"), ("chol", "high"), ("thalach", "low"), "high"),
            (("age", "young"), ("chol", "low"), ("thalach", "high"), "low"),
            (("age", "young"), ("chol", "medium"), ("thalach", "high"), "low"),
            (("age", "middle"), ("chol", "medium"), ("thalach", "medium"), "medium"),
            (("age", "old"), ("chol", "medium"), ("thalach", "medium"), "medium"),
            (("age", "young"), ("chol", "high"), ("thalach", "medium"), "medium"),
        ]

    def _membership(self, universe, membership_map, label, value):
        return fuzz.interp_membership(universe, membership_map[label], float(value))

    def predict_one(self, record):
        age_value = float(record["age"])
        chol_value = float(record["chol"])
        thalach_value = float(record["thalach"])

        evidence = {
            ("age", "young"): self._membership(self.age_universe, self.age_memberships, "young", age_value),
            ("age", "middle"): self._membership(self.age_universe, self.age_memberships, "middle", age_value),
            ("age", "old"): self._membership(self.age_universe, self.age_memberships, "old", age_value),
            ("chol", "low"): self._membership(self.chol_universe, self.chol_memberships, "low", chol_value),
            ("chol", "medium"): self._membership(self.chol_universe, self.chol_memberships, "medium", chol_value),
            ("chol", "high"): self._membership(self.chol_universe, self.chol_memberships, "high", chol_value),
            ("thalach", "low"): self._membership(
                self.thalach_universe, self.thalach_memberships, "low", thalach_value
            ),
            ("thalach", "medium"): self._membership(
                self.thalach_universe, self.thalach_memberships, "medium", thalach_value
            ),
            ("thalach", "high"): self._membership(
                self.thalach_universe, self.thalach_memberships, "high", thalach_value
            ),
        }

        aggregated = np.zeros_like(self.risk_universe, dtype=float)
        for antecedent_a, antecedent_b, antecedent_c, consequence in self.rules:
            rule_strength = min(evidence[antecedent_a], evidence[antecedent_b])
            if antecedent_c is not None:
                rule_strength = min(rule_strength, evidence[antecedent_c])
            aggregated = np.fmax(
                aggregated,
                np.fmin(rule_strength, self.risk_memberships[consequence]),
            )

        if np.allclose(aggregated, 0):
            return 0.5

        score = float(fuzz.defuzz(self.risk_universe, aggregated, "centroid"))
        return max(0.0, min(1.0, score))

    def predict_proba(self, dataframe):
        return np.array([self.predict_one(row) for row in dataframe.to_dict(orient="records")])


def get_fuzzy_model_path():
    return os.path.join(get_project_root(), "models", "fuzzy_model.pkl")


def train_fuzzy_model():
    try:
        model = FuzzyRiskModel()
        os.makedirs(os.path.dirname(get_fuzzy_model_path()), exist_ok=True)
        joblib.dump(model, get_fuzzy_model_path())
        return model
    except Exception as exc:
        raise RuntimeError(f"Unable to build fuzzy logic model: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to build the fuzzy model.")
