import os
import random

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data_download import get_project_root

RANDOM_STATE = 42


def get_ga_feature_path():
    return os.path.join(get_project_root(), "models", "ga_features.pkl")


def _ensure_non_empty(chromosome):
    if np.sum(chromosome) == 0:
        chromosome[np.random.randint(0, len(chromosome))] = 1
    return chromosome


def _tournament_selection(population, scores, tournament_size=3):
    contenders = random.sample(range(len(population)), tournament_size)
    winner_index = max(contenders, key=lambda index: scores[index])
    return population[winner_index].copy()


def _single_point_crossover(parent_a, parent_b, crossover_rate):
    if random.random() > crossover_rate:
        return parent_a.copy(), parent_b.copy()

    crossover_point = random.randint(1, len(parent_a) - 1)
    child_a = np.concatenate([parent_a[:crossover_point], parent_b[crossover_point:]])
    child_b = np.concatenate([parent_b[:crossover_point], parent_a[crossover_point:]])
    return _ensure_non_empty(child_a), _ensure_non_empty(child_b)


def _mutate(chromosome, mutation_rate):
    for feature_index in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[feature_index] = 1 - chromosome[feature_index]
    return _ensure_non_empty(chromosome)


def run_genetic_algorithm(features, target, feature_names):
    try:
        np.random.seed(RANDOM_STATE)
        random.seed(RANDOM_STATE)

        population_size = 30
        generations = 50
        crossover_rate = 0.8
        mutation_rate = 0.05
        feature_count = features.shape[1]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        fitness_cache = {}

        def fitness(chromosome):
            chromosome_key = tuple(int(bit) for bit in chromosome.tolist())
            if chromosome_key in fitness_cache:
                return fitness_cache[chromosome_key]

            selected_indices = np.where(chromosome == 1)[0]
            if len(selected_indices) == 0:
                return 0.0

            classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            subset = features[:, selected_indices]
            score = cross_val_score(
                classifier,
                subset,
                target,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
            ).mean()
            fitness_cache[chromosome_key] = float(score)
            return fitness_cache[chromosome_key]

        population = [
            _ensure_non_empty(np.random.randint(0, 2, feature_count))
            for _ in range(population_size)
        ]

        best_chromosome = population[0].copy()
        best_score = 0.0

        for generation in range(generations):
            scores = [fitness(chromosome) for chromosome in population]
            generation_best_index = int(np.argmax(scores))
            generation_best_score = float(scores[generation_best_index])
            generation_best_chromosome = population[generation_best_index].copy()

            if generation_best_score >= best_score:
                best_score = generation_best_score
                best_chromosome = generation_best_chromosome

            print(
                f"GA generation {generation + 1:02d}/{generations}: "
                f"best_accuracy={generation_best_score:.4f}"
            )

            next_population = [best_chromosome.copy()]
            while len(next_population) < population_size:
                parent_a = _tournament_selection(population, scores, tournament_size=3)
                parent_b = _tournament_selection(population, scores, tournament_size=3)
                child_a, child_b = _single_point_crossover(parent_a, parent_b, crossover_rate)
                next_population.append(_mutate(child_a, mutation_rate))
                if len(next_population) < population_size:
                    next_population.append(_mutate(child_b, mutation_rate))

            population = next_population

        selected_indices = np.where(best_chromosome == 1)[0].tolist()
        os.makedirs(os.path.dirname(get_ga_feature_path()), exist_ok=True)
        joblib.dump(selected_indices, get_ga_feature_path())

        selected_names = [feature_names[index] for index in selected_indices]
        print(f"Selected GA features: {selected_names}")
        return selected_indices
    except Exception as exc:
        raise RuntimeError(f"Unable to run genetic algorithm: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit("Run train.py to execute feature selection.")
