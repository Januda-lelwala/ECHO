"""Dataset-level ML analysis for discovering repeatable model failure slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree


TaskType = Literal["classification", "transcription"]


@dataclass(frozen=True)
class DiscoveryRecord:
    record_id: str
    features: dict[str, Any]
    prediction: str
    ground_truth: str


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1] / len(reference_words)


def _target(record: DiscoveryRecord, task: TaskType) -> float:
    if task == "transcription":
        return _word_error_rate(record.ground_truth, record.prediction)
    return float(record.prediction.strip().casefold() != record.ground_truth.strip().casefold())


def _clean_features(features: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in features.items():
        if value is None or isinstance(value, (dict, list, tuple, set)):
            continue
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, bool):
            cleaned[name] = int(value)
        elif isinstance(value, (int, float)):
            if np.isfinite(float(value)):
                cleaned[name] = float(value)
        else:
            text = str(value).strip()
            if text and len(text) <= 120:
                cleaned[name] = text
    return cleaned


def _format_rule(feature: str, threshold: float, goes_left: bool) -> str:
    if "=" in feature and 0 <= threshold <= 1:
        field, value = feature.split("=", 1)
        return f"{field} != {value}" if goes_left else f"{field} = {value}"
    operator = "≤" if goes_left else ">"
    return f"{feature} {operator} {threshold:.3g}"


def _leaf_rules(estimator, feature_names: list[str]) -> dict[int, list[str]]:
    tree = estimator.tree_
    rules: dict[int, list[str]] = {}

    def walk(node: int, path: list[str]) -> None:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            rules[node] = path
            return
        feature = feature_names[tree.feature[node]]
        threshold = float(tree.threshold[node])
        walk(tree.children_left[node], [*path, _format_rule(feature, threshold, True)])
        walk(tree.children_right[node], [*path, _format_rule(feature, threshold, False)])

    walk(0, [])
    return rules


def discover_failures(
    records: list[DiscoveryRecord],
    task: TaskType,
    *,
    min_slice_size: int = 5,
    max_depth: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    usable = [
        DiscoveryRecord(
            record_id=record.record_id,
            features=_clean_features(record.features),
            prediction=record.prediction,
            ground_truth=record.ground_truth,
        )
        for record in records
        if record.prediction.strip() and record.ground_truth.strip()
    ]
    usable = [record for record in usable if record.features]

    if len(usable) < 20:
        raise ValueError("At least 20 labeled predictions with usable metadata are required")

    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform([record.features for record in usable])
    targets = np.asarray([_target(record, task) for record in usable], dtype=float)

    if matrix.shape[1] == 0:
        raise ValueError("No usable numeric or categorical metadata features were provided")
    if np.allclose(targets, targets[0]):
        raise ValueError("Failure discovery requires variation in the observed model errors")

    indices = np.arange(len(usable))
    stratify = None
    if task == "classification":
        classes, counts = np.unique(targets, return_counts=True)
        if len(classes) == 2 and counts.min() >= 2:
            stratify = targets

    train_idx, validation_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )
    effective_leaf_size = max(2, min(min_slice_size, max(2, len(train_idx) // 4)))

    if task == "classification":
        estimator = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=effective_leaf_size,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        estimator = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=effective_leaf_size,
            random_state=random_state,
        )

    estimator.fit(matrix[train_idx], targets[train_idx])
    validation_prediction = estimator.predict(matrix[validation_idx])
    if task == "classification":
        validation_score = float(
            balanced_accuracy_score(targets[validation_idx], validation_prediction)
        )
        validation_metric = "balanced_accuracy"
    else:
        validation_score = float(
            mean_absolute_error(targets[validation_idx], validation_prediction)
        )
        validation_metric = "mean_absolute_error"

    feature_names = list(vectorizer.get_feature_names_out())
    rules_by_leaf = _leaf_rules(estimator, feature_names)
    all_leaves = estimator.apply(matrix)
    validation_leaves = estimator.apply(matrix[validation_idx])
    baseline_error = float(targets.mean())
    findings: list[dict[str, Any]] = []

    for leaf, rules in rules_by_leaf.items():
        all_matches = np.flatnonzero(all_leaves == leaf)
        validation_matches = validation_idx[validation_leaves == leaf]
        if len(all_matches) < min_slice_size or len(validation_matches) < 2:
            continue
        slice_error = float(targets[all_matches].mean())
        validation_error = float(targets[validation_matches].mean())
        if slice_error <= baseline_error or validation_error <= baseline_error:
            continue
        findings.append(
            {
                "id": f"leaf-{leaf}",
                "rules": rules or ["All evaluated examples"],
                "sample_count": int(len(all_matches)),
                "error_rate": slice_error,
                "validation_error_rate": validation_error,
                "baseline_error_rate": baseline_error,
                "error_lift": slice_error / baseline_error if baseline_error else None,
                "example_ids": [usable[index].record_id for index in all_matches[:10]],
            }
        )

    findings.sort(key=lambda item: (item["error_lift"] or 0, item["sample_count"]), reverse=True)
    importances = [
        {"feature": feature_names[index], "importance": float(importance)}
        for index, importance in enumerate(estimator.feature_importances_)
        if importance > 0
    ]
    importances.sort(key=lambda item: item["importance"], reverse=True)

    return {
        "task": task,
        "record_count": len(usable),
        "baseline_error_rate": baseline_error,
        "validation": {
            "metric": validation_metric,
            "value": validation_score,
            "sample_count": int(len(validation_idx)),
        },
        "feature_importance": importances[:10],
        "findings": findings[:10],
        "message": (
            "No repeatable high-error slices were found with the current settings."
            if not findings
            else None
        ),
    }
