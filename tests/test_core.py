import numpy as np

from histoset import ConceptHierarchy, conformal_threshold, macro_dice, prediction_sets, set_size, soft_dice


def test_conformal_sets():
    threshold = conformal_threshold([0.90, 0.80, 0.70, 0.60], alpha=0.25)
    probs = np.array([[0.8, 0.2], [0.1, 0.7], [0.1, 0.1]])
    sets = prediction_sets(probs, threshold)
    assert sets.shape == probs.shape
    assert np.all(set_size(sets) >= 1)


def test_hierarchy_aggregation():
    hierarchy = ConceptHierarchy(
        sub_to_explanation=np.array([0, 0, 1]),
        explanation_to_pattern=np.array([0, 1]),
    )
    sub = np.array([[0.2, 0.1], [0.3, 0.4], [0.5, 0.5]])
    expl = hierarchy.sub_to_explanation_probabilities(sub)
    assert np.allclose(expl[0], [0.5, 0.5])
    assert np.allclose(expl[1], [0.5, 0.5])


def test_metrics():
    pred = np.array([0, 1, 1, 2])
    tgt = np.array([0, 1, 2, 2])
    assert 0 <= macro_dice(pred, tgt, [0, 1, 2]) <= 1
    prob = np.eye(3)[pred].T
    soft = np.eye(3)[tgt].T
    assert soft_dice(prob, soft).shape == (3,)
