from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    tp = sum(1 for expected, actual in zip(expected_results, actual_results) if expected and actual) # true positive
    fn = sum(1 for expected, actual in zip(expected_results, actual_results) if expected and not actual) # false negative
    fp = sum(1 for expected, actual in zip(expected_results, actual_results) if not expected and actual) # false positive

    if (tp + fp) == 0:
        precision = 1
        recall = 1
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

    return (precision, recall)

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """

    precision, recall = precision_recall(expected_results, actual_results)
    return 2 * (precision * recall) / (precision + recall)
