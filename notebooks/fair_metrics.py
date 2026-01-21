def get_equalized_odds(df, group_column, group_value, true_col, pred_col):
    """
    Computes TPR and FPR for a single group.

    Parameters:
    - df: dataframe containing predictions and true labels
    - group_column: the sensitive feature column (e.g., 'HeatingType')
    - group_value: specific category to evaluate (e.g., 'Electric')
    - true_col: column containing true labels (e.g., 'y_true')
    - pred_col: column containing predicted labels (e.g., 'y_pred' or 'y_pred_adjusted')

    Returns:
    - TPR: True Positive Rate
    - FPR: False Positive Rate
    """

    group = df[df[group_column] == group_value]

    TP = ((group[true_col] == 1) & (group[pred_col] == 1)).sum()
    FN = ((group[true_col] == 1) & (group[pred_col] == 0)).sum()
    FP = ((group[true_col] == 0) & (group[pred_col] == 1)).sum()
    TN = ((group[true_col] == 0) & (group[pred_col] == 0)).sum()

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return TPR, FPR


def eo_difference(df, group_column, group_a, group_b, true_col, pred_col):
    """
    Computes Equalized Odds (EO) difference between two groups.

    Parameters:
    - df: dataframe containing predictions and true labels
    - group_column: sensitive feature column (e.g., 'HeatingType')
    - group_a: first category (e.g., 'Electric')
    - group_b: second category (e.g., 'Gas')
    - true_col: column with true labels (e.g., 'y_true')
    - pred_col: column with predicted labels (e.g., 'y_pred' or 'y_pred_adjusted')

    Returns:
    A dictionary with:
    - TPR_diff: absolute difference in TPR
    - FPR_diff: absolute difference in FPR
    - GroupA: {'TPR': value, 'FPR': value}
    - GroupB: {'TPR': value, 'FPR': value}
    """

    TPR_a, FPR_a = get_equalized_odds(df, group_column, group_a, true_col, pred_col)
    TPR_b, FPR_b = get_equalized_odds(df, group_column, group_b, true_col, pred_col)

    return {
        'TPR_diff': abs(TPR_a - TPR_b),
        'FPR_diff': abs(FPR_a - FPR_b),
        'GroupA': {'TPR': TPR_a, 'FPR': FPR_a},
        'GroupB': {'TPR': TPR_b, 'FPR': FPR_b}
    }


def get_selection_rate(df, group_column, group_value, pred_col):
    """
    Computes the selection rate for a given group.
    Selection rate = percentage of positive predictions (1s).

    Parameters:
    - df: your evaluation dataframe
    - group_column: the sensitive feature (e.g. 'HeatingType')
    - group_value: the specific category (e.g. 'Electric')
    - pred_col: the column containing model predictions (e.g. 'y_pred')
    """
    group = df[df[group_column] == group_value]
    if len(group) == 0:
        return 0
    return group[pred_col].mean()


def disparate_impact_ratio(df, group_column, protected_value, privileged_value, pred_col):
    """
    Computes DIR = protected_group_rate / privileged_group_rate.

    Students must pass:
    - pred_col: the name of the column that holds predictions
    """
    protected_rate = get_selection_rate(df, group_column, protected_value, pred_col)
    privileged_rate = get_selection_rate(df, group_column, privileged_value, pred_col)

    if privileged_rate == 0:
        return 0

    return protected_rate / privileged_rate


def demographic_parity_difference(df, group_column, group_a, group_b, pred_col):
    """
    Computes DP difference = |selection_rate_a - selection_rate_b|.

    Students must pass the prediction column name used in their model.
    """
    rate_a = get_selection_rate(df, group_column, group_a, pred_col)
    rate_b = get_selection_rate(df, group_column, group_b, pred_col)

    return abs(rate_a - rate_b)
