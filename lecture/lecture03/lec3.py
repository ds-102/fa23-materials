import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# These make our figures bigger
plt.rcParams['figure.figsize'] = (6, 4.5)
plt.rcParams['figure.dpi'] = 100

def show_p_values(p_values, threshold = None, show_labels = False):
    """Shows p-values with threshold and optionally labels (TP/FP/etc.)"""
    pv = p_values.copy()
    hue = None
    if threshold is not None and show_labels:
        
        decisions = pv['pvalue'] > threshold
        reality = pv['is_alternative']
        pv.loc[decisions & reality, 'result'] = 'TP'
        pv.loc[~decisions & reality, 'result'] = 'FN'
        pv.loc[decisions & ~reality, 'result'] = 'FP'
        pv.loc[~decisions & ~reality, 'result'] = 'TN'
        hue = 'result'
    sns.stripplot(
        data=pv, x='pvalue', y='is_alternative', 
        alpha = 0.8, order = [0, 1], hue=hue, orient = "h",
    )
    if threshold is not None:
        plt.axvline(threshold, color='black')
    plt.title(f'P-values and results using p-value threshold {threshold}')