import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.absolute()))

from auxiliary_partition.config import cfm_dir, graphs_dir

from src.postprocess.cfm_to_gephi import to_gephi


def age_from_status(s):
    return int(s/5)
def ethnicity_from_status(s):
    return s % 5

def empty_sub_cfms():
    # Age
    a_cfm = []
    for _ in range(3):
        a_cfm.append([0]*3)
    # Race
    r_cfm = []
    for _ in range(5):
        r_cfm.append([0]*5)
    return a_cfm,r_cfm

def create_sub_cfms(cfm):
    a_cfm,r_cfm = empty_sub_cfms()
    for i in range(15):
        for j in range(15):
            new_i = age_from_status(i)
            new_j = age_from_status(j)
            a_cfm[new_i][new_j] += cfm[i][j]
            
            new_i = ethnicity_from_status(i)
            new_j = ethnicity_from_status(j)
            r_cfm[new_i][new_j] += cfm[i][j]
    return a_cfm,r_cfm

def cfm_accuracy(cfm):
    correct = 0
    denom = 0
    for i in range(len(cfm)):
        correct += cfm[i][i]
        denom += sum(cfm[i])
    return correct/denom

def age_accuracy_by_race(cfm):
    # Given an actual race, is predicted age correct? (race does not have to be)
    # I is race, J is age
    measurements = sum([sum(row) for row in cfm])
    correct = 0
    for i in range(15):
        for j in range(15):
            if i < 5:
                if j < 5:
                    correct += cfm[i][j]
            elif i < 10:
                if j >= 5 and j < 10:
                    correct += cfm[i][j]
            else:
                if j >= 10:
                    correct += cfm[i][j]
    avg_acc = correct/measurements
    var_num = 0

    for i in range(5):
        denom = 0
        num = 0
        for j in range(3):
            for k in range(5*3):
                if k >= j*5 and k < (j+1)*5:
                    num += cfm[k][5*j+i]
                denom += cfm[k][5*j+i]
        var_num += denom*(num/denom - avg_acc)*(num/denom - avg_acc)
    weighted_variance = var_num / measurements
    return avg_acc, weighted_variance

# Macro F1: Arithmetic Mean of Race-Specific F1 scores 
# F1 Score: Harmonic Mean of Precision + Recall
def macro_f1_score_by_race(cfm):
    def f1_denominators(cfm,i):
        n = len(cfm)
        # Recall is sum over column
        # Precision is sum over row
        recall_denominator = 0
        precision_denominator = 0
        for j in range(n):
            precision_denominator += cfm[i][j]
            recall_denominator += cfm[j][i]
        return recall_denominator, precision_denominator

    def age_f1(cfm,i):
        race_recall_denominator = 0
        race_precision_denominator = 0
        race_numerator = 0

        for j in range(3):
            race_numerator += cfm[i+5*j][i+5*j]
            recall_denominator, precision_denominator = f1_denominators(cfm,i+5*j)
            race_recall_denominator += recall_denominator
            race_precision_denominator += precision_denominator
        
        recall = race_numerator / race_recall_denominator
        precision = race_numerator / race_precision_denominator

        if recall + precision == 0:
            return 0

        return 2 * (precision * recall)/(precision + recall)

    NUM_RACES = 5
    macro_f1_sum = 0
    for i in range(NUM_RACES):
        macro_f1_sum += age_f1(cfm,i)
    return macro_f1_sum / NUM_RACES


if __name__ == "__main__":
    # Evaluate Accuracy of Base Models (5-100)
    base_cfms = np.load(cfm_dir / 'base_cfms.npy',allow_pickle=True)
    print("Loaded CFMs.")
    x = []
    a_r_var = []
    a_acc = []
    macro_f1_vals = []

    for i in range(len(base_cfms)):
        cfm = base_cfms[i]
        to_gephi(cfm,cfm_dir / "{}.csv".format(i))
        a_cfm,r_cfm = create_sub_cfms(cfm)
        to_gephi(a_cfm,cfm_dir / "{}_age.csv".format(i))
        to_gephi(r_cfm,cfm_dir / "{}_race.csv".format(i))
        x.append((i+1)*5)
        acc, var = age_accuracy_by_race(cfm)
        macro_f1 = macro_f1_score_by_race(cfm)
        a_r_var.append(var)
        a_acc.append(acc)
        macro_f1_vals.append(macro_f1)
    
    plt.scatter(x,a_r_var)
    plt.title("Original: Statistical Parity of Age Accuracy (by Race) - 5 Classes")
    plt.savefig(graphs_dir / "og_a_r_acc.jpg")
    plt.show()
    plt.scatter(x,a_acc)
    plt.title("Original: Age Accuracy - 3 Classes")
    plt.savefig(graphs_dir / "og_a_acc.jpg")
    plt.show()
    plt.scatter(x,macro_f1_vals)
    plt.title("Original: Macro F1 (by Race) - 5 Classes")
    plt.savefig(graphs_dir / "og_macro_f1_vals.jpg")
    plt.show()

