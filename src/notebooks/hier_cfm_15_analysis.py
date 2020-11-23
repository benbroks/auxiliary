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


if __name__ == "__main__":
    # Evaluate Accuracy of Base Models (5-100)
    base_cfms = np.load(cfm_dir / 'final_cfms_15.npy',allow_pickle=True)
    print("Loaded CFMs.")
    x = []
    a_r_var = []
    a_acc = []

    for i in range(len(base_cfms)):
        cfm = base_cfms[i]
        to_gephi(cfm,cfm_dir / "{}_full_v2.csv".format(i))
        a_cfm,r_cfm = create_sub_cfms(cfm)
        to_gephi(a_cfm,cfm_dir /"{}_age_v2.csv".format(i))
        to_gephi(r_cfm,cfm_dir /"{}_race_v2.csv".format(i))
        x.append((i+1)*5)
        acc, var = age_accuracy_by_race(cfm)
        a_r_var.append(var)
        a_acc.append(acc)
    
    plt.scatter(x,a_r_var)
    plt.title("Hierarchical (2 Output Aux): Statistical Parity of Age Accuracy (by Race) - 5 Classes")
    plt.savefig(graphs_dir / "a_r_acc_v2.jpg")
    plt.show()
    plt.scatter(x,a_acc)
    plt.title("Hierarchical (2 Output Aux): Age Accuracy - 3 Classes")
    plt.savefig(graphs_dir / "a_acc_v2.jpg")
    plt.show()

