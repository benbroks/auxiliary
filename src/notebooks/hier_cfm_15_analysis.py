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
        if denom == 0:
            to_add = 0
        else:
            to_add = denom*(num/denom - avg_acc)*(num/denom - avg_acc)
        var_num += to_add
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
        
        if race_recall_denominator == 0:
            recall = 0
        else:
            recall = race_numerator / race_recall_denominator
        if race_precision_denominator == 0:
            precision = 0
        else:
            precision = race_numerator / race_precision_denominator

        if recall + precision == 0:
            return 0

        return 2 * (precision * recall)/(precision + recall)

    NUM_RACES = 5
    macro_f1_sum = 0
    for i in range(NUM_RACES):
        macro_f1_sum += age_f1(cfm,i)
    return macro_f1_sum / NUM_RACES

def distribution_collection_instead_of_variance(cfm):
    # Given an actual race, is predicted age correct? (race does not have to be)
    # I is race, J is age
    age_accuracies_by_race = []
    for i in range(5):
        denom = 0
        num = 0
        for j in range(3):
            for k in range(5*3):
                if k >= j*5 and k < (j+1)*5:
                    num += cfm[k][5*j+i]
                denom += cfm[k][5*j+i]
        if denom != 0:
            age_accuracies_by_race.append(num/denom)
    return age_accuracies_by_race

def f1_collection_instead_of_variance(cfm):
    f1_by_race = []
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
        
        if race_recall_denominator == 0:
            recall = 0
        else:
            recall = race_numerator / race_recall_denominator
        if race_precision_denominator == 0:
            precision = 0
        else:
            precision = race_numerator / race_precision_denominator

        if recall + precision == 0:
            return 0

        return 2 * (precision * recall)/(precision + recall)

    NUM_RACES = 5
    for i in range(NUM_RACES):
        f1_by_race.append(age_f1(cfm,i))
    return f1_by_race


def get_stats_for_cfm_list(cfm_list,n=16):
    x = []
    a_r_var = []
    a_acc = []
    macro_f1_vals = []

    f1_scatter_x = []
    f1_scatter_accuracies = []
    scatter_x = []
    scatter_accuracies = []

    for i in range(n):
        cfm = cfm_list[i]
        to_gephi(cfm,cfm_dir / "{}.csv".format(i))
        a_cfm,r_cfm = create_sub_cfms(cfm)
        to_gephi(a_cfm,cfm_dir / "{}_age.csv".format(i))
        to_gephi(r_cfm,cfm_dir / "{}_race.csv".format(i))
        x.append((i+1)*5)
        acc, var = age_accuracy_by_race(cfm)
        macro_f1 = macro_f1_score_by_race(cfm)
        age_accuracies_by_race = distribution_collection_instead_of_variance(cfm)
        f1_scores_by_race = f1_collection_instead_of_variance(cfm)

        f1_scatter_x = f1_scatter_x + [(i+1)*5]*len(f1_scores_by_race)
        f1_scatter_accuracies = f1_scatter_accuracies + f1_scores_by_race
        scatter_x = scatter_x + [(i+1)*5]*len(age_accuracies_by_race)
        scatter_accuracies = scatter_accuracies + age_accuracies_by_race
        a_r_var.append(var)
        a_acc.append(acc)
        macro_f1_vals.append(macro_f1)
    return x, a_r_var, a_acc, macro_f1_vals, scatter_x, scatter_accuracies, f1_scatter_x, f1_scatter_accuracies
    


def modified_cfm_race_age_main():
    # Evaluate Accuracy of Base Models (5-100)
    hierarchical_30_cfms = np.load(cfm_dir / 'final_cfms_30.npy',allow_pickle=True)
    hierarchical_15_cfms_pt1 = np.load(cfm_dir / 'final_cfms_one_15.npy',allow_pickle=True)
    hierarchical_15_cfms_pt2 = np.load(cfm_dir / 'final_cfms_full_15.npy',allow_pickle=True)
    base_cfms = np.load(cfm_dir / 'base_cfms.npy',allow_pickle=True)
    print("Loaded CFMs.")
    x, hier_30_a_r_var, hier_30_a_acc, hier_30_macro_f1_vals, hier_30_x, hier_30_scatter, hier_30_f1_x, hier_30_f1_scatter = get_stats_for_cfm_list(hierarchical_30_cfms)
    _, hier_15_1_a_r_var, hier_15_1_a_acc, hier_15_1_macro_f1_vals, hier_15_1_x, hier_15_1_scatter, hier_15_1_f1_x, hier_15_1_f1_scatter = get_stats_for_cfm_list(hierarchical_15_cfms_pt1)
    _, hier_15_2_a_r_var, hier_15_2_a_acc, hier_15_2_macro_f1_vals, hier_15_2_x, hier_15_2_scatter, hier_15_2_f1_x, hier_15_2_f1_scatter = get_stats_for_cfm_list(hierarchical_15_cfms_pt2)
    _, base_a_r_var, base_a_acc, base_macro_f1_vals, base_x, base_scatter, base_f1_x, base_f1_scatter = get_stats_for_cfm_list(base_cfms)
    

    plt.plot(x,base_a_r_var,label="Base")
    plt.plot(x,hier_15_1_a_r_var,label="Hierarchical_15 - Partition 1")
    plt.plot(x,hier_15_2_a_r_var,label="Hierarchical_15 - Partition 2")
    plt.plot(x,hier_30_a_r_var,label="Hierarchical_30")

    plt.title("UTKFace: Statistical Parity of Age Recall (by Race) - 5 Classes")
    plt.xlabel("Epochs")
    plt.ylabel("Variance")
    plt.legend()
    plt.savefig(graphs_dir / "combined_15_a_r_acc.jpg")
    plt.show()

    plt.plot(x,base_a_acc,label="Base")
    plt.plot(x,hier_15_1_a_acc,label="Hierarchical_15 - Partition 1")
    plt.plot(x,hier_15_2_a_acc,label="Hierarchical_15 - Partition 2")
    plt.plot(x,hier_30_a_acc,label="Hierarchical_30")
    
    plt.title("UTKFace: Age Recall - 3 Classes")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(graphs_dir / "combined_15_a_acc.jpg")
    plt.show()

    plt.plot(x,base_macro_f1_vals,label="Base")
    plt.plot(x,hier_15_1_macro_f1_vals,label="Hierarchical_15 - Partition 1")
    plt.plot(x,hier_15_2_macro_f1_vals,label="Hierarchical_15 - Partition 2")
    plt.plot(x,hier_30_macro_f1_vals,label="Hierarchical_30")

    plt.title("UTKFace: Macro F1 (by Race) - 5 Classes")
    plt.xlabel("Epochs")
    plt.ylabel("Macro F1 Score")
    plt.legend()
    plt.savefig(graphs_dir / "combined_15_macro_f1_vals.jpg")
    plt.show()

    plt.scatter(base_x, base_scatter,label="Base")
    plt.scatter(hier_15_1_x, hier_15_1_scatter,label="Hierarchical_15 - Partition 1")
    plt.scatter(hier_15_2_x, hier_15_2_scatter,label="Hierarchical_15 - Partition 2")
    plt.scatter(hier_30_x, hier_30_scatter,label="Hierarchical_30")

    plt.title("UTKFace: Age Recall Distribution by Race Label")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(graphs_dir / "combined_15_age_acc_dist.jpg")
    plt.show()

    plt.scatter(base_f1_x, base_f1_scatter,label="Base")
    plt.scatter(hier_15_1_f1_x, hier_15_1_f1_scatter,label="Hierarchical_15 - Partition 1")
    plt.scatter(hier_15_2_f1_x, hier_15_2_f1_scatter,label="Hierarchical_15 - Partition 2")
    plt.scatter(hier_30_f1_x, hier_30_f1_scatter,label="Hierarchical_30")

    plt.title("UTKFace: F1 Score by Race Label")
    plt.xlabel("Epochs")
    plt.ylabel("F1")
    plt.legend()
    plt.savefig(graphs_dir / "combined_15_f1_dist.jpg")
    plt.show()

    plt.scatter(hier_15_1_x, hier_15_1_scatter,label="Hierarchical_15 - Partition 1")
    plt.scatter(hier_15_2_x, hier_15_2_scatter,label="Hierarchical_15 - Partition 2")

    plt.title("UTKFace: Age Recall Distribution by Race Label")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(graphs_dir / "just_15_age_acc_dist.jpg")
    plt.show()

if __name__ == "__main__":
    modified_cfm_race_age_main()

