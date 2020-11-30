import numpy as np
import matplotlib.pyplot as plt

def age_from_status(s):
    return int(s/10)
def ethnicity_from_status(s):
    s = s % 10
    return int(s/2)
def gender_from_status(s):
    return s % 2

def empty_sub_cfms():
    # Age
    a_cfm = []
    for _ in range(3):
        a_cfm.append([0]*3)
    # Race
    r_cfm = []
    for _ in range(5):
        r_cfm.append([0]*5)
    # Gender
    g_cfm = []
    for _ in range(2):
        g_cfm.append([0]*2)
    return a_cfm,r_cfm,g_cfm

def create_sub_cfms(cfm):
    a_cfm,r_cfm,g_cfm = empty_sub_cfms()
    for i in range(30):
        for j in range(30):
            new_i = age_from_status(i)
            new_j = age_from_status(j)
            a_cfm[new_i][new_j] += cfm[i][j]
            
            new_i = ethnicity_from_status(i)
            new_j = ethnicity_from_status(j)
            r_cfm[new_i][new_j] += cfm[i][j]
            
            new_i = gender_from_status(i)
            new_j = gender_from_status(j)
            g_cfm[new_i][new_j] += cfm[i][j]
    return a_cfm,r_cfm,g_cfm

def cfm_accuracy(cfm):
    correct = 0
    denom = 0
    for i in range(len(cfm)):
        correct += cfm[i][i]
        denom += sum(cfm[i])
    return correct/denom

def fairness_from_cfm(cfm):
    measurements = sum([sum(row) for row in cfm])
    correct_pred = sum([cfm[i][i] for i in range(len(cfm))])
    if measurements == 0:
        return 0,0,0
    avg_acc = correct_pred / measurements
    
    ## PRECISION ## 
    sum_precision = 0
    for i in range(len(cfm)):
        if sum(cfm[i]) == 0:
            sum_precision += avg_acc*avg_acc*sum(cfm[i])
        else:
            sum_precision += sum(cfm[i])*(avg_acc - cfm[i][i] / sum(cfm[i]))**2
    var_of_precision = sum_precision/measurements
    
    ## RECALL ##
    sum_recall = 0
    for i in range(len(cfm)):
        # Column Sum
        column_sum = sum([row[i] for row in cfm])
        if column_sum == 0:
            sum_recall += avg_acc*avg_acc*column_sum
        else:
            sum_recall += (avg_acc - cfm[i][i] / column_sum)**2
    var_of_recall = sum_recall/measurements
    
    return avg_acc, var_of_precision, var_of_recall

def update(a_acc,a_prec,a_recall,r_acc,r_prec,r_recall,g_acc,g_prec,g_recall,a_cfm,r_cfm,g_cfm):
    a_a,a_p,a_r = fairness_from_cfm(a_cfm)
    r_a,r_p,r_r = fairness_from_cfm(r_cfm)
    g_a,g_p,g_r = fairness_from_cfm(g_cfm)
    a_acc.append(a_a)
    a_prec.append(a_p)
    a_recall.append(a_r)
    r_acc.append(r_a)
    r_prec.append(r_p)
    r_recall.append(r_r)
    g_acc.append(g_a)
    g_prec.append(g_p)
    g_recall.append(g_r)
    return a_acc,a_prec,a_recall,r_acc,r_prec,r_recall,g_acc,g_prec,g_recall

if __name__ == "__main__":
    # Evaluate Accuracy of Base Models (5-100)
    base_cfms = np.load('confusionMatrices/final_cfms.npy',allow_pickle=True) + np.load('confusionMatrices/final_cfms_15.npy',allow_pickle=True)
    print("Loaded CFMs.")
    x = []
    a_acc = []
    a_prec = []
    a_recall = []
    r_acc = []
    r_prec = []
    r_recall = []
    g_acc = []
    g_prec = []
    g_recall = []

    for i in range(len(base_cfms)):
        cfm = base_cfms[i]
        # Couple Errors Baked In
        if len(cfm) == 30:
            x.append((i+1)*5)
            a_cfm,r_cfm,g_cfm = create_sub_cfms(cfm)
            a_acc,a_prec,a_recall,r_acc,r_prec,r_recall,g_acc,g_prec,g_recall = update(a_acc,a_prec,a_recall,r_acc,r_prec,r_recall,g_acc,g_prec,g_recall,a_cfm,r_cfm,g_cfm)
    plt.scatter(x,a_acc)
    plt.title("Age Accuracy - 3 Classes")
    plt.savefig("plots/hierarchical/a_acc.jpg")
    plt.show()
    plt.scatter(x,a_prec)
    plt.title("Statistical Parity of Age - Variance of Precision")
    plt.savefig("plots/hierarchical/a_prec.jpg")
    plt.show()
    plt.scatter(x,a_recall)
    plt.title("Statistical Parity of Age - Variance of Recall")
    plt.savefig("plots/hierarchical/a_recall.jpg")
    plt.show()
    plt.scatter(x,r_acc)
    plt.title("Race Accuracy - 5 Classes")
    plt.savefig("plots/hierarchical/r_acc.jpg")
    plt.show()
    plt.scatter(x,r_prec)
    plt.title("Statistical Parity of Race - Variance of Precision")
    plt.savefig("plots/hierarchical/r_prec.jpg")
    plt.show()
    plt.scatter(x,r_recall)
    plt.title("Statistical Parity of Race - Variance of Recall")
    plt.savefig("plots/hierarchical/r_recall.jpg")
    plt.show()
    plt.scatter(x,g_acc)
    plt.title("Gender Accuracy - 2 Classes")
    plt.savefig("plots/hierarchical/g_acc.jpg")
    plt.show()
    plt.scatter(x,g_prec)
    plt.title("Statistical Parity of Gender - Variance of Precision")
    plt.savefig("plots/hierarchical/g_prec.jpg")
    plt.show()
    plt.scatter(x,g_recall)
    plt.title("Statistical Parity of Gender - Variance of Precision")
    plt.savefig("plots/hierarchical/g_recall.jpg")
    plt.show()

