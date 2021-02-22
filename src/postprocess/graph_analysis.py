import random
import numpy as np
import pickle

class MaxCutImplementations:
    def __init__(self, cfm):
        self.n = len(cfm)
        self.cfm = cfm
            
    def compareNodes(self,i,j):
        if i < 0 or i >= self.n or j < 0 or j >= self.n:
            raise ValueError("Node is invalid. Index is too big/small.")
        return self.cfm[i][j] + self.cfm[j][i]
    
    def maxCutValue(self,a_cut_side,b_cut_side):
        cut_val = 0
        for item1 in a_cut_side:
            for item2 in b_cut_side:
                cut_val += self.compareNodes(item1,item2)
        return cut_val

    def greedyMaxCut(self):
        # Randomizing Order
        to_shuffle = [i for i in range(self.n)]
        random.shuffle(to_shuffle)

        # Instantiating Sets
        a = set()
        b = set()

        # Deciding where to place
        for node in to_shuffle:
            if len(a) == 0 and len(b) == 0:
                a.add(node)
            else:
                # First Candidate
                a.add(node)
                left = self.maxCutValue(a,b)
                a.remove(node)
                # Second Candidate
                b.add(node)
                right = self.maxCutValue(a,b)
                if left > right:
                    b.remove(node)
                    a.add(node)
        return a, b

    def approxMaxCut(self):
        # Randomizing Order
        to_shuffle = [i for i in range(self.n)]
        random.shuffle(to_shuffle)

        # Creating Candidate Solution (that will be iterated upon)
        n = len(to_shuffle)
        a = set(to_shuffle[:int(n/2)])
        b = set(to_shuffle[int(n/2):])
        cut_val = self.maxCutValue(a,b)
        candidate_cut_val = -1

        while candidate_cut_val != cut_val:
            if candidate_cut_val != -1:
                cut_val = candidate_cut_val
            else:
                candidate_cut_val = cut_val
            candidate_a = set()
            candidate_b = set()
            for node in a:
                a.remove(node)
                b.add(node)
                temp_mcv = self.maxCutValue(a,b)
                if temp_mcv > candidate_cut_val:
                    candidate_cut_val = temp_mcv
                    candidate_a = set(a)
                    candidate_b = set(b)
                a.add(node)
                b.remove(node)
            for node in b:
                b.remove(node)
                a.add(node)
                temp_mcv = self.maxCutValue(a,b)
                if temp_mcv > candidate_cut_val:
                    candidate_cut_val = temp_mcv
                    candidate_a = set(a)
                    candidate_b = set(b)
                b.add(node)
                a.remove(node)
            if len(candidate_a) > 0 and len(candidate_b) > 0:
                a = set(candidate_a)
                b = set(candidate_b)
        return a,b

## THERE ARE FOUR GRAPH SETUP CHOICES ##
def single_cfm(base_cfms,num=20):
    set_partitions = []
    for i in range(num):
        # Full CFM
        c = MaxCutImplementations(base_cfms[i])
        a,b = c.approxMaxCut()
        print("Partition Base Epoch {i}:".format(i=(i+1)*5), a, b)
        set_partitions.append(a)
    return set_partitions

def dual_cfm(base_cfms,num=20):
    def tri_matrix(cfm,lower=False):
        n = len(cfm)
        result = []
        for i in range(n):
            result.append([0]*n)
            for j in range(n):
                if i < j and not lower:
                    result[i][j] = cfm[i][j]
                elif i > j and lower:
                    result[i][j] = cfm[i][j]
        return result
    
    set_partitions = []
    for i in range(num):
        # Lower CFM
        c = MaxCutImplementations(tri_matrix(base_cfms[i],lower=True))
        a,b = c.approxMaxCut()
        set_partitions.append(a)
        print("Lower Triangular Partition Base Epoch {i}:".format(i=(i+1)*5), a, b)
        # Upper CFM
        c = MaxCutImplementations(tri_matrix(base_cfms[i],lower=False))
        a,b = c.approxMaxCut()
        print("Upper Triangular Partition at Base Epoch {i}:".format(i=(i+1)*5), a, b)
        set_partitions.append(a)
    return set_partitions

def race_cfms(base_cfms,num=20):
    def race_connected_cfm(cfm):
        n = len(cfm)
        for row in range(n):
            for column in range(n):
                if column % 5 != row % 5:
                    cfm[row][column] = 0
        return cfm
    set_partitions = []
    for i in range(num):
        # Full CFM
        c = MaxCutImplementations(race_connected_cfm(base_cfms[i]))
        a,b = c.approxMaxCut()
        print("Partition Base Epoch {i}:".format(i=(i+1)*5), a, b)
        set_partitions.append(a)
    return set_partitions

def fully_connected_race_age_cfm(base_cfms,num=20):
    set_partitions = []
    for i in range(num):
        # Full CFM
        c = MaxCutImplementations(base_cfms[i])
        a,b = c.approxMaxCut()
        print("Partition Base Epoch {i}:".format(i=(i+1)*5), a, b)
        set_partitions.append(a)
    return set_partitions


def graph_pipeline(base_cfms_path, partitions_path=None, status=0):
    base_cfms = np.load(base_cfms_path,allow_pickle=True)
    num_iter = 20
    print(base_cfms[0])
    ## PICK YOUR GRAPH SETUP VIA STATUS PARAMETER ##
    if status == 0:
        set_partitions = single_cfm(base_cfms,num_iter)
    elif status == 1:
        set_partitions = dual_cfm(base_cfms,num_iter)
    elif status == 2:
        set_partitions = race_cfms(base_cfms,num_iter)
    elif status == 3:
        set_partitions = fully_connected_race_age_cfm(base_cfms,num_iter)
    else:
        raise Exception("This status isn't valid.")

    # if partitions_path is not None:
    #     with open(partitions_path, "wb") as fp:   #Pickling
    #         pickle.dump(set_partitions, fp)

    return set_partitions