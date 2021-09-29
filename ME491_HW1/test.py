from prob1_value_iter import get_optim_value, get_optim_policy, get_optim_path
import prob1_value_iter
import numpy as np
import pickle

D = np.genfromtxt('HW1_adjacency_matrix.csv', delimiter=',')
D = D.astype(int)

num_nodes = len(D)

# test set generation
test_cases = [] ## tuples of gamma, depart_pos, terminal_pos, optim_value, optim_policy, optim_path
for gamma in [0.99, 0.9, 0.8]:
    for depart_pos in range(num_nodes):
        for terminal_pos in range(num_nodes):
            optim_value = get_optim_value(D, threshold=0.001, gamma=gamma, depart_pos=depart_pos, terminal_pos=terminal_pos)
            optim_policy = get_optim_policy(D, optim_value, depart_pos, terminal_pos, gamma)
            optim_path = get_optim_path(D, optim_value, depart_pos, terminal_pos, gamma)
            if (depart_pos != terminal_pos and len(optim_path) <= num_nodes):
                test_cases.append([gamma, depart_pos, terminal_pos, optim_value, optim_policy, optim_path])

with open("test_cases.txt", "wb") as fp:
    pickle.dump(test_cases, fp)


# test
def array_equal(array1, array2):
    return np.max(np.abs(array1-array2))<0.001

with open("test_cases.txt", "rb") as fp:
    test_cases = pickle.load(fp)
for test_case in test_cases:
    gamma, depart_pos, terminal_pos, optim_value_mjh, optim_policy_mjh, optim_path_mjh = test_case
    optim_value = get_optim_value(D, threshold=0.001, gamma=gamma, depart_pos=depart_pos, terminal_pos=terminal_pos).astype(float)
    optim_policy = get_optim_policy(D, optim_value, depart_pos, terminal_pos, gamma).astype(int)
    optim_path = get_optim_path(D, optim_value, depart_pos, terminal_pos, gamma).astype(int)
    if(not (array_equal(optim_value_mjh, optim_value) and array_equal(optim_policy_mjh, optim_policy) and array_equal(optim_value_mjh, optim_value))):
        print("gamma = {}, d = {}, t = {}".format(gamma, depart_pos, terminal_pos))
        print("optim_path by mjh:{}\noptim_path by you:{}".format(optim_path_mjh, optim_path))
        print("optim_policy by mjh:{}\noptim_policy by you:{}".format(optim_policy_mjh, optim_policy))
        print("optim_value by mjh:{}\noptim_value by you:{}\n".format(optim_value_mjh, optim_value))
print("test done!")