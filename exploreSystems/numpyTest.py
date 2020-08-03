import numpy as np
import random
import timeit


#%%
rows = 1000
columns = 1000
testArray = np.ones((rows, columns))
nums = list(reversed(list(range(testArray.size))))
for i in range(rows):
    for j in range(columns):
        testArray[i, j] = nums.pop()

#%% sequenctially access
sequential_access_pairs = []
for i in range(rows):
    for j in range(columns):
        sequential_access_pairs.append( (i, j) )
        # print(i, j)

#%% sequenctially column wise access
sequential_column_wise_access_pairs = []
for i in range(columns):
    for j in range(rows):
        sequential_column_wise_access_pairs.append( (i, j) )
        # print(i, j)



#%% random access
random_access_pairs = []
for i in range(rows):
    for j in range(columns):
        random_access_pairs.append( (random.randint(0, rows - 1), random.randint(0, columns - 1)) )
        # print( (random.randint(0, rows - 1), random.randint(0, columns - 1)) )

#%% run simulation
tmp = 1
sequential_time_list = []
sequential_column_wise_time_list = []
random_time_list = []

def sequencial_access():
    for pair in sequential_access_pairs:
        tmp = testArray[pair[0], pair[1]]

def sequencial_column_wise_access():
    for pair in sequential_column_wise_access_pairs:
        tmp = testArray[pair[0], pair[1]]

def random_access():
    for pair in random_access_pairs:
        tmp = testArray[pair[0], pair[1]]

sequential_time_list.append(timeit.timeit(sequencial_access, number=1000))
sequential_column_wise_time_list.append(timeit.timeit(sequencial_column_wise_access, number=1000))
random_time_list.append(timeit.timeit(random_access, number=1000))

sequential_time_list.append(timeit.timeit(sequencial_access, number=1000))
sequential_column_wise_time_list.append(timeit.timeit(sequencial_column_wise_access, number=1000))
random_time_list.append(timeit.timeit(random_access, number=1000))


print()
print()
print()
print(sequential_time_list)
print(sequential_column_wise_time_list)
print(random_time_list)
print('mean of sequential', sum(sequential_time_list)/len(sequential_time_list))
print('mean of sequential column wise', sum(sequential_column_wise_time_list)/len(sequential_column_wise_time_list))
print('mean of random', sum(random_time_list)/len(random_time_list))

