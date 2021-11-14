import numpy as np

code_detected_all = np.asarray([11, 11, 12])
code_size_all = np.asarray([32.1, 45.2, 16.3])

code_detected_all_unique = np.unique(code_detected_all)
indexes_to_select = []
for unique_index in range(code_detected_all_unique.size):
    code_detected_all_unique_tmp = code_detected_all_unique[unique_index]
    indexes = np.nonzero(code_detected_all_unique_tmp == code_detected_all)[0]
    if indexes.size == 1:
        indexes_to_select.append(indexes[0])
    else:
        index_max = np.argmax(code_size_all[indexes])
        indexes_to_select.append(indexes[index_max])

indexes_to_select = np.asarray(indexes_to_select)

code_detected_all = code_detected_all[indexes_to_select]