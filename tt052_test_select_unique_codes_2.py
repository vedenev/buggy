import numpy as np
from localize_with_circles_2020_12_29 import select_unique_codes


x_detected_all = np.asarray([1.0, 2.0, 3.0])
y_detected_all = np.asarray([5.0, 6.0, 7.0])
code_detected_all = np.asarray([11, 11, 12])
code_size_all = np.asarray([20, 21, 22])
x_detected_pairs_all = np.asarray([[0.9, 1.1], [1.9, 2.1], [2.9, 3.1]])
y_detected_pairs_all = np.asarray([[4.9, 5.1], [5.9, 6.1], [6.9, 7.1]])

x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                            select_unique_codes(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)

print('x_detected_all =', x_detected_all)
print('y_detected_all =', y_detected_all)
print('code_detected_all =', code_detected_all)
print('code_size_all =', code_size_all)
print('x_detected_pairs_all =', x_detected_pairs_all)
print('y_detected_pairs_all =', y_detected_pairs_all)