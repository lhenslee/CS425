from read import DataSet
from svc import SVC
import sys


# Run a report for each set of data
problem_names = ['Ionosphere', 'Vowel Context', 'Satellites']
for i in range(3):
    print('Performing Report for', problem_names[i])
    data = DataSet(i)
    svc = SVC(data)
    svc.fit_clf()
    svc.report_c_mat('training')
    svc.report_c_mat('test')
    print()

