from read import DataSet
from svc import SVC
import sys


# Run a report for each set of data
for i in range(3):
    data = DataSet(i)
    svc = SVC(data)
    svc.fit_clf()
    svc.report_c_mat('training')
    svc.report_c_mat('test')

