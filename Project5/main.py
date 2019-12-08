from read import DataSet
from svc import SVC
import sys


seed = None
if len(sys.argv) == 3:
    seed = sys.argv[2]
data = DataSet(int(sys.argv[1]), seed_no=seed)
svc = SVC(data)
svc.fit_clf()
svc.report_c_mat('training')
svc.report_c_mat('validation')
svc.report_c_mat('test')

