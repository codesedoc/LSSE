import utils.hyperor as hpyeror
import time


hr = hpyeror.Hyperor(study_path = 'result/LSSE/optuna', study_name = 'LSSE')

hr.show_best_trial()
del hr
# time.sleep(5)
pass
