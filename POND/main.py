
import argparse
import numpy as np
from POND_sgd import POND
import scipy.stats as stats

ap = argparse.ArgumentParser()

ap.add_argument("-tr", required=True)
ap.add_argument("-te", required=True)
ap.add_argument('-r', type=int, required = True)
ap.add_argument('-dim', nargs='+',type=int, required = True)
ap.add_argument('-ne',type = int)
ap.add_argument('-lr',type = float)
ap.add_argument('-bs',type = int)
args = vars(ap.parse_args())



batch_size = args['bs']
if batch_size is None:
    batch_size = 256

lr = args['lr']
if lr is None:
    lr = .001

nepochs = args['ne']
if nepochs is None:
    nepochs = 100


#required arguments
train = np.loadtxt(args['tr'],delimiter = ',')
test =  np.loadtxt(args['te'],delimiter = ',')
size = args['dim']
nmod = len(size)
rank = args['r']


ind = train[:,:-1].astype(int)-1
y = train[:,-1]
ind_test = test[:,:-1].astype(int)-1
y_test = test[:,-1]
print('Data loaded')



U = [stats.truncnorm(-1, 1).rvs([size[i],rank]) for i in range(nmod)]
model = POND(ind, y, U, 100, batch_size, lr, 5)
print('Training')
model.train(nepochs)
y_pred = model.test(ind_test)

rmse = np.sqrt(np.mean(np.power(y_pred - y_test, 2)))
mae = np.mean(np.abs(y_pred - y_test))
print('rmse = ' + str(rmse))
print('mae = ' + str(mae))


y_pred = model.test(ind)

rmse_train = np.sqrt(np.mean(np.power(y_pred - y, 2)))
mae_train = np.mean(np.abs(y_pred - y))
print('train rmse = ' + str(rmse_train))
print('train mae = ' + str(mae_train))
model.sess.close()
