
import sys
import tensorflow as tf
import numpy as np
import scipy.io
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
import time
import sklearn
from sklearn.cluster import KMeans
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
import scipy
import scipy.stats as stats



np.random.seed(1)
tf.set_random_seed(1)

jitter = 1e-3

class POND:
    #self.tf_log_lengthscale: log of RBF lengthscale
    #self.tf_log_tau: log of inverse variance
    #self.tf_y: observed entries
    #self.tf_Z: pseudo inputs
    #self.tf_U: embeddings,
    #U: init U
    #m: no. of pseudo inputs
    #y: observed tensor entries
    #B: batch-size
    #lr: learning rate
    #ind: entry indices
    # num_c: number of channels
    def __init__(self, ind, y, U, m, B, lr, num_c):
        self.U = U
        self.m = m
        self.y = y.reshape([y.size,1])
        self.ind = ind
        self.B = B
        self.learning_rate = lr
        self.nmod = len(self.U)
        self.r = self.U[0].shape[1]
        self.tf_U = [tf.Variable(self.U[k], dtype=tf.float32) for k in range(self.nmod)]
        self.d = 0
        self.num_channels = num_c
        for k in range(self.nmod):
            self.d = self.d + self.U[k].shape[1]

        #init mu, L, Z
        Zinit = self.init_pseudo_inputs()
        self.tf_W = tf.reshape(tf.Variable(Zinit, dtype=tf.float32),[self.m, self.r, self.nmod,1])
        self.N = y.size


        #covariance of the noise
        self.U_covar_diag_tf = [tf.Variable(np.ones(self.U[k].shape[0]*self.U[k].shape[1]),
                                       dtype=tf.float32) for k in range(self.nmod)]
        U_covar_sqrt_mat_tf = [tf.linalg.diag(self.U_covar_diag_tf[k]) for k in range(self.nmod)]


        #variational posterior
        self.tf_mu = tf.Variable(np.zeros([m,1]), dtype=tf.float32)
        self.tf_L = tf.Variable(np.eye(m), dtype=tf.float32)
        #shallow kernel parameters
        self.tf_log_lengthscale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)

        #Stochastic Variational ELBO
        #A mini-batch of observed entry indices
        self.tf_sub = tf.placeholder(tf.int32, shape=[None, self.nmod])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])

        #convolution variables and parameters
        self.conv0_f_shape =[2,2,1,self.num_channels]
        self.tf_conv0_w = tf.Variable(tf.truncated_normal(self.conv0_f_shape, stddev=0.03))
        self.tf_bias0 = tf.Variable(tf.truncated_normal([self.num_channels]))
        self.conv1_f_shape = [self.r,1,self.num_channels,self.num_channels]


        self.tf_conv1_w = tf.Variable(tf.truncated_normal(self.conv1_f_shape, stddev=0.03))
        self.tf_bias1 = tf.Variable(tf.truncated_normal([self.num_channels]))

        self.conv2_f_shape = [1,self.nmod,self.num_channels,self.num_channels]
        self.tf_conv2_w = tf.Variable(tf.truncated_normal(self.conv2_f_shape,stddev=0.03))
        self.tf_bias2 = tf.Variable(tf.truncated_normal([self.num_channels]))

        #compute convolutions for pseudo inputs
        self.tf_Z = self.compute_convs(self.tf_W)


        #compute convolutions and generate noise for batch
        tf_noise = [tf.matmul(0.1*U_covar_sqrt_mat_tf[k],
                         tf.random_normal(shape=[self.U[k].shape[0] * self.U[k].shape[1], 1])) for k in range(self.nmod)]
        tf_noise = [tf.reshape(tf_noise[k],self.U[k].shape) for k in range(self.nmod)]

        tf_inputs = tf.concat([tf.gather((self.tf_U[k]+ tf_noise[k]),
                                          self.tf_sub[:, k]) for k in range(self.nmod) ], 1)

        tf_inputs = tf.reshape(tf_inputs,[-1, self.r, self.nmod,1])
        tf_inputs = self.compute_convs(tf_inputs)


        Ltril = tf.matrix_band_part(self.tf_L, -1, 0)
        Kmm = self.kernel_matrix(self.tf_Z)
        Kmn = self.kernel_cross(self.tf_Z, tf_inputs)
        Knm = tf.transpose(Kmn)
        KnmKmmInv = tf.transpose(tf.matrix_solve(Kmm, Kmn))
        KnmKmmInvL = tf.matmul(KnmKmmInv, Ltril)
        tau = tf.exp(self.tf_log_tau)
        lengthscale = tf.exp(self.tf_log_lengthscale)
        hh_expt = tf.matmul(Ltril, tf.transpose(Ltril)) + tf.matmul(self.tf_mu, tf.transpose(self.tf_mu))
        ELBO = -0.5*tf.linalg.logdet(Kmm) - 0.5*tf.trace(tf.matrix_solve(Kmm, hh_expt)) + 0.5*tf.reduce_sum(tf.log(tf.pow(tf.diag_part(Ltril), 2))) \
                + 0.5*self.N*self.tf_log_tau - 0.5*tau*self.N/self.B*tf.reduce_sum(tf.pow(self.tf_y - tf.matmul(KnmKmmInv,self.tf_mu), 2)) \
                - 0.5*tau*( self.N*(1+jitter) - self.N/self.B*tf.reduce_sum(KnmKmmInv*Knm) + self.N/self.B*tf.reduce_sum(tf.pow(KnmKmmInvL,2)) ) \
                + 0.5*self.m - 0.5*self.N*tf.log(2.0*tf.constant(np.pi, dtype=tf.float32))  #\
                #- 0.5*tf.reduce_sum(tf.pow(self.tf_U[0],2)) - 0.5*tf.reduce_sum(tf.pow(self.tf_U[1],2)) - 0.5*tf.reduce_sum(tf.pow(self.tf_U[2],2))
                #- 0.5*tf.pow(tau,2) - 0.5*tf.pow(lengthscale, 2)

        #Add entropy of variational posterior to ELBO
        # This uses the property that the log det(A) = 2*sum(log(real(diag(C))))
        # where C is the cholesky decomposition of A. This allows us to avoid computing the cholesky decomposition
        for k in range(self.nmod):
            ELBO += 0.5*2.0 * math_ops.reduce_sum(
                 math_ops.log(math_ops.real(array_ops.matrix_diag_part(0.1*U_covar_sqrt_mat_tf[k]))),
                 axis=[-1])\
                + (self.U[k].shape[0]*self.U[k].shape[1])/2*(1+tf.log(2.0*tf.constant(np.pi, dtype=tf.float32)))


        self.loss = -ELBO
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.minimizer = self.optimizer.minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def init_pseudo_inputs(self):
        part = [None for k in range(self.nmod)]
        for k in range(self.nmod):
                part[k] = self.U[k][self.ind[:,k], :]
        X = np.hstack(part)

        X = X[np.random.randint(X.shape[0], size=self.m*100),:]

        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(X)
        return kmeans.cluster_centers_
        #idx = np.random.permutation(X.shape[0])
        #self.tf_mu = tf.Variable(self.y[idx[0:self.m],:], dtype=tf.float32)
        #return X[idx[0:self.m],:].copy()

    def compute_convs(self,tf_X):
        tf_Y = tf.nn.conv2d(tf_X, self.tf_conv0_w, [1, 1, 1, 1], padding="SAME")
        tf_Y = tf.nn.conv2d(tf.math.tanh(tf_Y+self.tf_bias0), self.tf_conv1_w, [1, 1, 1, 1], padding="VALID")
        tf_Y = tf.nn.conv2d(tf.math.tanh(tf_Y+self.tf_bias1), self.tf_conv2_w, [1, self.nmod, 1, 1], padding="VALID")
        tf_Y = tf.math.tanh(tf_Y+self.tf_bias2)
        return tf.reshape(tf_Y, [-1, self.num_channels])

    def kernel_matrix(self, tf_X):
        #rbf kernel
        col_norm2 = tf.reduce_sum(tf_X*tf_X, 1)
        col_norm2 = tf.reshape(col_norm2, [-1,1])
        K = col_norm2 - 2.0*tf.matmul(tf_X, tf.transpose(tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        K = K + jitter*tf.eye(tf.shape(K)[0])
        return K

    def kernel_cross(self, tf_Xt, tf_X):
        col_norm1 = tf.reshape(tf.reduce_sum(tf_Xt*tf_Xt, 1), [-1, 1])
        col_norm2 = tf.reshape(tf.reduce_sum(tf_X*tf_X, 1), [-1, 1])
        K = col_norm1 - 2.0*tf.matmul(tf_Xt, tf.transpose(tf_X)) + tf.transpose(col_norm2)
        K = tf.exp(-1.0/tf.exp(self.tf_log_lengthscale)*K)
        return K

    def pred(self, test_ind):
        U_covar_sqrt_mat_tf = [tf.linalg.diag(self.U_covar_diag_tf[k]) for k in range(self.nmod)]
        tf_noise = [tf.matmul(0.1*U_covar_sqrt_mat_tf[k],
                              tf.random_normal(shape=[self.U[k].shape[0] * self.U[k].shape[1], 1])) for k in
                                range(self.nmod)]
        tf_noise = [tf.reshape(tf_noise[k], self.U[k].shape) for k in range(self.nmod)]
        tf_inputs = tf.concat([tf.gather(self.tf_U[k]+tf_noise[k], (test_ind[:,k]))  for k in range(self.nmod) ], 1)
        tf_inputs = tf.reshape(tf_inputs, [-1, self.r, self.nmod, 1])
        tf_inputs = self.compute_convs(tf_inputs)

        Knm = self.kernel_cross(tf_inputs, self.tf_Z)
        Kmm = self.kernel_matrix(self.tf_Z)
        pred_mean = tf.matmul(Knm, tf.matrix_solve(Kmm, self.tf_mu))
        return pred_mean

    def test(self, test_ind):
        pred_mean = self.pred(test_ind)
        print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),
                                             np.exp(self.tf_log_lengthscale.eval(session=self.sess))))
        res = self.sess.run(pred_mean, {self.tf_y:self.y})
        res = res.reshape(res.size)
        return res

    def get_inputs(self, ind):
        tf_inputs = tf.concat([tf.gather(self.tf_U[k], (ind[:,k]))  for k in range(self.nmod) ], 1)
        res = self.sess.run(tf_inputs, {self.tf_y:self.y})
        return res


    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nepoch = 10):
         print('start')
         times = []
         for iter in range(nepoch):
             start = time.time()
             curr = 0
             while curr < self.N:
                 batch_ind = np.random.choice(self.N, self.B, replace=False)
                 tf_dict = {self.tf_sub: self.ind[batch_ind, :], self.tf_y: self.y[batch_ind]}
                 curr = curr + self.B
                 self.sess.run(self.minimizer, feed_dict=tf_dict)
             stop = time.time()
             times.append(stop - start)
             print('epoch %d finished' % iter)
             print('Duration is ' + str(stop - start))
         print('Avg Duration is ' + str(sum(times) / 3))
         #print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),  np.exp(self.tf_log_lengthscale.eval(session=self.sess))))
         #print('mu = ')
         #print(self.tf_mu.eval(session=self.sess))
         #print('U0 diff = %g'%(np.linalg.norm(self.tf_U[0].eval(session=self.sess) - self.U[0], 'fro')))
         #print('U1 diff = %g'%(np.linalg.norm(self.tf_U[1].eval(session=self.sess) - self.U[1], 'fro')))
         #print('U2 diff = %g'%(np.linalg.norm(self.tf_U[2].eval(session=self.sess) - self.U[2], 'fro')))
         #print(self.tf_L.eval(session=self.sess))
