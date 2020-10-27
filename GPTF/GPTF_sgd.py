import sys
import tensorflow as tf
import numpy as np
import time
from sklearn.cluster import KMeans
import scipy
import scipy.stats as stats

np.random.seed(1)
tf.set_random_seed(1)

jitter = 1e-3

class GPTF:
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
    def __init__(self, ind, y, U, m, B, lr):
        self.U = U
        self.m = m
        self.y = y.reshape([y.size,1])
        self.ind = ind
        self.B = B
        self.learning_rate = lr
        self.nmod = len(self.U)
        self.tf_U = [tf.Variable(self.U[k], dtype=tf.float32) for k in range(self.nmod)]
        #dim. of pseudo input
        self.d = 0
        for k in range(self.nmod):
            self.d = self.d + self.U[k].shape[1]
        #init mu, L, Z
        Zinit = self.init_pseudo_inputs()
        self.tf_Z = tf.Variable(Zinit, dtype=tf.float32)
        self.N = y.size
        #variational posterior
        self.tf_mu = tf.Variable(np.zeros([m,1]), dtype=tf.float32)
        self.tf_L = tf.Variable(np.eye(m), dtype=tf.float32)
        #kernel parameters
        self.tf_log_lengthscale = tf.Variable(0.0, dtype=tf.float32)
        self.tf_log_tau = tf.Variable(0.0, dtype=tf.float32)


        #Stochastic Variational ELBO
        #A mini-batch of observed entry indices
        self.tf_sub = tf.placeholder(tf.int32, shape=[None, self.nmod])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, 1])

        tf_inputs = tf.concat([ tf.gather(self.tf_U[k], self.tf_sub[:, k])  for k in range(self.nmod) ], 1)
        print(tf_inputs.get_shape())
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
                + 0.5*self.m - 0.5*self.N*tf.log(2.0*tf.constant(np.pi, dtype=tf.float32))#\
                #- 0.5*tf.reduce_sum(tf.pow(self.tf_U[0],2)) - 0.5*tf.reduce_sum(tf.pow(self.tf_U[1],2)) - 0.5*tf.reduce_sum(tf.pow(self.tf_U[2],2))
                #- 0.5*tf.pow(tau,2) - 0.5*tf.pow(lengthscale, 2)

        self.loss = -ELBO
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.minimizer = self.optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def init_pseudo_inputs(self):
        part = [None for k in range(self.nmod)]
        for k in range(self.nmod):
            part[k] = self.U[k][self.ind[:,k], :]
        X = np.hstack(part)

        X = X[np.random.randint(X.shape[0], size=self.m * 100), :]
        print(X.shape)

        kmeans = KMeans(n_clusters=self.m, random_state=0).fit(X)
        return kmeans.cluster_centers_



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
        tf_inputs = tf.concat([tf.gather(self.tf_U[k], (test_ind[:,k]))  for k in range(self.nmod) ], 1)
        Knm = self.kernel_cross(tf_inputs, self.tf_Z)
        Kmm = self.kernel_matrix(self.tf_Z)
        pred_mean = tf.matmul(Knm, tf.matrix_solve(Kmm, self.tf_mu))
        return pred_mean

    def test(self, test_ind):
        pred_mean = self.pred(test_ind)
        print('tau = %g, length-scale = %g'%(np.exp(self.tf_log_tau.eval(session=self.sess)),  np.exp(self.tf_log_lengthscale.eval(session=self.sess))))
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
        print(self.N/self.B)
        times = []
        for iter in range(nepoch):
            start = time.time()
            curr = 0
            while curr < self.N:
                batch_ind = np.random.choice(self.N, self.B, replace=False)
                tf_dict = {self.tf_sub:self.ind[batch_ind,:], self.tf_y:self.y[batch_ind]}
                curr = curr + self.B
                self.sess.run(self.minimizer,feed_dict = tf_dict)
            stop = time.time()
            times.append(stop-start)
            print('epoch %d finished'%iter)
            print('Duration is ' + str(stop-start))
        print('Avg Duration is ' + str(sum(times)/3))
