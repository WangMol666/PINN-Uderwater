"""
@author: Chao Song
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import cmath

np.random.seed(1234)
tf.set_random_seed(1234)

#频率参数（根据模型选择）
# fre = 4.0 #for homogeneous model（均匀介质模型）
fre = 3.0 #for Marmousi model（复杂地质模型）
PI = 3.1415926
omega = 2.0*PI*fre #角频率（亥姆霍兹方程参数）
# 迭代次数（根据模型复杂度调整）
# niter = 50000 #for homogeneous model（均匀介质）
niter = 100000 #for Marmousi model（复杂地质）

w0 = 3 #固定正弦激活函数的频率参数

#损失记录列表
misfit = [] #Adam优化器损失
misfit1 = [] #L-BFGS-B优化器损失

#梯度计算辅助函数
def fwd_gradients(Y, x):
    dummy = tf.ones_like(Y)
    G = tf.gradients(Y, x, grad_ys=dummy, colocate_gradients_with_ops=True)[0]
    Y_x = tf.gradients(G, dummy, colocate_gradients_with_ops=True)[0]
    return Y_x

#PINN网络结构、物理约束构建、优化器配置与训练逻辑
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, z, A, B, C, ps, m, layers, omega):
        
        X = np.concatenate([x, z], 1) #合并x、z坐标为输入特征
       
        self.iter=0
        self.start_time=0
 
        self.lb = X.min(0) #输入特征(物理域)的下界（用于归一化）
        self.ub = X.max(0) #输入特征(物理域)的上界（用于归一化）
                
        self.X = X
        
        self.x = X[:,0:1]
        self.z = X[:,1:2]

        #物理参数与源项
        self.ps = ps #源项（如声源/震源）
        self.m = m #介质参数（如密度）
        #亥姆霍兹方程系数
        self.A = A
        self.B = B
        self.C = C 

        self.layers = layers #神经网络层结构
        self.omega = omega #角频率
        
        # Initialize NN（初始化网络权重与偏置）
        self.weights, self.biases = self.initialize_NN(layers)  

        # tf placeholders（TensorFlow计算图配置）
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,  #若GPU操作不支持，自动切换到CPU
                                                     log_device_placement=True,  #打印设备分配日志
                                                     #kimi修改：按需分配GPU内存（避免内存溢出）
                                                     gpu_options=tf.GPUOptions(allow_growth=True)
                                                     #end
                                                     ))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])

        #kimi修改：显式指定GPU设备
        with tf.device('/gpu:0'):
        #end
            #预测波场与物理残差
            self.u_real_pred, self.u_imag_pred, self.f_loss = self.net_NS(self.x_tf, self.z_tf)
            # loss function we define（损失函数）
            self.loss = tf.reduce_sum(tf.square(tf.abs(self.f_loss))) #物理方程残差的平方和
        # optimizer used by default (in original paper)
        # 优化器配置（先Adam后L-BFGS-B）
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    #网络参数初始化
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            
            if (l == 0): #第一层单独初始化（输入层到隐藏层）
               W = self.xavier_init_first(size=[layers[l], layers[l+1]])
            else: #隐藏层之后的初始化
               W = self.xavier_init(size=[layers[l], layers[l+1]])

            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32)+0.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size): #隐藏层初始化
        in_dim = size[0]
        out_dim = size[1]        
        w_std = np.sqrt(1.0/in_dim)/w0 #结合正弦激活函数参数w0

        return tf.Variable(tf.random_uniform([in_dim, out_dim], -w_std, w_std), dtype=tf.float32)

    def xavier_init_first(self, size): #第一层权重初始化（不除以w0）
        in_dim = size[0]
        out_dim = size[1]        
        w_std = np.sqrt(1.0/in_dim)
        return tf.Variable(tf.random_uniform([in_dim, out_dim], -w_std, w_std), dtype=tf.float32)

    #神经网络前向传播
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        #输入归一化
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        H =  w0*H 

        #隐藏层：使用固定正弦激活函数
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        #输出层：无激活函数，直接输出波场实部与虚部
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    #物理约束构建
    def net_NS(self, x, z):

        omega = self.omega
        m = self.m
        ps = self.ps

        A = self.A
        B = self.B
        C = self.C

        # 预测波场实部与虚部
        ureal_and_uimag = self.neural_net(tf.concat([x,z], 1), self.weights, self.biases)
        u_real = ureal_and_uimag[:,0:1]
        u_imag = ureal_and_uimag[:,1:2]

        u = tf.complex(u_real, u_imag) #结合为复数波场
        #计算一阶导数（波场对x、z的偏导）
        dudx = fwd_gradients(u, x)
        dudz = fwd_gradients(u, z)
        #计算二阶导数（用于亥姆霍兹方程）
        dudxx = fwd_gradients((A*dudx), x) # A*du/dx 的x偏导
        dudzz = fwd_gradients((B*dudz), z) # B*du/dx 的x偏导

        # 亥姆霍兹方程残差：L(u) - ps = 0（物理约束）
        # 其中 L(u) = C*ω²*m*u + d²(Au)/dx² + d²(Bu)/dz²
        f_loss =  C*omega*omega*m*u + dudxx + dudzz - ps #  L u - f
    
        return u_real, u_imag, f_loss        

    def callback(self, loss):
        #print('Loss: %.3e' % (loss))
        misfit1.append(loss)
        self.iter=self.iter+1
        if self.iter % 10 == 0:
                elapsed = time.time() - self.start_time
                print('It: %d, LBFGS Loss: %.3e,Time: %.2f' %
                      (self.iter, loss, elapsed))
                self.start_time = time.time()     

    #模型训练与预测
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.z_tf: self.z}
        
        start_time = time.time()
        #阶段1：Adam优化器快速收敛
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            misfit.append(loss_value) #记录损失
            # Print（每10次迭代打印进度）
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                #misfit.append(loss_value)
                print('It: %d, Loss: %.3e,Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

        #阶段2：L-BFGS-B优化器精细优化
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

            
    
    def predict(self, x_star, z_star):
        
        tf_dict = {self.x_tf: x_star, self.z_tf: z_star}
        
        u_real_star = self.sess.run(self.u_real_pred, tf_dict)
        u_imag_star = self.sess.run(self.u_imag_pred, tf_dict)

        return u_real_star, u_imag_star
        
        
if __name__ == "__main__": 
      
    layers = [2, 40, 40, 40, 40, 40, 40,40, 40, 2]

    ###选择加载的介质模型###
    #确保模型文件与代码在同一目录，或使用完整的路径
    # Load Data for Homogeneous model
    data = scipy.io.loadmat('Homo_4Hz_singlesource_ps.mat')
    # Load Data for Marmousi model
    #data = scipy.io.loadmat('Marmousi_3Hz_singlesource_ps.mat')
               
    u_real = data['U_real'] 
    u_imag = data['U_imag'] 

    ps = data['Ps'] 

    x = data['x_star'] 
    z = data['z_star'] 
    m = data['m'] 

    A = data['A'] 
    B = data['B'] 
    C = data['C'] 

    # 随机采样部分数据作为训练样本
    N = x.shape[0] # 总数据量（全网格数据）
    N_train = round(N/4*4)  # 训练样本量
 # Training Data    
    idx = np.random.choice(N, N_train, replace=False) # 随机选取索引

    x_train = x[idx,:]
    z_train = z[idx,:]

    ps_train = ps[idx,:]
    A_train = A[idx,:]
    B_train = B[idx,:]
    C_train = C[idx,:]

    m_train = m[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, z_train, A_train, B_train, C_train,  ps_train, m_train, layers, omega)
    model.train(niter)

    ###保存输出结果文件，修改保存路径
    scipy.io.savemat('./results/Helm_pinn_sine_fixed.py/lab002/loss_adam_sine_fixed_mar.mat',{'misfit':misfit})
    scipy.io.savemat('./results/Helm_pinn_sine_fixed.py/lab002/loss_lbfgs_sine_fixed_mar.mat',{'misfit1':misfit1})
    
    # Test Data

    x_star = x
    z_star = z

    u_real_star = u_real
    u_imag_star = u_imag

    # Prediction
    u_real_pred, u_imag_pred = model.predict(x_star, z_star)
    
    # Error
    error_u_real = np.linalg.norm(u_real_star-u_real_pred,2)/np.linalg.norm(u_real_star,2)
    error_u_imag = np.linalg.norm(u_imag_star-u_imag_pred,2)/np.linalg.norm(u_imag_star,2)

    print('Error u_real: %e, Error u_imag: %e' % (error_u_real,error_u_imag))    

    scipy.io.savemat('./results/Helm_pinn_sine_fixed.py/lab002/u_real_pred_sine_fixed_mar.mat',{'u_real_pred':u_real_pred})
    scipy.io.savemat('./results/Helm_pinn_sine_fixed.py/lab002/u_imag_pred_sine_fixed_mar.mat',{'u_imag_pred':u_imag_pred})


