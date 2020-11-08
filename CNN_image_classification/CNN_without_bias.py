# -*- coding: utf-8 -*- 
''' TensorFlow를 이용한 CNN 구현 ''' 
import tensorflow as tf 
from visualize_filter.filter_show import filter_show 
from tensorflow.examples.tutorials.mnist import input_data 

######### 
# 신경망 데이터 가져오기 
###### 

# read_data_sets: 라이브러리에서 자동으로 MNIST 데이터 가져옴
# - one_hot: one hot encoding 옵션
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True) 



######### 
# 신경망 모델 구성 
###### 

# tf.placeholder: 입력으로 사용할 데이터의 타입만 지정해주고 실제값은 나중에 세션에서 실행될때 입력
# X: input이 28x28 이미지, [None, 28, 28, 1] -> input data num: None, height: 28, width: 28, filter output에 따른 data group: 1개
# Y: output은 10가지 class (숫자 0~9), [None, 10] -> input data num: None, detected class: 10가지
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) 
Y = tf.placeholder(tf.float32, [None, 10]) 

# params: 가중치 초기화 (parameters)
params = {} 



############### 
# [ Layer-1 ]
# Parameters(filter) shape   = (5, 5, 1, 32)
# Layer-1 Conv input shape   = (?, 28, 28, 1)
# Layer-1 Conv output shape  = (?, 28, 28, 32) by filter num
# Layer-1 Pool input shape   = (?, 28, 28, 32) 
# Layer-1 Pool output shape  = (?, 14, 14, 32) by strides 2x2 filter size
#####

# tf.Variable: 생성자으로 구성된 object 형태
# - tf.random_normal: 정규 분포에서 임의의 값을 출력
#   + shape: integer Tensor or Python array, [5 5]: 커널 크기, 1: X의 특성수, 32: 필터 갯수
#   + stddev: standard deviation(표준편차)
params['W1'] = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.01)) 

# tf.nn.relu: Non-Linearity (Activation func)
# - tf.nn.conv2d: Convolution layer
#   + strides: convolution 연산할때 pixel shifting 크기
#   + padding: 커널 슬라이딩시 원본 데이터 최외곽에서 한칸 밖으로 더 움직이는 옵션 
L1 = tf.nn.relu(tf.nn.conv2d(X, params['W1'], strides=[1, 1, 1, 1], padding='SAME')) 

# tf.nn.max_pool:
# - x(L1): Feature Map input
# - ksize: pooling size, 2x2 크기 묶어서 filter
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# tf.nn.dropout: 노드간 연결성 80%로 세팅, Overfitting 방지를 위해서 이용(regularization 기)
L1 = tf.nn.dropout(L1, 0.8) 



###############
# [ Layer-2 ]
# Parameters(filter) shape   = (5, 5, 32, 64)
# Layer-2 Conv input shape   = (?, 14, 14, 32) = L1 output volume
# Layer-2 Conv output shape  = (?, 14, 14, 64) by filter num
# Layer-2 Pool input shape   = (?, 14, 14, 64)
# Layer-2 Pool output shape  = (?, 7, 7, 64) by strides 2x2 filter size
# Flatten shape              = (1, 7 * 7 * 64)
#####

params['W2'] = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.01)) 
L2 = tf.nn.relu(tf.nn.conv2d(L1, params['W2'], strides=[1, 1, 1, 1], padding='SAME')) 
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# tf.reshape(Flatten layer): Full Connect input을 위해 크기 변환(차원 감소)
L2 = tf.reshape(L2, [-1, 7 * 7 * 64]) 
L2 = tf.nn.dropout(L2, 0.8) 



###############
# [ Layer-3 ]
# Parameters(filter) shape   = (7 * 7 * 64, 256)
# Layer-3 FC input shape     = (1, 7 * 7 * 64) = L2 Flatten output
# Layer-3 FC output shape    = (1, 256)
#####

W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01)) 

# tf.matmul: matrix multiplication
L3 = tf.nn.relu(tf.matmul(L2, W3)) 
L3 = tf.nn.dropout(L3, 0.5) 



###############
# [ Layer-4 ]
# Parameters(filter) shape   = (256, 10) to class 
# Layer-3 Softmax input shape   = (1, 256) = L3 FC output
# Layer-3 Softmax output shape  = (1, 10)
#####

W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01)) 
model = tf.matmul(L3, W4) 

# tf.reduce_mean: 전체 평균을 구함(모든 차원을 제거)
# - tf.nn.softmax_cross_entropy_with_logits: Computes softmax cross entropy between logits and labels.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y)) 

# tf.train.AdamOptimizer: Momentum과 RMSprop을 조합시킨 Gradient Descnet
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost) 



######### 
# 신경망 모델 학습 
###### 

# init: Session 공간에 변수 초기화하기 위해서 사용되는 변수
init = tf.global_variables_initializer() 
# sess: Session 공간 확보
sess = tf.Session() 

sess.run(init) 
# batch size: sample 데이터 중 한번에 모델에 넘겨주는 데이터의 수
batch_size = 100 
total_batch = int(mnist.train.num_examples/batch_size) 

# 결과 저장
for epoch in range(15): 
    total_cost = 0 
    for i in range(total_batch): 
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
        
        # reshape NxN -> 28x28 for input
        batch_xs = batch_xs.reshape(-1, 28, 28, 1) 
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys}) 
        total_cost += cost_val 
        
    print('Epoch:', '%04d' % (epoch + 1), 
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)) 
    print(params['W2'].get_shape().as_list()) 
        
    # Visualize Feature Map 
    filter = sess.run(params['W2']) 
    filter_show(filter.transpose(3,2,0,1)) 
    
print('최적화 완료!') 



######### 
# 결과 확인 
###### 
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)) 
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32)) 
print('정확도:', sess.run(accuracy, 
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1), 
                                  Y: mnist.test.labels}))

