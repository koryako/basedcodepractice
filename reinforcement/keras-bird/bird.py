#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

GAME = 'bird' # 游戏名
CONFIG = 'nothreshold'
ACTIONS = 2 # 有效动作数：不动+跳=2个
GAMMA = 0.99 # 折扣系数，未来的奖励转化为现在的要乘的一个系数
OBSERVATION = 3200. # 训练之前观察多少步
EXPLORE = 3000000. # epsilon衰减的总步数
FINAL_EPSILON = 0.0001 # epsilon的最小值
INITIAL_EPSILON = 0.1 # epsilon的初始值，epsilon逐渐减小
REPLAY_MEMORY = 50000 # 记住的情景(状态s到状态s'的所有信息)数
BATCH = 32 # 选取的小批量训练样本数
# 一帧一个输入动作
FRAME_PER_ACTION = 1
# 预处理后的图片尺寸
img_rows , img_cols = 80, 80
# 每次堆叠4帧灰阶图像，相当于4通道
img_channels = 4 


# 构建神经网络模型
def buildmodel():
    print("Now we build the model")
    # 以下注释见文中
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1),init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(2,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam) # 使用损失函数为均方误差，优化器为Adam。
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # 得到一个游戏模拟器
    game_state = game.GameState()

    # 保存之前的观察到回放存储器D
    D = deque()

    # 什么也不做来得到第一个状态然后预处理图片为80x80x4格式
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # do_nothing 为 array([1,0])
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
    # 初始化时，堆叠4张图都为初始的同1张
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)  # s_t为四张图的堆叠

    # 为了在Keras中使用，我们需要调整数组形状，在头部增加一个维度
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    if args['mode'] == 'Run':
        OBSERVE = 999999999    # 我们一直观察，而不训练
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                      # 否则我们在观察一段时间之后开始训练
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0 # t为总帧数
    while (True):
        # 每次循环重新初始化的值
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        # 通过epsilon贪心算法选择行为
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS) # 随机选取一个动作
                a_t[action_index] = 1         # 生成相应的规范化动作输入参数
            else:
                q = model.predict(s_t)       # 输入当前状态得到预测的Q值
                max_Q = np.argmax(q)         # 返回数组中最大值的索引
                # numpy.argmax(a, axis=None, out=None)
                # Returns the indices of the maximum values along an axis.
                action_index = max_Q         # 索引0代表啥也不做，索引1代表跳一下
                a_t[max_Q] = 1                 # 生成相应的规范化动作输入参数

        # 在开始训练之后并且epsilon小于一定值之前，我们逐步减小epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选定的动作，并观察返回的下一状态和奖励
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # 将图像处理为灰阶，调整尺寸、亮度
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        # 调整图像数组形状，增加头两维到4维
        x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
        # 将s_t的前三帧添加在新帧的后面，新帧的索引为0，形成最后的4帧图像
        s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)


        # 存储状态转移到回放存储器
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 如果观察完成，则
        if t > OBSERVE:
            # 抽取小批量样本进行训练
            minibatch = random.sample(D, BATCH)
            # inputs和targets一起构成了Q值表
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            # 开始经验回放
            for i in range(0, len(minibatch)):
                # 以下序号对应D的存储顺序将信息全部取出，
                # D.append((s_t, action_index, r_t, s_t1, terminal))
                state_t = minibatch[i][0]     # 当前状态
                action_t = minibatch[i][1]   # 输入动作
                reward_t = minibatch[i][2]     # 返回奖励
                state_t1 = minibatch[i][3]   # 返回的下一状态
                terminal = minibatch[i][4]   # 返回的是否终止的标志

                inputs[i:i + 1] = state_t    # 保存当前状态，即Q(s,a)中的s

                # 得到预测的以输入动作x为索引的Q值列表
                targets[i] = model.predict(state_t)  
                # 得到下一状态下预测的以输入动作x为索引的Q值列表
                Q_sa = model.predict(state_t1)

                if terminal:  # 如果动作执行后游戏终止了，该状态下(s)该动作(a)的Q值就相当于奖励
                    targets[i, action_t] = reward_t
                else:          # 否则，该状态(s)下该动作(a)的Q值就相当于动作执行后的即时奖励和下一状态下的最佳预期奖励乘以一个折扣率
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # 用生成的Q值表训练神经网络，同时返回当前的误差
            loss += model.train_on_batch(inputs, targets)

        s_t = s_t1 # 下一状态变为当前状态
        t = t + 1  # 总帧数+1

        # 每100次迭代存储下当前的训练模型
        if t % 100 == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # 输出信息
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel() # 先建立模型
    trainNetwork(model,args) # 开始训练

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)  #接受参数 mode
    args = vars(parser.parse_args())  # args是字典，'mode'是键
    playGame(args)    # 开始游戏

if __name__ == "__main__":
    main()  #执行本脚本时以main函数开始

作者：treelake
链接：http://www.jianshu.com/p/3ba69493f020
來源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


