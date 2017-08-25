整个算法还是一直不断更新 Q table 里的值, 然后再根据新的值来判断要在某个 state 采取怎样的 action. 不过于 Qlearning 不同之处: 
Sarsa在当前 state 已经想好了 state 对应的 action, 而且想好了 下一个 state_ 和下一个 action_ (Qlearning 还没有想好下一个 action_) 
更新 Q(s,a) 的时候基于的是下一个 Q(s_, a_) (Qlearning 是基于 maxQ(s_))，这种不同之处使得 Sarsa 相对于 Qlearning, 更加的胆小. 因为 Qlearning 永远都是想着 maxQ 最大化, 因为这个 maxQ 而变得贪婪, 不考虑其他非 maxQ 的结果. 我们可以理解成 Qlearning 是一种贪婪, 大胆, 勇敢的算法. 而 Sarsa 是一种保守的算法, 他在乎每一步决策, 对于错误和死亡比较铭感. 这一点我们会在可视化的部分看出他们的不同. 两种算法都有他们的好处, 比如在实际中, 你比较在乎机器的损害, 用一种保守的算法, 在训练时就能减少损坏的次数. 


sarsa逻辑

![逻辑](http://180.76.148.87/sarsasu.png)

q learning 逻辑

![逻辑](http://180.76.148.87/qlearningtu.png)

Sarsa-lambda 

![逻辑](http://180.76.148.87/sarsa-lambda.png)


Sarsa-lambda 是基于 Sarsa 方法的升级版, 他能更有效率地学习到怎么样获得好的 reward. 如果说 Sarsa 和 Qlearning 都是每次获取到 reward, 只更新获取到 reward 的前一步. 那 Sarsa-lambda 就是更新获取到 reward 的前 lambda 步. lambda 是在 [0, 1] 之间取值, 
如果 lambda = 0, Sarsa-lambda 就是 Sarsa, 只更新获取到 reward 前经历的最后一步. 
如果 lambda = 1, Sarsa-lambda 更新的是 获取到 reward 前所有经历的步. 

此外，作者采取了和原始的GAN不同的结构和训练方法。总的训练框架来自于DRAGAN（arxiv：https://arxiv.org/pdf/1705.07215.pdf），经过实验发现这种训练方法收敛更快并且能产生更稳定的结果。
生成器G的结构类似于SRResNet（arxiv：https://arxiv.org/pdf/1609.04802.pdf）：


https://makegirlsmoe.github.io/assets/pdf/technical_report.pdf


challenger.ai

Deep Q network (off-policy) 