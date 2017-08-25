整个算法还是一直不断更新 Q table 里的值, 然后再根据新的值来判断要在某个 state 采取怎样的 action. 不过于 Qlearning 不同之处: 
Sarsa在当前 state 已经想好了 state 对应的 action, 而且想好了 下一个 state_ 和下一个 action_ (Qlearning 还没有想好下一个 action_) 
更新 Q(s,a) 的时候基于的是下一个 Q(s_, a_) (Qlearning 是基于 maxQ(s_))，这种不同之处使得 Sarsa 相对于 Qlearning, 更加的胆小. 因为 Qlearning 永远都是想着 maxQ 最大化, 因为这个 maxQ 而变得贪婪, 不考虑其他非 maxQ 的结果. 我们可以理解成 Qlearning 是一种贪婪, 大胆, 勇敢的算法. 而 Sarsa 是一种保守的算法, 他在乎每一步决策, 对于错误和死亡比较铭感. 这一点我们会在可视化的部分看出他们的不同. 两种算法都有他们的好处, 比如在实际中, 你比较在乎机器的损害, 用一种保守的算法, 在训练时就能减少损坏的次数. 


逻辑

![逻辑](http://180.76.148.87/sarsasu.png)