# 待处理任务

## 内存

10^6 * 200 * 10 * 9 bytes ~= 20 G

- 棋子信息使用 np.int8
- 将状态与特征分离，以保证`replay buffer`能够存储100万盘对局【假设长度为200】

## 游戏

- 环境始终以红方角度返回游戏结果，即值区间为`[-1,1]`，即`1`代表当前红胜，`-1`代表红负
- 模型预测与此不同，状态值函数值区间为`[0,1]`，即`1`代表当前走子方获胜，`-1`代表当前走子方负
- MCTS模拟时，黑方走子后的奖励，使用负号转换为黑方角度的奖励，同样，其值也需要反号

## 矢量环境【VectorEnv】

1. `reset`
   观察副本复制N次。`info`字典包含`to_play`及`legal_actions`信息。

2. `step(actions)`
   正常情形下，观察为`next state`，但如终止或截断，则返回初始状态。`next state`在`info`字典`final_observation`中以列表形式，非终止或截断以`None`表达。

3. `close`

## 特征

+ 除观察特征外，至少需要叠加上一步移动信息作为模型输入。

## 修改标记

相对于原文，修改处标记：🚨

## 修订清单

- [ ] 棋盘连续未吃子、步数、半步数检验问题
- [x] 清除节点`be_killed`属性
- [x] 使用`graphviz`显示MCTS根节点树
- [x] 反向传播值计算方式
- [x] 解决`game`在`ray`包中序列化问题
- [x] 移动编码 -> 0-2085 + 1【2086代表空白】
- [x] 确认样本均匀抽样
- [x] 验证吸收状态：终止状态递归推理出现重复结果
- [x] 以png格式显示MCTS树时，打印程序未安装字体库导致中文乱码
- [x] 分离 ray
- [x] pygame:render 修正统计结果
- [x] pygame:render 记谱使用列表
- [x] pygame显示：黑方先行时
  > |----  xxxx|

  > |xxxx xxxx|
- [ ] 模型版本号：只有样本数量达到批量才增加
- [ ] target计算，重点在于符号
- [ ] reward相对于指定player?
- [ ] 修正MCTS树表达
- [ ] 修正model v1 动态函数问题
- [ ] GPU:训练时使用GPU，selfplay使用cpu
- [ ] replay buffer -> mongodb
- [ ] 取消参数 use_max_priority
- [ ] 递归推理应该基于 action list ?
- [ ] game限制200步 ?
- [ ] 检查100万缓存内存耗用，相应调整存储对象方式
- [ ] MCST中VALUE与REWARD，以及反向传播必须限定值区间 -> [-1, 1]
- [ ] 吸收状态 policy_logits 表达 均匀分布?全部为0? 训练 target 表达方式
- [ ] recurrent_inference value -> v(s a)
- [ ] selfplay next state value back and truncated
- [ ] This is achieved by treating terminal states as absorbing states during training.
- [ ] 待定

## 超参数

- num_unroll_steps look ahead steps 30,50,100,120
- td_steps paper 论文建议 5 ==> [5,10,15,20]
- num_simulations 论文建议 800 ==> [60,120,180,240,480,640]

## 备注

- 单游戏训练内存1G

- 当存在2种可行路径时，MCTS模拟400次基本可确保概率相等

- 显示`dot`图

- Graphviz (dot) language support for Visual Studio Code

- `ctrl+shift+v` 预览

```cmd
# GraphViz (dot) language support 本地可预览
dot -Tsvg input.dot -o output.svg
```

```python
import xqcpp

result = ""
# maps = ""
for i in range(2086):
    result += "{" + str(i) + ',"' + xqcpp.action2str(i+1) + '"},'
    # maps += '{"' + xqcpp.action2str(i+1) + '",' + str(i) + '},'
    if (i + 1) % 5 == 0:
        result += "\n"
        # maps += "\n"
print(result)
```
## 工具

### 内存使用
```bash
mprof run <path to file>
mprof plot
```

## 实验

| 案例     | fen                                              | 目标              | 问题                        | 备注 |
| :------- | ------------------------------------------------ | :---------------- | :-------------------------- | :--- |
| 单步杀   | 2bak4/5R3/9/3N5/9/5C3/9/9/4p1pr1/5K3 w - 110 0 1 | 检验Q(s,a)        |                             |      |
| 停着     |                                                  | 检验目标计算      | 1. `6768`与`4041`应该等概率 |      |
|          |                                                  | 检验吸收状态      |                             |      |
| 最短路径 |                                                  | 1. 多种杀，取最短 |                             |      |


## tasks

+ vec env
+ ray batch sezrch