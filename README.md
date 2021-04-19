# SymNet_torch_dev
SymNet_torch development repo

# requirement

See `requirementes.txt`. Python 3.7 + PyTorch 1.8.1


# changes

1. logs和weights合并到了logs
5. args.weight_type 改成了可读的str类型
6. args.trained_weight现在是直接的绝对/相对路径
10. prediction现在不是dict是只有一个list了


# TODOs

1. activation function和weight initializer没有设置
2. GCZSL evaluator还没有改过，可能会有问题
3. args的key名字跟operator不太一样，可以考虑统一一下
4. data部分可以加cache
7. lr scheduler还没实现。如果要加的话还要存进statedict
8. GRADIENT_CLIPPING还没实现
9. focal loss not implemented
10. loss的log精简一下，tb不要显示那么多（参考tf版本
11. reshape->view
12. symnet的compute_loss参数prob_RMD_plus, prob_RMD_minus太明显了 藏起来