# SymNet_torch_dev
SymNet_torch development repo

# requirement

See `requirementes.txt`. Python 3.7 + PyTorch 1.8.1


# changes/TODOs

1. logs和weights合并到了logs
2. GCZSL evaluator还没有改过，可能会有问题
3. args的key名字跟operator不太一样，可以考虑统一一下
4. data部分可以加cache
5. args.weight_type 改成可读的str类型
6. args.trained_weight现在是直接的绝对/相对路径
7. lr scheduler还没实现。如果要加的话还要存进statedict
8. GRADIENT_CLIPPING还没实现
9. tensorboard要重写
10. prediction现在不是dict是只有一个list了