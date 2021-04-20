# SymNet_torch_dev
SymNet_torch development repo

# requirement

See `requirementes.txt`. Python 3.7 + PyTorch 1.8.1

# usage


MIT

    python run_symnet.py --network fc_obj --name MIT_obj_lr3e-3 --data MIT --epoch 1500 --batchnorm --lr 3e-3
    python run_symnet.py --name MIT_best --data MIT --epoch 400 --obj_pred MIT_obj_lr3e-3_ep1120.pkl --batchnorm --lr 5e-4 --bz 512 --lambda_cls_attr 1 --lambda_cls_obj 0.01 --lambda_trip 0.03 --lambda_sym 0.05 --lambda_axiom 0.01 --rmd_metric softmax
    

UT

    python run_symnet.py --network fc_obj --name UT_obj_lr1e-3 --data UT --epoch 300 --batchnorm --lr 1e-3
    python run_symnet.py --name UT_best --data UT --epoch 700 --obj_pred UT_obj_lr1e-3_ep140.pkl --batchnorm  --wordvec onehot  --lr 1e-4 --bz 256 --lambda_cls_attr 1 --lambda_cls_obj 0.5 --lambda_trip 0.5 --lambda_sym 0.01 --lambda_axiom 0.03 --rmd_metric softmax


# progress

UT已经有正常分数了(51.3+)
MIT14分（差很多）


1. test_obj
2. gczsl run/test (GCZSL evaluator还没有改过，可能会有问题)
3. 多卡训练


# changes/notes

训练loss时用`self.args.rmd_metric`, test时原本（开源版本）是用"rmd"，现在测试也用`self.args.rmd_metric`

1. logs和weights合并到了logs
2. args.weight_type 改成了可读的str类型
3. args.trained_weight现在是直接的绝对/相对路径
4. prediction现在不是dict是只有一个list了
5. prob_pair, prob_attr开源时是分开产生的，现在是同一个


# TODOs

1. yaml以及自动备份
0. MSEloss在L2的时候不对：少个平方
1. activation function和weight initializer没有设置
3. args的key名字跟operator不太一样，可以考虑统一一下
7. lr scheduler还没实现。如果要加的话还要存进statedict
8. GRADIENT_CLIPPING还没实现
9. focal loss not implemented
10. loss的log精简一下，tb不要显示那么多（参考tf版本
11. reshape->view
12. symnet的compute_loss参数prob_RMD_plus, prob_RMD_minus太明显了 藏起来
13. make this repo more Python3 (type, etc.)
0. 检查snapshot继续训练时读取有没有错，分数是不是合理