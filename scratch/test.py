import torch
a = torch.Tensor([[2, 3, 4], [5, 6, 7]])
print(a[:, 1])

def tile_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,2,3,1,2,3,1,2,3)"""
    mul = [1] * len(tensor.shape)
    mul[axis] = multiple

    return tensor.repeat(mul)

def repeat_tensor(tensor, axis, multiple):
    """e.g. (1,2,3)x3 = (1,1,1,2,2,2,3,3,3)"""
    
    result_shape = list(tensor.shape)
    for i, v in enumerate(result_shape):
        if v is None:
            result_shape[i] = tensor.shape[i]
    result_shape[axis] *= multiple

    tensor = torch.unsqueeze(tensor, axis+1)
    mul = [1] * len(tensor.shape)
    mul[axis+1] = multiple
    tensor = tensor.repeat(mul)
    tensor = torch.reshape(tensor, result_shape)

    return tensor

print(tile_tensor(a, 1, 3))
print(repeat_tensor(a, 1, 3))