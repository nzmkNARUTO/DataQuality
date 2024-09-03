import torch
import torch.nn as nn


# 假设使用PyTorch定义了一个神经网络模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义模型层
        self.fc1 = nn.Linear(10, 20)  # 假设输入有10个特征，输出为10类
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        y_pred = self.fc1(x)
        y_pred = torch.relu(y_pred)
        y_pred = self.fc2(y_pred)
        y_pred = torch.softmax(y_pred, dim=1)
        return y_pred


model = MyModel()
print(model)
# 定义输入数据
input_data = torch.randn(1, 10, requires_grad=True)

# 计算模型输出
output = model(input_data)
output_first_class = output[0, 0]  # 第一类的输出

# 计算第一类输出对所有模型参数的梯度
model.zero_grad()
output_first_class.backward()

# 获取所有参数的梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Parameter: {name}, Gradient: {param.grad}")


print("####################")
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()

model = MyModel()
# 计算模型输出和损失
output = model(input_data)
target = torch.tensor([0])  # 目标类别为第一类
loss = loss_fn(output, target)

# 计算损失对所有参数的梯度
model.zero_grad()
loss.backward()

# 获取所有参数的梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
