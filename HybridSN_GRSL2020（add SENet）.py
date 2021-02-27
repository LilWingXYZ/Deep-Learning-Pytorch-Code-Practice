class_num = 16
rate = 16	# SENet论文中建议rate=sqrt(256)=16
class HybridSN(nn.Module):
  def __init__(self):
    super(HybridSN, self).__init__()

    self.conv1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0)
    self.conv2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0)
    self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0)

    inputX = self.get2Dinput()
    inputConv4 = inputX.shape[1] * inputX.shape[2]
    self.conv4 = nn.Conv2d(inputConv4, 64, kernel_size=(3, 3), stride=1, padding=0)
    #加入atention机制
    self.fc_41 = nn.Conv2d(64, 64//rate, kernel_size=1)
    self.fc_42 = nn.Conv2d(64//rate, 64, kernel_size=1)

    num = inputX.shape[3]-2 #二维卷积后（64, 17, 17）-->num = 17
    inputFc1 = 64 * num * num
    self.fc1 = nn.Linear(inputFc1, 256) # 64 * 17 * 17 = 18496
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, class_num)
    self.dropout = nn.Dropout(0.4)

  def get2Dinput(self):
    with torch.no_grad():
      x = torch.zeros((1, 1, self.L, self.S, self.S))
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
    return x

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))

    x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
    x = F.relu(self.conv4(x))

    # Squeeze 操作：global average pooling
    w = F.avg_pool2d(x, x.size(2))
    # Excitation 操作： fc（压缩到16分之一）--Relu--fc（激到之前维度）--Sigmoid（保证输出为 0 至 1 之间）
    w = F.relu(self.fc_41(w))
    w = F.sigmoid(self.fc_42(w))
    x = x * w

    x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x
