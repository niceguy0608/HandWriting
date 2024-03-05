import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary

from model import ConvNet
from hwdb import HWDB
import logging
import datetime


def get_log(file_name):
    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级
    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level
    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger


def valid(epoch, net, test_loarder, writer, logger):
    logger.info("epoch %d 开始验证..." % epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loarder:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('correct number: ', correct)
        print('totol number:', total)
        acc = 100 * correct / total
        logger.info('第%d个epoch的识别准确率为：%d%%' % (epoch, acc))


def train(epoch, net, criterion, optimizer, train_loader, writer, logger, save_iter=100):
    print("epoch %d 开始训练..." % epoch)
    net.train()
    sum_loss = 0.0
    sum_loss0 = 0.0
    total = 0
    total0 = 0
    correct = 0
    correct0 = 0
    # 数据读取
    for i, (inputs, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()     # 梯度清零
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 取得分最高的那个类
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        total0 += labels.size(0)
        correct += (predicted == labels).sum().item()
        correct0 += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        # 每训练100个batch打印一次平均loss与acc
        sum_loss += loss.item()
        sum_loss0 += loss.item()

        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            # 每跑完一次epoch测试一下准确率
            acc = 100 * correct / total
            logger.info('epoch: %d, batch: %d loss: %.03f, acc: %.04f'
                        % (epoch, i + 1, batch_loss, acc))
            total = 0
            correct = 0
            sum_loss = 0.0

    loss0 = sum_loss0 / total0 * save_iter
    acc0 = 100 * correct0 / total0
    logger.info("epoch %d 的平均loss为%.03f，平均acc为%.04f..." % (epoch, loss0, acc0))


if __name__ == "__main__":
    # 超参数
    epochs = 20
    batch_size = 100
    lr = 0.02

    now = datetime.datetime.now()  # 获得当前时间
    timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    print('年_月_日_时_分_秒：', timestr)
    dir = os.getcwd() + '/logs/ traininglog_' + timestr  # os.getcwd()获得当前执行目录
    if os.path.exists(dir):  # 看文件夹是否存在
        print('文件夹已存在')
    else:  # 如果不存在
        os.makedirs(dir)  # 则创建文件夹
    dir = dir + '/log.txt'
    logger = get_log(dir)
    logger.info("训练的batch_size以及lr分别是%d,%d" % (batch_size,lr))

    data_path = r'data'
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)
    save_path = r'checkpoints/'
    #  logger = get_log('D:\log\log.log')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 读取分类类别
    with open('char_dict', 'rb') as f:
        class_dict = pickle.load(f) # 将字节流转换成对象，即反串行化。
    num_classes = len(class_dict)

    # 读取数据
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = HWDB(path=data_path, transform=transform)
    print("训练集数据:", dataset.train_size)
    print("测试集数据:", dataset.test_size)
    trainloader, testloader = dataset.get_loader(batch_size)

    net = ConvNet(num_classes)
    if torch.cuda.is_available():
        net = net.cuda()

    print('网络结构：\n')
    summary(net, input_size=(3, 64, 64), device='cuda')
    criterion = nn.CrossEntropyLoss() #  交叉熵主要是用来判定 实际的输出与期望的输出的接近程度
    optimizer = optim.SGD(net.parameters(), lr=lr) #   SGD就是optim中的一个算法（优化器）：随机梯度下降算法
    #当训练数据N很大时，计算总的cost function来求梯度代价很大，所以一个常用的方法是计算训练集中的小批量（minibatches），这就是SGD。
    writer = SummaryWriter(log_path)
    for epoch in range(epochs):
        train(epoch, net, criterion, optimizer, trainloader, writer=writer, logger=logger)
        valid(epoch, net, testloader, writer=writer, logger=logger)
        print("epoch%d 结束, 正在保存模型..." % epoch)
        torch.save(net.state_dict(), save_path + 'handwriting_iter_%03d.pth' % epoch)
