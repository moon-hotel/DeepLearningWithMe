"""
文件名: Code/Chapter08/C05_ConvLSTM/ConvLSTM.py
创建时间: 2023/6/3 2:35 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
Shi X, Chen Z, Wang H, et al. Convolutional LSTM network: A machine learning approach for precipitation nowcasting [J].
Advances in neural information processing systems, 2015, 28.
implementation by: https://github.com/ndrplz/ConvLSTM_pytorch
"""
import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        in_channels: int 输入特征图的通道数
        out_channels: int 输出特征图的通道数
        kernel_size: (int, int) 卷积核的宽和高
        bias: bool 是否使用偏置
        """
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # 需要强制进行padding以保证每次卷积后形状不发生变化
        # 根据之前第4.3.2节内容的介绍，在stride=1的情况下，padding = kernel_size // 2
        # 如：卷积核为3×3则需要padding=1即可
        # 在下面的卷积操作中stride使用的是默认值1
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.out_channels,
                              out_channels=4 * self.out_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, last_state):
        """

        :param input_tensor: 当前时刻的输入x_t, 形状为[batch_size, in_channels, height, width]
        :param last_state: 上一时刻的状态c_{t-1}和h_{t-1}, 形状均为 [batch_size, out_channels, height, width]
        :return:
        """
        h_last, c_last = last_state
        combined_input = torch.cat([input_tensor, h_last], dim=1)
        # [batch_size, in_channels+out_channels, height, width]
        combined_conv = self.conv(combined_input)  # [batch_size, 4 * out_channels, height, width]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1)
        # 分割得到每个门对应的卷积计算结果，形状均为 [batch_size, out_channels, height, width]
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_last + i * g  # [batch_size, out_channels, height, width]
        h_next = o * torch.tanh(c_next)  # [batch_size, out_channels, height, width]
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        初始化记忆单元的C和H
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.out_channels, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        in_channels: 输入特征图的通道数，为整型
        out_channels: 每一层输出特征图的通道数，可为整型也可以是列表；
                      为整型时表示每一层的输出通道数均相等，为列表时则列表的长度必须等于num_layer
                      例如 out_channels =[32,64,128] 表示3层ConvLSTM的输出特征图通道数分别为
                      32、64和128，且此时的num_layer也必须为3
        kernel_size:  每一层中卷积核的长和宽，可以为一个tuple，如(3,3)表示每一层的卷积核窗口大小均为3x3；
                      也可以是一个列表分别用来指定每一层卷积核的大小，如[(3,3),(5,5),(7,7)]表示3层卷积各种的窗口大小
                      此时需要注意的是，如果为列表也报保证其长度等于num_layer
        num_layers: ConvLSTM堆叠的层数
        batch_first: 输入数据的第1个维度是否为批大小
        bias: 卷积中是否使用偏置
        return_all_layers: 是否返回每一层各个时刻的输出结果

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
        [Batch_size, Time_step, Channels, Height, Width]  or [Time_step, Batch_size, Channels, Height, Width]
    Output:
        当return_all_layers 为 True 时：
        layer_output_list: 每一层的输出结果，包含有num_layer个元素的列表，
                           每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_states: 每一层最后一个时刻的输出结果，同样是包含有num_layer个元素的列表，
                     列表中的每个元素均为一个包含有两个张量的列表，
                     如last_states[-1][0]和last_states[-1][1]分别表示最后一层最后一个时刻的h和c
                     layer_output_list[-1][:, -1] == last_states[-1][0]
                     shape:  [Batch_size, Channels, Height, Width]

        当return_all_layers 为 False 时：
        layer_output_list: 最后一层每个时刻的输出，形状为 [batch_size, time_step, out_channels, height, width]
        last_states: 最后一层最后一个时刻的输出，形状为 [batch_size, out_channels, height, width]

    Example:
        >> model = ConvLSTM(in_channels=3,
                 out_channels=2,
                 kernel_size=(3, 3),
                 num_layers=3,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
        x = torch.rand((1, 4, 3, 5, 5)) # [batch_size, time_step, channels, height, width]
        layer_output_list, last_states = model(x)
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # 检查kernel_size是否符合上面说的取值情况

        # Make sure that both `kernel_size` and `out_channels` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        # 将kernel_size和out_channels扩展到多层时的情况

        if not len(kernel_size) == len(out_channels) == num_layers:
            raise ValueError('len(kernel_size) == len(out_channels) == num_layers 三者的值必须相等')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 实例化每一层的ConvLSTM记忆单
            cur_in_channels = self.in_channels if i == 0 else self.out_channels[i - 1]
            # 当前层的输入通道数，除了第一层为self.in_channels之外，其它的均为上一层的输出通道数

            cell_list.append(ConvLSTMCell(in_channels=cur_in_channels, out_channels=self.out_channels[i],
                                          kernel_size=self.kernel_size[i], bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        # 必须要放到nn.ModuleList，否则在GPU上云运行时会报错张量不在同一个设备上的问题

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor: [Batch_size, Time_step, Channels, Height, Width]  or
                        [Time_step, Batch_size, Channels, Height, Width]
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # 将(t, b, c, h, w) 转为 (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        batch_size, time_step, _, height, width = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=batch_size,
                                             image_size=(height, width))

        layer_output_list = []  # 保存每一层的输出h，每个元素的形状为[batch_size, time_step, out_channels, height, width]
        last_state_list = []  # 保存每一层最后一个时刻的输出h和c，即[(h,c),(h,c)...]
        cur_layer_input = input_tensor  # [batch_size, time_step, in_channels, height, width]
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]  # 开始遍历每一层的ConvLSTM记忆单元，并取对应的初始值
            # h 和 c 的形状均为[batch_size, out_channels, height, width]
            output_inner = []
            cur_layer_cell = self.cell_list[layer_idx]  # 为一个ConvLSTMCell记忆单元
            for t in range(time_step):  # 对于每一层的记忆单元，按照时间维度展开进行计算
                h, c = cur_layer_cell(input_tensor=cur_layer_input[:, t, :, :, :], last_state=[h, c])
                output_inner.append(h)  # 当前层，每个时刻的输出h, 形状为 [batch_size, out_channels, height, width]

            layer_output = torch.stack(output_inner, dim=1)  # [batch_size, time_step, out_channels, height, width]
            cur_layer_input = layer_output  # 当前层的输出h，作为下一层的输入
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        init_states中的每个元素为一个tuple，包含C和H两个部分，如 [(h,c),(h,c)...]
        形状均为 [batch_size, out_channels, height, width]
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):  # 初始化每一层的初始值
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMKTH(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.conv_lstm = ConvLSTM(config.in_channels, config.out_channels,
                                  config.kernel_size, config.num_layers, config.batch_first)
        self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=2)
        self.hidden_dim = (config.width * config.height) // 4 * self.conv_lstm.out_channels[-1]
        # 除以4是因为长宽均要除以stride, 使用self.conv_lstm.out_channels[-1]
        # 主要是为了兼容out_channels传入整型或列表的情况，因为传入整型的话在ConvLSTM的初始化方法中_extend_for_multilayer()
        # 方法也会将其扩充一个list
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(self.hidden_dim, config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, time_step, channels, height, width]
        :param labels: [batch_size,]
        :return: logits: [batch_size, num_classes]
        """
        _, layer_output = self.conv_lstm(x)
        # layer_output: [h:[batch_size, out_channels, height, width], c:[batch_size, out_channels, height, width]]
        pool_output = self.max_pool(layer_output[-1][0])  # [batch_size, out_channels, height//2, width//2]
        logits = self.classifier(pool_output)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.num_classes = 6
        self.in_channels = 3
        self.out_channels = [32, 16, 8]
        self.kernel_size = [(3, 3), (5, 5), (7, 7)]
        self.num_layers = len(self.out_channels)
        self.batch_size = 8
        self.height = 120
        self.width = 160
        self.batch_first = True
        self.time_step = 15


if __name__ == '__main__':
    config = ModelConfig()
    model = ConvLSTMKTH(config)
    x = torch.randn([config.batch_size, config.time_step,
                     config.in_channels, config.height,
                     config.width])  # [batch_size, time_step, channels, height, width]
    label = torch.randint(0, config.num_classes, [config.batch_size], dtype=torch.long)
    loss, logits = model(x, label)
    print(loss)
    print(logits)
