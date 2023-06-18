from ConvLSTM import ConvLSTM
import torch


def example1():
    in_channels = 3
    out_channels = [5, 6, 7]
    kernel_size = [(3, 3), (5, 5), (7, 7)]
    num_layers = 3
    batch_size = 1
    time_step = 4
    height, width = 16, 16
    x = torch.rand((batch_size, time_step, in_channels, height, height))
    model = ConvLSTM(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     num_layers=num_layers,
                     batch_first=True,
                     bias=True,
                     return_all_layers=True)

    layer_output_list, last_states = model(x)
    print(last_states[-1][0])
    print(layer_output_list[-1][:, -1])

def example2():
    in_channels = 3
    out_channels = 5
    kernel_size = (3, 3)
    num_layers = 3
    batch_size = 1
    time_step = 4
    height, width = 16, 16
    x = torch.rand((batch_size, time_step, in_channels, height, height))
    model = ConvLSTM(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     num_layers=num_layers,
                     batch_first=True,
                     bias=True,
                     return_all_layers=True)

    layer_output_list, last_states = model(x)
    print(last_states[-1][0])
    print(layer_output_list[-1][:, -1])

if __name__ == '__main__':
    example1()
    # example2()

