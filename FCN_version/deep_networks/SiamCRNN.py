from deep_networks.resnet_18_34 import ResNet34, ResNet18
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn


class SiamCRNN(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, pretrained=True, output_stride=16, BatchNorm=nn.BatchNorm2d, bias=True):
        super(SiamCRNN, self).__init__()
        self.encoder_1 = ResNet34(input_dim=in_dim_1, BatchNorm=BatchNorm, pretrained=False, output_stride=output_stride)
        # self.encoder_1.conv1 = nn.Conv2d(in_dim_1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # If your dataset is heterogeneous, then please utilize pesudo-siamese architecture
        # self.encoder_2 = ResNet18(input_dim=in_dim_2, BatchNorm=BatchNorm, pretrained=True, output_stride=output_stride)
        self.convlstm_4 = ConvLSTM(input_dim=512, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_3 = ConvLSTM(input_dim=256, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_2 = ConvLSTM(input_dim=128, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.convlstm_1 = ConvLSTM(input_dim=64, hidden_dim=128, kernel_size=(3, 3), num_layers=1,
                                   batch_first=True)
        self.smooth_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
                                            nn.BatchNorm2d(128), nn.ReLU())
        self.smooth_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
                                            nn.BatchNorm2d(128), nn.ReLU())
        self.smooth_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=3, in_channels=128, out_channels=128, padding=1),
                                            nn.BatchNorm2d(128), nn.ReLU())

        self.main_clf_1 = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        pre_low_level_feat_1, pre_low_level_feat_2, pre_low_level_feat_3, pre_output = \
            self.encoder_1(pre_data)
        post_low_level_feat_1, post_low_level_feat_2, post_low_level_feat_3, post_output = \
            self.encoder_1(post_data)

        # Concatenate along the time dimension
        combined_4 = torch.stack([pre_output, post_output], dim=1)
        # Apply ConvLSTM
        _, last_state_list_4 = self.convlstm_4(combined_4)
        p4 = last_state_list_4[0][0]

        combined_3 = torch.stack([pre_low_level_feat_3, post_low_level_feat_3], dim=1)
        # Apply ConvLSTM
        _, last_state_list_3 = self.convlstm_3(combined_3)
        p3 = last_state_list_3[0][0]
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        combined_2 = torch.stack([pre_low_level_feat_2, post_low_level_feat_2], dim=1)
        # Apply ConvLSTM
        _, last_state_list_2 = self.convlstm_2(combined_2)
        p2 = last_state_list_2[0][0]
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        combined_1 = torch.stack([pre_low_level_feat_1, post_low_level_feat_1], dim=1)
        # Apply ConvLSTM
        _, last_state_list_1 = self.convlstm_1(combined_1)
        p1 = last_state_list_1[0][0]
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        output_1 = self.main_clf_1(p1)
        output_1 = F.interpolate(output_1, size=pre_data.size()[-2:], mode='bilinear')
        return output_1


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
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
