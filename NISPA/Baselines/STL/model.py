import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import numpy as np
from config import DEVICE

class ConvNet(nn.Module):
    def __init__(self, input_channels, output_dim, conv2lin_kernel_size = 64):
        super(ConvNet, self).__init__()
        self.conv2lin_kernel_size = conv2lin_kernel_size
        self.conv1 = MaskedConv2dDynamic(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = MaskedConv2dDynamic(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv3 = MaskedConv2dDynamic(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = MaskedConv2dDynamic(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        
        
        self.fc1 = MaskedLinearDynamic(128 * self.conv2lin_kernel_size, 1024)
        self.fc2 = MaskedLinearDynamic(1024, output_dim)
        self.output_dim = output_dim
        self._initialize_weights()
        
    def forward_activations(self, x, current_classes = None):
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(F.relu(self.conv2(x1)), 2)
        
        x3 = F.relu(self.conv3(x2))
        x4 = F.max_pool2d(F.relu(self.conv4(x3)), 2)
        
        
        #Flatten
        x4_flat = x4.view(-1, 128 *  self.conv2lin_kernel_size)
        x5 = F.relu(self.fc1(x4_flat))
        x6 = self.fc2(x5)
        
        if current_classes is not None:
            mask = torch.zeros(self.output_dim)
            mask[current_classes] = 1
            x6 = x6 * mask.to(DEVICE)
    
        return x6, [x1, x2, x3, x4, x5, x6]
        
    def forward(self, x, current_classes = None, save_activations = False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
    
        
        #Flatten
        x = x.view(-1, 128 *  self.conv2lin_kernel_size)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if current_classes is not None:
            mask = torch.zeros(self.output_dim)
            mask[current_classes] = 1
            x = x * mask.to(DEVICE)
    
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (MaskedLinearDynamic, MaskedConv2dDynamic)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def re_initialize(self, freeze_masks):
        i = 0
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                old_weights = m.weight.data.clone().detach()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data[torch.tensor(freeze_masks[i][0]).to(torch.bool)] = old_weights[torch.tensor(freeze_masks[i][0]).to(torch.bool)]
                
                old_bias = m.bias.clone().detach()
                nn.init.constant_(m.bias.data, 0)
                m.bias.data[torch.tensor(freeze_masks[i][1]).to(torch.bool)] =  old_bias[torch.tensor(freeze_masks[i][1]).to(torch.bool)]
                i += 1

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinearDynamic, MaskedConv2dDynamic)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = MaskedLinearDynamic(input_dim, 400)
        self.layer2 = MaskedLinearDynamic(400, 400)
        self.layer3 = MaskedLinearDynamic(400, 400)
        self.layer4 = MaskedLinearDynamic(400, output_dim)
        self.output_dim = output_dim
        self._initialize_weights()   
        
    def forward_activations(self, x, current_classes = None):
         
        #Forward pass as usual
        x = x.view(-1, 28 * 28)
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = F.relu(self.layer3(x2))
        x4 = self.layer4(x3)
        
        if current_classes is not None:
            mask = torch.zeros(self.output_dim)
            mask[current_classes] = 1
            x4 = x4 * mask.to(DEVICE)
            
        return x4, [x, x1, x2, x3, x4]
        
        
    def forward(self, x, current_classes = None):
        
        #Forward pass as usual
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        
        if current_classes is not None:
            mask = torch.zeros(self.output_dim)
            mask[current_classes] = 1
            x_masked = x * mask.to(DEVICE)
            return x_masked
        return x
    
    def re_initialize(self, freeze_masks):
        i = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                old_weights = m.weight.data.clone().detach()
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data[torch.tensor(freeze_masks[i][0]).to(torch.bool)] = old_weights[torch.tensor(freeze_masks[i][0]).to(torch.bool)]
                
                old_bias = m.bias.clone().detach()
                nn.init.constant_(m.bias.data, 0)
                m.bias.data[torch.tensor(freeze_masks[i][1]).to(torch.bool)] =  old_bias[torch.tensor(freeze_masks[i][1]).to(torch.bool)]
                i += 1
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def set_masks(self, weight_mask,bias_mask):
       i = 0
       for module in self.modules():
           if isinstance(module, nn.Linear):
               module.set_mask(weight_mask[i],bias_mask[i])
               i = i + 1
    
    
def to_var(x, requires_grad = False, volatile = False):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return Variable(x, requires_grad = requires_grad, volatile = volatile)


class MaskedLinearDynamic(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinearDynamic, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.bias_flag = bias
        self.sparse_grads = True
        
    def set_mask(self, weight_mask, bias_mask):
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag == True:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data
        self.mask_flag = True

    def get_mask(self):
        return self.weight_mask, self.bias_mask

    def forward(self, x):
        if self.mask_flag == True and self.sparse_grads:
            weight = self.weight * self.weight_mask
            if self.bias_flag == True:
                bias = self.bias * self.bias_mask
            else:
                bias = self.bias
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)
        
class MaskedConv2dDynamic(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2dDynamic, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.bias_flag = bias

    def set_mask(self, weight_mask, bias_mask):
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag == True:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data
        self.mask_flag = True

    def get_mask(self):
        return self.weight_mask, self.bias_mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.weight_mask
            if self.bias_flag == True:
                bias = self.bias * self.bias_mask
            else:
                bias = self.bias
            return F.conv2d(x, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

               
    
#No Bias pruning
def random_prune(model, pruning_perc, skip_first = True):
    if pruning_perc > 0.0:
        model = copy.deepcopy(model)
        pruning_perc = pruning_perc / 100.0
        weight_masks = []
        bias_masks = []
    first_conv = skip_first
    for module in model.modules():
        if isinstance(module, MaskedLinearDynamic):
            weight_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.weight.shape,
                                            p =  [pruning_perc, 1 - pruning_perc]))
            weight_masks.append(weight_mask)
            #do not prune biases
            bias_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.bias.shape,
                                            p =  [0, 1]))
            bias_masks.append(bias_mask)
        #Channel wise pruning Conv Layer
        elif isinstance(module, MaskedConv2dDynamic):
           if first_conv:
               connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                   (module.weight.shape[0],  module.weight.shape[1]),
                                                    p =  [0, 1]))
               first_conv = False
           else:
               connectivity_mask = torch.from_numpy(np.random.choice([0, 1],
                                                   (module.weight.shape[0],  module.weight.shape[1]),
                                                    p =  [pruning_perc, 1 - pruning_perc]))
           filter_masks = []
           for conv_filter in range(module.weight.shape[0]):
              
               filter_mask = []
               for inp_channel  in range(module.weight.shape[1]):
                   if connectivity_mask[conv_filter, inp_channel] == 1:
                       filter_mask.append(np.ones((module.weight.shape[2], module.weight.shape[3])))
                   else:
                       filter_mask.append(np.zeros((module.weight.shape[2], module.weight.shape[3])))
               filter_masks.append(filter_mask)
               
           weight_masks.append(torch.from_numpy(np.array(filter_masks)).to(torch.float32))
           
           #do not prune biases
           bias_mask = torch.from_numpy(np.random.choice([0, 1],
                                            module.bias.shape,
                                            p =  [0, 1])).to(torch.float32)
           bias_masks.append(bias_mask)
    model.set_masks(weight_masks, bias_masks)
    model.to(DEVICE)
    return model
 
    
