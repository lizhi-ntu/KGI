import torch
import torch.nn as nn
import torch.nn.functional as F


def pred_to_onehot(prediction):
    size = prediction.shape
    prediction_max = torch.argmax(prediction, dim=1)
    oneHot_size = (size[0], 13, size[2], size[3])
    pred_onehot = torch.FloatTensor(torch.Size(oneHot_size)).zero_().cuda()
    pred_onehot = pred_onehot.scatter_(1, prediction_max.unsqueeze(1).data.long(), 1.0)
    return pred_onehot

class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_nc, out_nc, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class ParseGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ParseGenerator, self).__init__()
        
        self.Encoder = nn.Sequential(
            ResBlock(input_nc, ngf, norm_layer=norm_layer, scale='down'),    # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),      # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')   # 8
        )
          
        self.SegDecoder = nn.Sequential(
            ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
            ResBlock(ngf * 4 * 2, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
            ResBlock(ngf * 4 * 2, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
            ResBlock(ngf * 2 * 2, ngf, norm_layer=norm_layer, scale='up'),  # 128
            ResBlock(ngf * 1 * 2, ngf, norm_layer=norm_layer, scale='up')  # 256
        )
        
        self.out_layer = nn.Sequential(
            ResBlock(ngf + input_nc, ngf, norm_layer=norm_layer, scale='same'),
            nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
        )
         
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        
    def normalize(self, x):
        return x
    
    def forward(self, input_):
        E_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E_list.append(self.Encoder[i](input_))
            else:
                E_list.append(self.Encoder[i](E_list[i - 1]))

        # Segmentation Network
        for i in range(5):
            if i == 0:
                x = self.conv(E_list[4 - i])
                x = self.SegDecoder[i](x)

            else:
                x = self.SegDecoder[i](torch.cat([x, E_list[4-i]], 1))
        
        x = self.out_layer(torch.cat([x, input_], 1))
        x = F.softmax(x, dim=1)

        return x

def main():
    network = ParseGenerator(input_nc=19, output_nc=13)
    network.cuda()

    input_ = torch.zeros([1, 19, 256, 192]).cuda()
    output = network(input_)
    print(output, output.shape)
    output = pred_to_onehot(output)
    print(output, output.shape)

if __name__ == "__main__":
    main()