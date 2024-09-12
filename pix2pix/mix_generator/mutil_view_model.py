import torch
import torch.nn as nn
import torch.nn.functional as F



class TripleCrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(TripleCrossAttention, self).__init__()
        self.cross_attn1 = CrossAttention(in_dim)
        self.cross_attn2 = CrossAttention(in_dim)

    def forward(self, x, y, z):
        # Apply cross attention between x and y
        out1 = self.cross_attn1(x, y)

        # Apply cross attention between the result of the first cross attention and z
        out2 = self.cross_attn2(out1, z)

        return out2

class CrossAttention(nn.Module):
    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_dim, num_heads=1)

    def forward(self, x, y):
        # x: query, y: key, value

        # Cross Attention
        proj_query = self.query_conv(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)  # B * N * C
        proj_key = self.key_conv(y).view(y.size(0), -1, y.size(2) * y.size(3))  # B * C * N
        energy = torch.bmm(proj_query, proj_key)  # batch matrix-matrix product
        attention = torch.softmax(energy, dim=2)  # attention
        proj_value = self.value_conv(y).view(y.size(0), -1, y.size(2) * y.size(3))  # B * C * N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # batch matrix-matrix product
        out = out.view(y.size(0), x.size(1), x.size(2), x.size(3))  # B * C * H * W

        # Residual connection
        out = self.gamma * out + x

        return out
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, depth, height = x.size()
        query = self.query(x).view(batch_size, -1, depth * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, depth * height)
        #print(query.shape,key.shape)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, depth * height)
        #print(attention.shape,value.shape)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height)
        out = self.gamma * out + x

        return out

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
class Generator(nn.Module):
    def __init__(self,d = 64):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(2,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#16->8
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#8->4
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#4->2
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#2->1

        #self.sa1 =SelfAttention(64)
        #self.sa2 = SelfAttention(128)
        #self.sa3 = SelfAttention(256)
        self.sa1 = SelfAttention(512)
        self.sa2 = SelfAttention(512)
        self.sa3 = SelfAttention(512)
        #self.sa4 = SelfAttention(512)


        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)#1->2
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#2->4
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#4->8
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#8->16
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)#16->32
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)#32->64
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)#64->128
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)#128->256

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input):
        e1 = self.conv1(input)
        #e1 = self.sa1(e1)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        #e2 = self.sa2(e2)
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        #e3 = self.sa3(e3)
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e4 = self.sa1(e4)
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        #e5 = self.sa5(e5)
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        #e6 = self.sa6(e6)
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        #e7 = self.sa7(e7)
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        e8 = self.sa2(e8)
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d5 = self.sa3(d5)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        out = torch.tanh(d8)
        return out


class wGenerator(nn.Module):
    def __init__(self,d = 64):
        super(wGenerator,self).__init__()
        self.conv1 = nn.Conv2d(2,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv_mix = nn.Conv2d(d*8*2,d*8,1,1,0)
        self.conv_mix_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#16->8
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#8->4
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#4->2
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#2->1

        #self.sa1 =SelfAttention(64)
        #self.sa2 = SelfAttention(128)
        #self.sa3 = SelfAttention(256)
        self.sa1 = SelfAttention(512)
        self.sa2 = SelfAttention(512)
        self.sa3 = SelfAttention(512)
        #self.sa4 = SelfAttention(512)


        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)#1->2
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#2->4
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#4->8
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#8->16
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)#16->32
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)#32->64
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)#64->128
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)#128->256

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input,z):
        e1 = self.conv1(input)
        #e1 = self.sa1(e1)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        #e2 = self.sa2(e2)
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        #e3 = self.sa3(e3)
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        #e4 = F.leaky_relu(e4,0.2)
        #print('dsa',e4.shape,z.shape)
        x = torch.cat([e4,z],1)
        x = self.conv_mix_bn(self.conv_mix(F.leaky_relu(x,0.2)))

        e5 = self.conv5_bn(self.conv5(F.leaky_relu(x, 0.2)))
        e5 = self.sa1(e5)
        #e5 = self.sa5(e5)
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        #e6 = self.sa6(e6)
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        #e7 = self.sa7(e7)
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        e8 = self.sa2(e8)
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, x], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d5 = self.sa3(d5)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        out = torch.tanh(d8)
        return e4,e8,out

class Mutil_Basic_Generator(nn.Module):
    def __init__(self,in_channels,d = 64):
        super(Mutil_Basic_Generator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 4, 4, 2, 1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d * 4)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#16->8
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#8->4
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#4->2
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)#2->1

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)#1->2
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#2->4
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#4->8
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)#8->16
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)#16->32
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)#32->64
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)#64->128
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 1, 4, 2, 1)#128->256
        self.sa1 = SelfAttention(512)
        self.sa2 = SelfAttention(512)
        self.sa3 = SelfAttention(512)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input,z):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e4 = F.leaky_relu(e4,0.2)
        #print(e4.shape,z.shape)
        x = torch.cat([e4,z],1)
        x = self.sa1(x)
        e5 = self.conv5_bn(self.conv5(x))

        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        e8 = self.sa2(e8)
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, x], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d5 = self.sa3(d5)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        out = torch.tanh(d8)
        return e4,out


class Discriminator(nn.Module):
    def __init__(self,in_channels,d = 64):
        super(Discriminator,self).__init__()
        self.conv1  = nn.Conv2d(in_channels,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4,d*8,4,2,1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8,d*8,4,1,1)#16->15
        self.conv5_bn = nn.BatchNorm2d(d*8)
        self.conv6 = nn.Conv2d(d*8,1,4,1,1)#15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input,label):
        x = torch.cat([input,label],1)
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = torch.sigmoid(self.conv6(x))
        return x


class Mutil_Discriminator(nn.Module):
    def __init__(self,in_channels,d = 64):
        super(Mutil_Discriminator,self).__init__()
        self.conv1  = nn.Conv2d(in_channels,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4,d*8,4,2,1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8,d*8,4,1,1)#16->15
        self.conv5_bn = nn.BatchNorm2d(d*8)
        self.conv6 = nn.Conv2d(d*8,1,4,1,1)#15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,up,left,right):
        x = torch.cat([up,left,right],1)
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = torch.sigmoid(self.conv6(x))
        return x


class LocalDiscriminator(nn.Module):
    def __init__(self,d = 128):
        super(LocalDiscriminator, self).__init__()
        self.conv1  = nn.Conv2d(1,d,4,2,1)#128->64
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#64->32
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#32->16
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, d * 4, 4, 1, 1)  # 16->15
        self.conv4_bn = nn.BatchNorm2d(d * 4)
        self.conv5 = nn.Conv2d(d * 4, 1, 4, 1, 1)  # 15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x




class WDiscriminator(nn.Module):
    def __init__(self,in_channels,d = 64):
        super(WDiscriminator,self).__init__()
        self.conv1  = nn.Conv2d(in_channels,d,4,2,1)#256->128
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#128->64
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#64->32
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4,d*8,4,2,1)#32->16
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8,d*8,4,1,1)#16->15
        self.conv5_bn = nn.BatchNorm2d(d*8)
        self.conv6 = nn.Conv2d(d*8,1,4,1,1)#15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = self.conv6(x)
        return x
    def feature_extraction(self,x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        return x.view(-1,512*15*15)


class LocalWDiscriminator(nn.Module):
    def __init__(self,d = 128):
        super(LocalWDiscriminator, self).__init__()
        self.conv1  = nn.Conv2d(1,d,4,2,1)#128->64
        self.conv2 = nn.Conv2d(d,d*2,4,2,1)#64->32
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2,d*4,4,2,1)#32->16
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, d * 4, 4, 1, 1)  # 16->15
        self.conv4_bn = nn.BatchNorm2d(d * 4)
        self.conv5 = nn.Conv2d(d * 4, 1, 4, 1, 1)  # 15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,x):
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x)
        return x

    def feature_extraction(self,x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        return x.view(-1,512*15*15)



class DCGenerator(nn.Module):
    def __init__(self,d = 64):
        super(DCGenerator, self).__init__()
        self.conv1 = nn.Conv3d(2, d, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv3d(d, d * 2, 4, 2, 1)  # 128->64
        self.conv2_bn = nn.BatchNorm3d(d * 2)
        self.conv3 = nn.Conv3d(d * 2, d * 4, 4, 2, 1)  # 64->32
        self.conv3_bn = nn.BatchNorm3d(d * 4)
        self.conv4 = nn.Conv3d(d * 4, d * 8, 4, 2, 1)  # 32->16
        self.conv4_bn = nn.BatchNorm3d(d * 8)
        self.conv5 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)  # 16->8
        self.conv5_bn = nn.BatchNorm3d(d * 8)
        self.conv6 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)  # 8->4
        self.conv6_bn = nn.BatchNorm3d(d * 8)
        self.conv7 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)  # 4->2
        self.conv7_bn = nn.BatchNorm3d(d * 8)
        self.conv8 = nn.Conv3d(d * 8, d * 8, 4, 2, 1)  # 2->1

        self.deconv1 = nn.ConvTranspose3d(d * 8, d * 8, 4, 2, 1)  # 1->2
        self.deconv1_bn = nn.BatchNorm3d(d * 8)
        self.deconv2 = nn.ConvTranspose3d(d * 8 * 2, d * 8, 4, 2, 1)  # 2->4
        self.deconv2_bn = nn.BatchNorm3d(d * 8)
        self.deconv3 = nn.ConvTranspose3d(d * 8 * 2, d * 8, 4, 2, 1)  # 4->8
        self.deconv3_bn = nn.BatchNorm3d(d * 8)
        self.deconv4 = nn.ConvTranspose3d(d * 8 * 2, d * 8, 4, 2, 1)  # 8->16
        self.deconv4_bn = nn.BatchNorm3d(d * 8)
        self.deconv5 = nn.ConvTranspose3d(d * 8 * 2, d * 4, 4, 2, 1)  # 16->32
        self.deconv5_bn = nn.BatchNorm3d(d * 4)
        self.deconv6 = nn.ConvTranspose3d(d * 4 * 2, d * 2, 4, 2, 1)  # 32->64
        self.deconv6_bn = nn.BatchNorm3d(d * 2)
        self.deconv7 = nn.ConvTranspose3d(d * 2 * 2, d, 4, 2, 1)  # 64->128
        self.deconv7_bn = nn.BatchNorm3d(d)
        self.deconv8 = nn.ConvTranspose3d(d * 2, 1, 4, 2, 1)  # 128->256


    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        out = torch.tanh(d8)
        return out

class DCDiscriminator(nn.Module):
    def __init__(self,d = 64):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Conv3d(3, d, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv3d(d, d * 2, 4, 2, 1)  # 128->64
        self.conv2_bn = nn.BatchNorm3d(d * 2)
        self.conv3 = nn.Conv3d(d * 2, d * 4, 4, 2, 1)  # 64->32
        self.conv3_bn = nn.BatchNorm3d(d * 4)
        self.conv4 = nn.Conv3d(d * 4, d * 8, 4, 2, 1)  # 32->16
        self.conv4_bn = nn.BatchNorm3d(d * 8)
        self.conv5 = nn.Conv3d(d * 8, d * 8, 4, 1, 1)  # 16->15
        self.conv5_bn = nn.BatchNorm3d(d * 8)
        self.conv6 = nn.Conv3d(d * 8, 1, 4, 1, 1)  # 15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = torch.sigmoid(self.conv6(x))
        return x

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_layer = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #print(x.shape)
        b, c, _, _ = x.size()
        # 全局平均池化 (4,256,16,16) -> (4,256,1,1) -> (4,256)
        y = self.avg_pool(x).view(b, c)
        #print(y.shape)
        attention_weights = self.attention_layer(y)
        #print(attention_weights.shape)
        attention_weights = self.softmax(attention_weights)
        attention_weights = attention_weights.view(b,1,1,1)
        attended_values = x * attention_weights
        return attended_values

class Mutil_view_Generator(nn.Module):
    def __init__(self,d = 64):
        super(Mutil_view_Generator, self).__init__()
        self.up_Model = wGenerator()
        self.left_Model = wGenerator()
        self.right_Model = wGenerator()

        self.triple_cross_attn = TripleCrossAttention(in_dim=512)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,up,left,right):
        _z = torch.rand(up.shape[0],512,16,16).cuda()
        up_e4,_,_ = self.up_Model(up,_z)
        left_e4,_,_ = self.left_Model(left,_z)
        right_e4,_,_ = self.right_Model(right,_z)
        z = self.triple_cross_attn(up_e4,left_e4,right_e4)

        _, up_map, _ = self.up_Model(up, z)
        _, left_map, _ = self.left_Model(left, z)
        _, right_map, _ = self.right_Model(right, z)
        #预训练的输出
        # _,_,out_up = self.up_Model(up,z)
        # _,_,out_left = self.left_Model(left,z)
        # _,_,out_right = self.right_Model(right,z)
        return up_map,left_map,right_map


class Mutil_view_Generator_2(nn.Module):
    def __init__(self,d = 64):
        super(Mutil_view_Generator_2, self).__init__()
        self.up_Model = Generator()
        self.left_Model = Generator()
        self.right_Model = Generator()

        #self.triple_cross_attn = TripleCrossAttention(in_dim=256)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,up,left,right):
        out_up = self.up_Model(up)
        out_left = self.left_Model(left)
        out_right = self.right_Model(right)
        return out_up,out_left,out_right


class DSConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(DSConvLayer, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,dilation=rate, padding=rate) for rate in dilation_rates
        ])

    def forward(self, x):
        conv_outputs = [conv_layer(x) for conv_layer in self.conv_layers]
        concatenated = torch.cat(conv_outputs, dim=1)
        return concatenated


class CustomLayer(nn.Module):
    def __init__(self, in_channels=128, out_channels=32, kernel_size=3, stride=1):
        super(CustomLayer, self).__init__()

        # Conv2d layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

        # BatchNorm2d layer
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Convolutional layer
        conv_output = self.conv(x)

        # Batch normalization
        batch_norm_output = self.batch_norm(conv_output)

        # Elementwise SUM
        output = torch.add(x, batch_norm_output)

        return output


class Optimiza_Generator(nn.Module):
    def __init__(self):
        #padding还要改
        super(Optimiza_Generator, self).__init__()
        self.preli_extra = nn.Conv2d(1,32,9,1,padding=4)
        self.multi1_dsconv = DSConvLayer(32,32,dilation_rates=[1,2,4,8])
        self.custom_layer1 = CustomLayer()
        self.multi2_dsconv = DSConvLayer(32,32,dilation_rates=[1,2,4,8])
        self.custom_layer2 = CustomLayer()
        self.multi3_dsconv = DSConvLayer(32,32,dilation_rates=[1,2,4,8])
        self.custom_layer3 = CustomLayer()
        self.multi4_dsconv = DSConvLayer(32,32,dilation_rates=[1,2,4,8])
        self.custom_layer4 = CustomLayer()
        self.prefinal_layer = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32)
        )
        self.final_conv = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)


    def forward(self,input):
        x = self.preli_extra(input)
        preli_future = F.relu(x)
        x = self.multi1_dsconv(x)
        x = self.custom_layer1(x)
        x = self.multi2_dsconv(x)
        x = self.custom_layer2(x)
        x = self.multi3_dsconv(x)
        x = self.custom_layer3(x)
        x = self.multi4_dsconv(x)
        x = self.custom_layer4(x)
        x = self.prefinal_layer(x)
        x = torch.add(x,preli_future)
        out = self.final_conv(x)

        return out


class Optimiza_Discriminator(nn.Module):
    def __init__(self):
        super(Optimiza_Discriminator,self).__init__()



#PConv
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not是否使用多通道mask
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False
        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = True
        super(PartialConv2d, self).__init__(*args, **kwargs)
        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)
            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)
                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)
        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)
        if self.return_mask:
            return output, self.update_mask
        else:
            return output

#Gated convolution
class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class PConvBNActiv(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, sample='none-3', activ='relu', bias=False):
        super(PConvBNActiv, self).__init__()
        if sample == 'down-7':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=bias, multi_channel = True)
        elif sample == 'down-5':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=bias, multi_channel = True)
        elif sample == 'down-3':
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias, multi_channel = True)
        elif sample == 'down-4':
            self.conv = PartialConv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1,bias=bias,multi_channel=True)
        else:
            self.conv = PartialConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, multi_channel = True)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images, masks):
        images, masks = self.conv(images, masks)
        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        return images, masks


class PUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, up_sampling_node='nearest'):
        super(PUNet, self).__init__()
        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = PConvBNActiv(in_channels, 64, bn=False, sample='down-7')
        self.ec_images_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_images_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_images_4 = PConvBNActiv(256, 512, sample='down-3')

        self.conv_mix = nn.Conv2d(512+512, 512, 3, 1, 1)
        self.conv_mix_bn = nn.BatchNorm2d(512)

        self.ec_images_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_images_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_images_7 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_images_8 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_images_8 = PConvBNActiv(512, 512, activ='leaky')
        self.dc_images_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_4 = PConvBNActiv(512 + 512, 256, activ='leaky')
        self.dc_images_3 = PConvBNActiv(256 + 256, 128, activ='leaky')
        self.dc_images_2 = PConvBNActiv(128 + 128, 64, activ='leaky')
        self.dc_images_1 = PConvBNActiv(64 + 64, out_channels, bn=False, sample='none-3', activ=None, bias=True)
        self.tanh = nn.Tanh()

        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

    def forward(self, input_images, input_masks,z):
        ec_images = {}
        ec_images['ec_images_0'], ec_images['ec_images_masks_0'] = input_images, input_masks
        #print(ec_images['ec_images_0'].shape,ec_images['ec_images_masks_0'].shape)
        ec_images['ec_images_1'], ec_images['ec_images_masks_1'] = self.ec_images_1(input_images, input_masks)
        #print(ec_images['ec_images_1'].shape, ec_images['ec_images_masks_1'].shape)
        ec_images['ec_images_2'], ec_images['ec_images_masks_2'] = self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_masks_1'])
        #print(ec_images['ec_images_2'].shape, ec_images['ec_images_masks_2'].shape)
        ec_images['ec_images_3'], ec_images['ec_images_masks_3'] = self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_masks_2'])
        #print(ec_images['ec_images_3'].shape, ec_images['ec_images_masks_3'].shape)
        ec_images['ec_images_4'], ec_images['ec_images_masks_4'] = self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_masks_3'])

        #print(ec_images['ec_images_4'].shape, ec_images['ec_images_masks_4'].shape)
        x = torch.cat([ec_images['ec_images_4'], z], 1)
        x = self.conv_mix_bn(self.conv_mix(F.leaky_relu(x,0.2)))
        #print(x.shape)

        ec_images['ec_images_5'], ec_images['ec_images_masks_5'] = self.ec_images_5(x, ec_images['ec_images_masks_4'])
        #print(ec_images['ec_images_5'].shape, ec_images['ec_images_masks_5'].shape)
        ec_images['ec_images_6'], ec_images['ec_images_masks_6'] = self.ec_images_6(ec_images['ec_images_5'], ec_images['ec_images_masks_5'])
        #print(ec_images['ec_images_6'].shape, ec_images['ec_images_masks_6'].shape)
        ec_images['ec_images_7'], ec_images['ec_images_masks_7'] = self.ec_images_7(ec_images['ec_images_6'], ec_images['ec_images_masks_6'])
        #print(ec_images['ec_images_7'].shape, ec_images['ec_images_masks_7'].shape)
        ec_images['ec_images_8'], ec_images['ec_images_masks_8'] = self.ec_images_8(ec_images['ec_images_7'], ec_images['ec_images_masks_7'])
        #print(ec_images['ec_images_8'].shape, ec_images['ec_images_masks_8'].shape)

        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_masks = ec_images['ec_images_8'], ec_images['ec_images_masks_8']
        dc_conv = 'dc_images_8'
        dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
        dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
        dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)#B,512,1->B,512,2
        for _ in range(7, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_)
            ec_images_masks = 'ec_images_masks_{:d}'.format(_)
            dc_conv = 'dc_images_{:d}'.format(_)
            if _== 4:
                #print(dc_images.shape,x.shape)
                dc_images = torch.cat((dc_images,x),dim=1)
            else:
                #print(dc_images.shape, ec_images[ec_images_skip].shape)
                dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
                #print(dc_images.shape,dc_images_masks.shape)
            dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)

            # print(dc_images.shape,dc_images_masks.shape)
            # print(_)
        outputs = self.tanh(dc_images)

        return ec_images['ec_images_4'],ec_images['ec_images_8'],outputs


class PUNet2(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, up_sampling_node='nearest', init_weights=True):
        super(PUNet2, self).__init__()
        self.freeze_ec_bn = False
        self.up_sampling_node = up_sampling_node
        self.ec_images_1 = PConvBNActiv(in_channels, 64, bn=False, sample='down-7')#64,128
        self.ec_images_2 = PConvBNActiv(64, 128, sample='down-5')#128,64
        self.ec_images_3 = PConvBNActiv(128, 256, sample='down-5')#256,32
        self.ec_images_4 = PConvBNActiv(256, 512, sample='down-3')#512,16
        self.conv_mix = nn.Conv2d(512 + 512, 512, 3, 1, 1)
        self.conv_mix_bn = nn.BatchNorm2d(512)
        self.ec_images_5 = PConvBNActiv(512, 512, sample='down-3')#512,8
        self.ec_images_6 = PConvBNActiv(512, 512, sample='down-3')#512,4
        self.ec_images_7 = PConvBNActiv(512, 512, sample='down-3')#512,2
        self.ec_images_8 = PConvBNActiv(512, 512, sample='down-3')
        self.dc_images_8 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_images_7 = PConvBNActiv(512 + 512, 512, activ='leaky')#512+512,4->512,4
        self.dc_images_6 = PConvBNActiv(512 + 512, 512, activ='leaky')#512+512,8->512,8
        self.dc_images_5 = PConvBNActiv(512 + 512, 512, activ='leaky')#512+512,16->512,16
        self.dc_images_4 = PConvBNActiv(512 + 256, 256, activ='leaky')#
        self.dc_images_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_images_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_images_1 = PConvBNActiv(64 + in_channels, out_channels, bn=False, sample='none-3', activ=None, bias=True)
        self.tanh = nn.Tanh()


    def forward(self, input_images, input_masks,z):
        ec_images = {}
        ec_images['ec_images_0'], ec_images['ec_images_masks_0'] = input_images, input_masks
        ec_images['ec_images_1'], ec_images['ec_images_masks_1'] = self.ec_images_1(input_images, input_masks)
        ec_images['ec_images_2'], ec_images['ec_images_masks_2'] = self.ec_images_2(ec_images['ec_images_1'], ec_images['ec_images_masks_1'])
        ec_images['ec_images_3'], ec_images['ec_images_masks_3'] = self.ec_images_3(ec_images['ec_images_2'], ec_images['ec_images_masks_2'])
        ec_images['ec_images_4'], ec_images['ec_images_masks_4'] = self.ec_images_4(ec_images['ec_images_3'], ec_images['ec_images_masks_3'])
        x = torch.cat([ec_images['ec_images_4'], z], 1)
        x = self.conv_mix_bn(self.conv_mix(F.leaky_relu(x, 0.2)))
        ec_images['ec_images_5'], ec_images['ec_images_masks_5'] = self.ec_images_5(x, ec_images['ec_images_masks_4'])

        ec_images['ec_images_6'], ec_images['ec_images_masks_6'] = self.ec_images_6(ec_images['ec_images_5'], ec_images['ec_images_masks_5'])
        ec_images['ec_images_7'], ec_images['ec_images_masks_7'] = self.ec_images_7(ec_images['ec_images_6'], ec_images['ec_images_masks_6'])
        ec_images['ec_images_8'], ec_images['ec_images_masks_8'] = self.ec_images_8(ec_images['ec_images_7'], ec_images['ec_images_masks_7'])
        # --------------
        # images decoder
        # --------------
        dc_images, dc_images_masks = ec_images['ec_images_8'], ec_images['ec_images_masks_8']
        for _ in range(8, 0, -1):
            ec_images_skip = 'ec_images_{:d}'.format(_ - 1)
            ec_images_masks = 'ec_images_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_images_{:d}'.format(_)
            dc_images = F.interpolate(dc_images, scale_factor=2, mode=self.up_sampling_node)
            dc_images_masks = F.interpolate(dc_images_masks, scale_factor=2, mode=self.up_sampling_node)
            if _== 5:
                #print(dc_images.shape,x.shape)
                dc_images = torch.cat((dc_images,x),dim=1)
            else:
                dc_images = torch.cat((dc_images, ec_images[ec_images_skip]), dim=1)
            dc_images_masks = torch.cat((dc_images_masks, ec_images[ec_images_masks]), dim=1)
            dc_images, dc_images_masks = getattr(self, dc_conv)(dc_images, dc_images_masks)
        outputs = self.tanh(dc_images)
        return ec_images['ec_images_4'], ec_images['ec_images_8'], outputs



class Mutil_view_PGenerator(nn.Module):
    def __init__(self):
        super(Mutil_view_PGenerator, self).__init__()
        self.up_Model = PUNet2()
        self.left_Model = PUNet2()
        self.right_Model = PUNet2()

        self.triple_cross_attn = TripleCrossAttention(in_dim=512)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,up,left,right,up_mask,left_mask,right_mask):
        _z = torch.rand(up.shape[0],512,16,16).cuda()
        up_e4,_,_ = self.up_Model(up,up_mask,_z)
        left_e4,_,_ = self.left_Model(left,left_mask,_z)
        right_e4,_,_ = self.right_Model(right,right_mask,_z)
        z = self.triple_cross_attn(up_e4,left_e4,right_e4)

        _, up_map, out_up = self.up_Model(up,up_mask, z)
        _, left_map, out_left = self.left_Model(left,left_mask, z)
        _, right_map, out_right = self.right_Model(right,right_mask, z)

        # _,_,out_up = self.up_Model(up,z)
        # _,_,out_left = self.left_Model(left,z)
        # _,_,out_right = self.right_Model(right,z)
        #这是预训练时的输出
        #return up_map,left_map,right_map
        return out_up,out_left,out_right




class Multi_Globle_Discriminator(nn.Module):
    def __init__(self):
        super(Multi_Globle_Discriminator, self).__init__()
        self.up_discriminator = WDiscriminator(in_channels=3)
        self.left_discriminator = WDiscriminator(in_channels=3)
        self.right_discriminator = WDiscriminator(in_channels=3)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,up,left,right):
        predict_up = self.up_discriminator(up)
        predict_left = self.left_discriminator(left)
        predict_right = self.right_discriminator(right)
        return predict_up,predict_left,predict_right


class Multi_Local_Discriminator(nn.Module):
    def __init__(self):
        super(Multi_Local_Discriminator, self).__init__()
        self.up_discriminator = LocalWDiscriminator()
        self.left_discriminator = LocalWDiscriminator()
        self.right_discriminator = LocalWDiscriminator()
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    def forward(self, up, left, right):
        predict_up = self.up_discriminator(up)
        predict_left = self.left_discriminator(left)
        predict_right = self.right_discriminator(right)
        return predict_up, predict_left, predict_right




