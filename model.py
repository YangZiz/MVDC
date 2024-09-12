import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Mutil_Generator(nn.Module):
    def __init__(self,d = 64):
        super(Mutil_Generator,self).__init__()
        self.conv1 = nn.Conv2d(2,d,4,2,1)#256->128
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

        self.attention = AttentionModule(input_dim=256)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self,input,z):
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e4 = F.leaky_relu(e4,0.2)
        attended_e4 = self.attention(e4)
        #print(e4.shape,z.shape)
        x = torch.cat([e4,z],1)
        e5 = self.conv5_bn(self.conv5(x))

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
        d4 = torch.cat([d4, x], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        out = torch.tanh(d8)
        return attended_e4,out


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

class Pix2pix_Generator(nn.Module):
    def __init__(self,d = 64):
        super(Pix2pix_Generator,self).__init__()
        self.conv1 = nn.Conv2d(1,d,4,2,1)#256->128
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
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
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

class Pix2pix_Discriminator(nn.Module):
    def __init__(self,in_channels,d = 64):
        super(Pix2pix_Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, d, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)  # 128->64
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)  # 64->32
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)  # 32->16
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 1, 1)  # 16->15
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, 1, 4, 1, 1)  # 15->14

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        x = self.conv6(x)
        return x

    def feature_extraction(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
        return x.view(-1, 512 * 15 * 15)



