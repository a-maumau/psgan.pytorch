import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

UNIFOMR_RANGE_MIN = -1.0
UNIFOMR_RANGE_MAX = 1.0

class PSGANGenerator(nn.Module):
    inplace_flag = False
    def __init__(self, conv_channels=[64, 512, 256, 128, 64, 3], kernel_size=4, local_noise_dim=40, global_noise_dim=20, periodic_noise_dim=4, spatial_size=6, hidden_noise_dim=60):
        """
            args:
                input_channel: int
                    input channel size. It should be consider as noise+condition size

                output chanel: int
                    the output channel size. RGB image, it would be 3

                global_noise_dim: int
                    dimension of local part noise Z_l, usually this is the noise of normal Generator of GAN

                global_noise_dim: int
                    dimension of global part noise Z_g

                periodic_noise_dim: int
                    dimension of periodic part noise Z_p

                spatial_size: int
                    size of spatial dimension. it will be (spatial_size x spatial_size) -> (L x M)

                hidden_noise_dim: int
                    dimension of MLP hidden layer of generation periodic part noise
        """
        super(PSGANGenerator, self).__init__()

        self.local_noise_dim = local_noise_dim
        self.global_noise_dim = global_noise_dim
        self.periodic_noise_dim = periodic_noise_dim
        self.spatial_size = spatial_size

        layers = []
        
        for ch_index in range(1, len(conv_channels)-1):
            layers.append(nn.ConvTranspose2d(in_channels=conv_channels[ch_index-1], out_channels=conv_channels[ch_index], kernel_size=kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_channels[ch_index]))
            layers.append(nn.ReLU(inplace=self.inplace_flag))

        layers.append(nn.ConvTranspose2d(in_channels=conv_channels[-2], out_channels=conv_channels[-1], kernel_size=kernel_size, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.generate = nn.Sequential(*layers)

        # MLP that generates K
        self.periodic_noise_mlp_layer1 = nn.Linear(global_noise_dim, hidden_noise_dim)
        self.periodic_noise_mlp_layer2_1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_noise_dim, periodic_noise_dim)
        )
        self.periodic_noise_mlp_layer2_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_noise_dim, periodic_noise_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                c = torch.rand(1).item()*math.pi
                nn.init.normal_(m.bias, mean=c, std=0.02*c)

    def forward(self, Z, tile=None):
        batch_size = Z.shape[0]

        #Z = self._fill_periodic_noise(Z, tile)
        Z_p = self.gen_periodic_noise(Z, tile)
        Z = torch.cat((Z, Z_p), dim=1)

        x = self.generate(Z)

        return x

    def gen_periodic_noise(self, Z, tile=None):
        batch_size = Z.shape[0]
        
        Z_p = torch.zeros(batch_size, self.periodic_noise_dim, self.spatial_size, self.spatial_size).type(Z.type())

        if tile is None:
            # pick one Z_g which is same value in the spatial dimension
            # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
            k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, 0, 0].view(batch_size, -1))

            k1 = self.periodic_noise_mlp_layer2_1(k)
            k2 = self.periodic_noise_mlp_layer2_2(k)

            # naive...
            for l in range(self.spatial_size):
                for m in range(self.spatial_size):
                    Z_p[:, :, l, m] = k1*l + k2*m

            Z_p = torch.sin(Z_p + torch.rand(batch_size, self.periodic_noise_dim, 1, 1).type(Z.type())*2*math.pi)

        else:
            # different from tile == None
            if tile == 1:
                for i in range(self.spatial_size):
                    for j in range(self.spatial_size):
                        # pick one Z_g which is same value in the tile
                        # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
                        k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, i, j].view(batch_size, -1))

                        k1 = self.periodic_noise_mlp_layer2_1(k)
                        k2 = self.periodic_noise_mlp_layer2_2(k)

                        phi = torch.rand(batch_size, self.periodic_noise_dim).type(Z.type())*2*math.pi
                        Z_p[:, :, i, j] = torch.sin(k1 + k2 + phi)
            else:
                # naive...
                block_size = self.spatial_size//tile
                for i in range(tile):
                    for j in range(tile):
                        # pick one Z_g which is same value in the tile
                        # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
                        k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, i*block_size, j*block_size].view(batch_size, -1))

                        k1 = self.periodic_noise_mlp_layer2_1(k)
                        k2 = self.periodic_noise_mlp_layer2_2(k)

                        phi = torch.rand(batch_size, self.periodic_noise_dim).type(Z.type())*2*math.pi
                        # naive...
                        for l in range(block_size):
                            for m in range(block_size):
                                # no confidence at this part...
                                Z_p[:, :, i*block_size+l, j*block_size+m] = torch.sin(k1*l + k2*m + phi)

        return Z_p

    # inplace operation
    def _fill_periodic_noise(self, Z, tile):
        batch_size = Z.shape[0]
        if tile is None:
            # pick one Z_g which is same value in the spatial dimension
            # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
            k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, 0, 0].view(batch_size, -1))

            # batch x hidden_noise_dim -> batch x periodic_noise_dim -> batch x 1 x periodic_noise_dim
            k1 = self.periodic_noise_mlp_layer2_1(k)
            k2 = self.periodic_noise_mlp_layer2_2(k)

            # naive...
            for l in range(self.spatial_size):
                for m in range(self.spatial_size):
                    Z[:, self.local_noise_dim+self.global_noise_dim:, l, m] = k1*l + k2*m

            Z[:, self.local_noise_dim+self.global_noise_dim:] = torch.sin(Z[:, self.local_noise_dim+self.global_noise_dim:] + torch.rand(self.periodic_noise_dim, 1, 1).type(Z.type())*2*math.pi)

        else:
            # naive...
            # different from tile == None
            if tile == 1:
                for i in range(self.spatial_size):
                    for j in range(self.spatial_size):
                        # pick one Z_g which is same value in the tile
                        # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
                        k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, i, j].view(batch_size, -1))

                        k1 = self.periodic_noise_mlp_layer2_1(k)
                        k2 = self.periodic_noise_mlp_layer2_2(k)
                        
                        phi = torch.rand(batch_size, self.periodic_noise_dim).type(Z.type())*2*math.pi
                        Z[:, self.local_noise_dim+self.global_noise_dim:, i, j] = torch.sin(k1*i + k2*j + phi)
            else:
                # naive...
                block_size = self.spatial_size//tile
                for i in range(tile):
                    for j in range(tile):
                        # pick one Z_g which is same value in the tile
                        # batch x Z_p dim x 1 x 1 -> batch x hidden_noise_dim
                        k = self.periodic_noise_mlp_layer1(Z[:, self.local_noise_dim:self.local_noise_dim+self.global_noise_dim, i*block_size, j*block_size].view(batch_size, -1))

                        k1 = self.periodic_noise_mlp_layer2_1(k)
                        k2 = self.periodic_noise_mlp_layer2_2(k)

                        # naive...
                        phi = torch.rand(batch_size, self.periodic_noise_dim).type(Z.type())*2*math.pi
                        for l in range(block_size):
                            for m in range(block_size):
                                # no confidence at this part...
                                Z[:, self.local_noise_dim+self.global_noise_dim:, i*block_size+l, j*block_size+m] = torch.sin(k1*l + k2*m + phi)

        return Z

    def generate_noise(self, batch_size, local_dim, global_dim, periodic_dim, spatial_size, tile=None):
        """
            output of this function doesn't fill the periodic part noise of Z.
            that will be filled in the forwading phase.

            I think this implimentaton is very slow.
            should be fixed in the future.
        """

        #Z = np.zeros((batch_size, local_dim+global_dim+periodic_dim, spatial_size, spatial_size))
        Z = np.zeros((batch_size, local_dim+global_dim, spatial_size, spatial_size))

        # set local noise
        Z[:, :local_dim] = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, local_dim, spatial_size, spatial_size))
        
        # set global noise
        if tile is None:
            Z_g = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))

            # use numpy's broadcast to fill all spatial dimension to be same (repeated)
            # global noise
            Z[:, local_dim:local_dim+global_dim] = Z_g

            # periodic noise will be filled at forwarding
            #Z[:, local_dim+global_dim:, h, w] = Z_g

        else:
            block_size = spatial_size//tile
            for i in range(tile):
                for j in range(tile):
                    Z_g = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))

                    # use numpy's broadcast to fill all spatial dimension to be same in the tile
                    Z[:, local_dim:local_dim+global_dim, i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = Z_g

                    # periodic noise will be filled at forwarding
                    #Z[:, local_dim+global_dim:, h, w] = Z_g

        # return the noise tensor without filling the periodic noise
        return torch.FloatTensor(Z)

    # bilinear interpolation of 4 point of corner
    def generate_noise_interpolation(self, batch_size, local_dim, global_dim, periodic_dim, spatial_size):
        """
            output of this function doesn't fill the periodic part noise of Z.
            that will be filled in the forwading phase.

            I think this implimentaton is very slow.
            should be fixed in the future.
        """
        
        #Z = np.zeros((batch_size, local_dim+global_dim+periodic_dim, spatial_size, spatial_size))
        Z = np.zeros((batch_size, local_dim+global_dim, spatial_size, spatial_size))

        # set local noise
        Z[:, :local_dim] = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, local_dim, spatial_size, spatial_size))

        upleft = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))
        upright = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))
        downleft = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))
        downright = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))

        rate_term = spatial_size-1

        #print("distance, l1 norm")
        #print("mean upleft-upright     :", np.linalg.norm(upleft-upright)/batch_size)
        #print("mean upleft-downleft    :", np.linalg.norm(upleft-downleft)/batch_size)
        #print("mean upleft-downrigh    :", np.linalg.norm(upleft-downright)/batch_size)
        #print("mean upright-downleft   :", np.linalg.norm(upright-downleft)/batch_size)
        #print("mean upright-downright  :", np.linalg.norm(upright-downright)/batch_size)
        #print("mean downleft-downright :", np.linalg.norm(downleft-downright)/batch_size)

        for i in range(spatial_size):
            for j in range(spatial_size):
                Z_g = (1-i/rate_term)*(1-j/rate_term)*upleft + (1-i/rate_term)*(j/rate_term)*upright + (i/rate_term)*(1-j/rate_term)*downleft + (i/rate_term)*(j/rate_term)*downright

                # use numpy's broadcast to fill all spatial dimension to be same in the tile
                Z[:, local_dim:local_dim+global_dim, i:i+1, j:j+1] = Z_g

                # periodic noise will be filled at forwarding
                #Z[:, local_dim+global_dim:, h, w] = Z_g

        # return the noise tensor without filling the periodic noise
        return torch.FloatTensor(Z)

    # linear interpolation of 2 points of left edge and right edge
    def generate_noise_left2right_interpolation(self, batch_size, local_dim, global_dim, periodic_dim, spatial_size):
        """
            output of this function doesn't fill the periodic part noise of Z.
            that will be filled in the forwading phase.
        """

        #Z = np.zeros((batch_size, local_dim+global_dim+periodic_dim, spatial_size, spatial_size))
        Z = np.zeros((batch_size, local_dim+global_dim, spatial_size, spatial_size))

        # set local noise
        Z[:, :local_dim] = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, local_dim, spatial_size, spatial_size))

        left = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))
        right = np.random.uniform(UNIFOMR_RANGE_MIN, UNIFOMR_RANGE_MAX, (batch_size, global_dim, 1, 1))

        rate_term = spatial_size-1

        #print("distance, l1 norm")
        #print("mean left-right ; ", np.linalg.norm(left-right)/batch_size)

        for i in range(spatial_size):
            Z_g = (1-i/rate_term)*left + (i/rate_term)*right

            # use numpy's broadcast to fill all spatial dimension to be same in the tile
            Z[:, local_dim:local_dim+global_dim, :, i:i+1] = Z_g

            # periodic noise will be filled at forwarding
            #Z[:, local_dim+global_dim:, h, w] = Z_g

        # return the noise tensor without filling the periodic noise
        return torch.FloatTensor(Z)

    def save(self, add_state={}, file_name="model_param.pth"):
        #assert type(add_state) is dict, "arg1:add_state must be dict"
        
        if "state_dict" in add_state:
            print("the value of key:'state_dict' will be over write with model's state_dict parameters")

        _state = add_state
        _state["state_dict"] = self.state_dict()
        
        try:
            torch.save(_state, file_name)
        except:
            torch.save(self.state_dict(), "./model_param.pth.tmp")
            print("save_error.\nsaved at ./model_param.pth.tmp only model params.")

    def load_trained_param(self, parameter_path, print_debug=False):
        chkp = torch.load(os.path.abspath(parameter_path), map_location=lambda storage, location: storage)

        if print_debug:
            print(chkp.keys())

        self.load_state_dict(chkp["state_dict"])

class PSGANDiscriminator(nn.Module):
    inplace_flag = False
    def __init__(self, conv_channels=[3, 64, 128, 256, 512, 1], kernel_size=4):
        super(PSGANDiscriminator, self).__init__()
        """
            args:
                conv_channel: list of int
                    the channel of convolution layer will be construct on this.

                kernel_size; int 
                    the size of kernel. 
        """

        layers = []
        for ch_index in range(1, len(conv_channels)-1):
            layers.append(nn.Conv2d(in_channels=conv_channels[ch_index-1], out_channels=conv_channels[ch_index], kernel_size=kernel_size, stride=2, padding=1))
            if ch_index != 1:
                layers.append(nn.BatchNorm2d(conv_channels[ch_index]))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=self.inplace_flag))

        layers.append(nn.Conv2d(in_channels=conv_channels[-2], out_channels=conv_channels[-1], kernel_size=kernel_size, stride=2, padding=1))
        layers.append(nn.Sigmoid())

        self.discriminate = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.discriminate(x)

        spatial_size = x.shape[2]
        return x.view(batch_size, spatial_size*spatial_size)

    def save(self, add_state={}, file_name="model_param.pth"):
        #assert type(add_state) is dict, "arg1:add_state must be dict"
        
        if "state_dict" in add_state:
            print("the value of key:'state_dict' will be over write with model's state_dict parameters")

        _state = add_state
        _state["state_dict"] = self.state_dict()
        
        try:
            torch.save(_state, file_name)
        except:
            torch.save(self.state_dict(), "./model_param.pth.tmp")
            print("save_error.\nsaved at ./model_param.pth.tmp only model params.")
