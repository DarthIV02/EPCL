'''
Reference:
    [1] https://github.com/NVIDIA/MinkowskiEngine
    [2] https://github.com/mit-han-lab/spvnas
'''


import torchsparse
import torchsparse.nn as spnn
import torch
from torch import nn
# from tools.utils.common.seg_utils import PointTensor
from torchsparse import PointTensor
from pcseg.model.segmentor.base_segmentors import BaseSegmentor
from torchsparse import SparseTensor
from torchsparse.nn.utils import fapply
from .utils import initial_voxelize, voxel_to_point
from pcseg.loss import Losses
from .transformer import EPCLEncoder
import torchhd
import torch.nn as nn
from tqdm import tqdm
from torchhd.types import VSAOptions
import torch.nn.functional as F
import torchhd.functional as functional

__all__ = ['EPCLOutdoorSeg']


class SyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)
        
class BatchNorm(nn.BatchNorm1d):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return fapply(input, super().forward)

class BasicConvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                stride=stride,
                transposed=True,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                dilation=dilation,
                stride=1,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
    
class BatchProjection(nn.Module):
    __constants__ = ["num_projections", "in_features", "out_features", "vsa"]
    in_features: int
    out_features: int
    vsa: VSAOptions
    weight: torch.Tensor

    def __init__(
        self,
        num_projections,
        in_features,
        out_features,
        vsa: VSAOptions = "MAP",
        requires_grad=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(BatchProjection, self).__init__()
        self.num_projections = num_projections
        self.in_features = in_features
        self.out_features = out_features
        self.vsa = vsa

        if vsa not in {"MAP", "HRR", "VTB"}:
            raise ValueError(
                f"Projection embedding supports MAP, HRR, VTB but provided: {vsa}"
            )

        self.weight = nn.parameter.Parameter(
            torch.empty((num_projections, in_features, out_features), **factory_kwargs),
            requires_grad=requires_grad,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 1)
        self.weight.data.copy_(F.normalize(self.weight.data))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        vsa_tensor = functional.get_vsa_tensor_class(self.vsa)
        #print(input.get_device())
        #print(self.weight.get_device())
        return torch.bmm(input, self.weight).as_subclass(vsa_tensor)


class HD_model():
    def __init__(self, classes = 20, d = 2500, num_features=(409, 204, 153), lr = 0.01, **kwargs):
        self.d = d
        self.div = kwargs['div']
        self.device = kwargs['device']
        self.classes_hv = torch.zeros((classes, self.d))
        self.flatten = nn.Flatten(0,1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax_2 = torch.nn.Softmax(dim=2)
        self.random_projection_0 = torchhd.embeddings.Projection(num_features[0], self.d, device=kwargs['device'])
        self.random_projection_1 = torchhd.embeddings.Projection(num_features[1], self.d, device=kwargs['device'])
        self.random_projection_2 = torchhd.embeddings.Projection(num_features[2], self.d, device=kwargs['device'])
        self.stages = torchhd.random(3, d, device=kwargs['device'])
        #self.random_projection = {0:self.random_projection_0, 1:self.random_projection_1, 2:self.random_projection_2,}
        #self.random_projection = (self.random_projection_0, self.random_projection_1, self.random_projection_2)
        #self.random_projection = BatchProjection(3, num_features[0], self.d, device=kwargs['device'])
        #self.random_projection_global = torchhd.embeddings.Projection(num_features, self.d)
        self.lr = lr
        self.num_features = num_features

    def to(self, *args):
        self.classes_hv = self.classes_hv.to(*args)
        self.random_projection_0 = self.random_projection_0.to(*args)
        self.random_projection_1 = self.random_projection_1.to(*args)
        self.random_projection_2 = self.random_projection_2.to(*args)
        self.random_projection = {0:self.random_projection_0, 1:self.random_projection_1, 2:self.random_projection_2}
        #self.random_projection = self.random_projection.to(*args)

    def encode(self, input_x, infer=False):
        #print(input_x.get_device())
        #hv_0 = self.random_projection(input_x) # <-- BATCH
        #print(hv_0.shape) # (3,#,d)
        hv_0 = torchhd.bind(self.random_projection[0](input_x[0]).sign(), self.stages[0])
        hv_1 = torchhd.bind(self.random_projection[1](input_x[1]).sign(), self.stages[1])
        hv_2 = torchhd.bind(self.random_projection[2](input_x[2]).sign(), self.stages[2])
        hv_all = torch.stack((hv_0, hv_1, hv_2))
        if infer:
            hv_all = torch.sum(hv_all, dim=0).sign()

        #x = input("Enter")

        return hv_all
    
    def forward(self, input_h):
        #print(input_h.shape)
        #print(input_h[1,:,:self.num_features[1]].shape)
        input_h = (input_h[0], input_h[1,:,:self.num_features[1]], input_h[2,:,:self.num_features[2]])
        hv = self.encode(input_h)
        #hv = torch.sum(hv, dim=0).sign()
        sim = self.similarity(hv, True)
        #print("sim: ", sim.shape)
        best_ind = torch.argmax(sim, dim=2)
        #print(best_ind.shape)
        best_sim = torch.max(sim, dim=2).values
        #print(best_sim.shape)
        best_sim_2 = torch.argmax(best_sim, dim=0)
        #print(torch.bincount(best_sim_2))
        #print(best_sim_2.shape)
        pred_label = best_ind[best_sim_2, torch.arange(best_ind.shape[1])]
        #print("ALL ", pred_label)
        hv = torch.sum(hv, dim=0).sign() # hv = hv[best_sim_2, torch.arange(pred_label.shape[0])] <- Best just gets 0
        #print("hv", hv.shape)
        #print("sim: ", sim[best_sim_2, torch.arange(sim.shape[1])].shape)
        return hv, sim[best_sim_2, torch.arange(sim.shape[1])], pred_label
        
    def similarity(self, point, group=False):
        sim = torchhd.cosine_similarity(point, self.classes_hv)
        if group:
            sim = self.softmax_2(sim)
        else:
            sim = self.softmax(sim)

        return sim
    
    def train(self, input_points, classification, **kwargs):
        #print(input_points.shape)
        for idx in torch.arange(input_points.shape[1]).chunk(self.div):
            hv_all, sim_all, pred_labels = self.forward(input_points[:, idx, :])
            idx = idx.to(self.device)
            class_batch = classification[idx].type(torch.LongTensor).to(self.device)
            novelty = 1 - sim_all[torch.arange(idx.shape[0]), class_batch]
            updates = hv_all.transpose(0,1)*torch.mul(novelty, self.lr) # Normal HD with novelty
            updates = updates.transpose(0,1)
            
            # Update all of the classes with the actual label
            self.classes_hv.index_add_(0, class_batch, updates)
            
            #Substract when class is different then actual
            
            #if (pred_labels != c):
            #    self.classes_hv[pred_labels] += -1*hv_all*self.lr*(1-sim_all[pred_labels])
            
            
            # ONLINEHD
            mask_dif = class_batch != pred_labels
            novelty = 1 - sim_all[mask_dif, pred_labels[mask_dif]] # only the ones updated
            updates = hv_all[mask_dif].transpose(0,1)*torch.mul(novelty, self.lr)
            updates = torch.mul(updates, -1)
            updates = updates.transpose(0,1)
            updates_2 = torch.zeros((idx.shape[0], self.d), device=self.device) # all zeros original
            updates_2[mask_dif] = updates # update vectors for the ones that changed

            self.classes_hv.index_add_(0, pred_labels, updates_2)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inc: int,
        outc: int,
        ks: int = 3,
        stride: int = 1,
        dilation: int = 1,
        if_dist: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(
                inc, outc,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc,
                kernel_size=ks,
                stride=stride,
                bias=False,
                dilation=dilation,
            ),
            SyncBatchNorm(outc) if if_dist else BatchNorm(outc),
            spnn.Conv3d(
                outc, outc * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
        )
        if inc == outc * self.expansion and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(
                    inc, outc * self.expansion,
                    kernel_size=1,
                    dilation=1,
                    stride=stride,
                ),
                SyncBatchNorm(outc * self.expansion) if if_dist else BatchNorm(outc * self.expansion),
            )
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class EPCLOutdoorSegHD(BaseSegmentor):    
    def __init__(
        self,
        model_cfgs,
        num_class: int,
        lr: float
    ):
        super().__init__(model_cfgs, num_class)
        self.in_feature_dim = model_cfgs.IN_FEATURE_DIM
        self.num_layer = model_cfgs.get('NUM_LAYER', [2, 3, 4, 6, 2, 2, 2, 2])
        self.block = {
            'ResBlock': ResidualBlock,
            'Bottleneck': Bottleneck,
        }[model_cfgs.get('BLOCK', 'Bottleneck')]

        cr = model_cfgs.get('cr', 1.0)
        cs = model_cfgs.get('PLANES', [32, 32, 64, 128, 256, 256, 128, 96, 96])
        cs = [int(cr * x) for x in cs]

        self.pres = model_cfgs.get('pres', 0.05)
        self.vres = model_cfgs.get('vres', 0.05)

        self.stem = nn.Sequential(
            spnn.Conv3d(
                self.in_feature_dim, cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
            spnn.Conv3d(
                cs[0], cs[0],
                kernel_size=3,
                stride=1,
            ),
            SyncBatchNorm(cs[0]) if model_cfgs.IF_DIST else BatchNorm(cs[0]),
            spnn.ReLU(True),
        )

        self.in_channels = cs[0]
        if_dist = model_cfgs.IF_DIST

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[1], self.num_layer[0], if_dist=if_dist),
        )
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[2], self.num_layer[1], if_dist=if_dist),
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[3], self.num_layer[2], if_dist=if_dist),
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(
                self.in_channels, self.in_channels,
                ks=2,
                stride=2,
                dilation=1,
                if_dist=model_cfgs.IF_DIST,
            ),
            *self._make_layer(
                self.block, cs[4], self.num_layer[3], if_dist=if_dist),
        )

        self.up1 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[5],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[5], self.num_layer[4], if_dist=if_dist))
        )
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[6],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[6], self.num_layer[5], if_dist=if_dist))
        )
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[7],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[7], self.num_layer[6], if_dist=if_dist))
        )
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [
            BasicDeconvolutionBlock(
                self.in_channels, cs[8],
                ks=2,
                stride=2,
                if_dist=model_cfgs.IF_DIST,
            )
        ]
        self.in_channels = cs[8] + cs[0]
        self.up4.append(
            nn.Sequential(*self._make_layer(
                self.block, cs[8], self.num_layer[7], if_dist=if_dist))
        )
        self.up4 = nn.ModuleList(self.up4)

        self.classifier = nn.Sequential(
            nn.Linear((cs[4] + cs[6] + cs[8]) * self.block.expansion, self.num_class)
        )

        self.weight_initialization()

        dropout_p = model_cfgs.get('DROPOUT_P', 0.3)
        self.dropout = nn.Dropout(dropout_p, True)

        label_smoothing = model_cfgs.get('LABEL_SMOOTHING', 0.0)

        # loss
        default_loss_config = {
            'LOSS_TYPES': ['CELoss', 'LovLoss'],
            'LOSS_WEIGHTS': [1.0, 1.0],
            'KNN': 10,
        }
        loss_config = self.model_cfgs.get('LOSS_CONFIG', default_loss_config)

        loss_types = loss_config.get('LOSS_TYPES', default_loss_config['LOSS_TYPES'])
        loss_weights = loss_config.get('LOSS_WEIGHTS', default_loss_config['LOSS_WEIGHTS'])
        assert len(loss_types) == len(loss_weights)
        k_nearest_neighbors = loss_config.get('KNN', default_loss_config['KNN'])

        self.criterion_losses = Losses(
            loss_types=loss_types,
            loss_weights=loss_weights,
            ignore_index=model_cfgs.IGNORE_LABEL,
            knn=k_nearest_neighbors,
            label_smoothing=label_smoothing,
        )
        
        self.epcl_encoder = EPCLEncoder(model_cfgs.EPCL)

        #HD Initialization
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hd_model = HD_model(device=self.device, div=2, lr=lr, classes=num_class)
        self.hd_model.to(self.device)

        print("--------------Loading experiment 5--------------")

    def _make_layer(self, block, out_channels, num_block, stride=1, if_dist=False):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride, if_dist=if_dist)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels, out_channels, if_dist=if_dist)
            )
        return layers
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_dict, return_logit=False, return_tta=False, train_hd=False, **kwargs):
        print("Objectives: ", batch_dict['targets'].feats)
        
        x = batch_dict['lidar']
        x.F = x.F[:, :self.in_feature_dim]
        z = PointTensor(x.F, x.C.float()) # dim=4

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)


        x1 = self.stage1(x0) 
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3) 
        z1 = voxel_to_point(x4, z0)
        encode_z1 = self.hd_model.random_projection_0(z1.F).sign()
        sim = self.hd_model.similarity(encode_z1)
        #print(sim.shape)
        #sim = torch.max(sim, dim=1).values
        sim_arg = torch.argmax(sim, dim=1)
        #print("max z_1: ", torch.max(sim))
        #print("mean z_1: ", torch.mean(sim))
        print("z1: ", sim_arg)
        print("z1: ", sim[torch.arange(sim_arg.shape[0]),sim_arg])

        # epcl encoder
        #xyz, feats = x4.C, x4.F # <----------------
        #print("Input CLIP")
        #print(xyz)
        #print(xyz.shape)
        #print(feats)
        #print(feats.shape)
        #enter_here = input()
        #x4.F = self.epcl_encoder(xyz, feats)
        #print("Output CLIP")
        #print(x4.F)
        #print(x4.F.shape)
        #output_clip = x4.F # <----------------
        
        x4.F = self.dropout(x4.F)# <----------------
        y1 = self.up1[0](x4)# <----------------
        #print("y1")
        #print(y1.F.shape)
        y1 = torchsparse.cat([y1, x3])# <----------------
        y1 = self.up1[1](y1)# <----------------
        #print("y1")
        #print(y1.F.shape)

        y2 = self.up2[0](y1)# <----------------
        #print("y2")
        #print(y2.F.shape)
        y2 = torchsparse.cat([y2, x2])# <----------------
        y2 = self.up2[1](y2)# <----------------

        encode_y2 = self.hd_model.random_projection_1(y2.F).sign()
        sim = self.hd_model.similarity(encode_y2)
        #print(sim.shape)
        sim_arg = torch.argmax(sim, dim=1)
        #sim = torch.max(sim, dim=1).values
        #mask_sim = sim < 0.070
        #print("Skipped: ", torch.sum(~mask_sim))
        #print("max y_2: ", torch.max(sim))
        #print("mean y_2: ", torch.mean(sim))
        print("z2: ", sim_arg)
        print("z2: ", sim[torch.arange(sim_arg.shape[0]),sim_arg])

        #temp = torch.zeros((y2.F.shape[0]), device=self.device)

        #print(y2.F.shape)
        #y2.F = y2.F[mask_sim, :]
        #print(y2.F.shape)
        #temp[~mask_sim] = sim_arg[~mask_sim].float()
        #print(torch.sum(temp != 0))

        #print("y2")
        #print(y2.F.shape)
        z2 = voxel_to_point(y2, z1) # <----------------
        #print("z2")
        #print(z2.F.shape)
        #print(z2.C.shape)
        
        #encode_z12 = torch.sum(torch.stack((encode_z1, encode_z2)), dim=0)
        #sim = self.hd_model.similarity(encode_z12)
        #print(sim.shape)
        #sim = torch.max(sim, dim=1).values
        #print("max z_12: ", torch.max(sim))
        #print("mean z_12: ", torch.mean(sim))
        

        y2.F = self.dropout(y2.F)# <----------------
        y3 = self.up3[0](y2) # <----------------
        #print("y3")
        #print(y3.F.shape)
        y3 = torchsparse.cat([y3, x1]) # <----------------
        y3 = self.up3[1](y3) # <----------------
        #print("y3")
        #print(y3.F.shape)
 
        y4 = self.up4[0](y3)# <----------------
        #print("y4")
        #print(y4.F.shape)
        y4 = torchsparse.cat([y4, x0]) # <----------------
        y4 = self.up4[1](y4) # <----------------
        #print("y4")
        #print(y4.F.shape)
        z3 = voxel_to_point(y4, z2)# <----------------

        # -------------------TEST Z3 Encoding alone ---------------------------
        encode_z3 = self.hd_model.random_projection_2(z3.F).sign()
        sim = self.hd_model.similarity(encode_z3)
        #print(sim.shape)
        #sim = torch.max(sim, dim=1).values
        #print("max z_3: ", torch.max(sim))
        #print("mean z_3: ", torch.mean(sim))
        sim_arg = torch.argmax(sim, dim=1)
        print("z3: ", sim_arg)
        print("z3: ", sim[torch.arange(sim_arg.shape[0]),sim_arg])


        #print("z3")
        #print(z3.F.shape)
        #print(z3.C.shape)
        # ------------------------------------BATCH MUL ----------------------------------
        #tuple_feat = torch.zeros((3,z1.F.shape[0], z1.F.shape[1]))
        #if self.device != "cpu":
        #    tuple_feat = tuple_feat.to(self.device)
        #tuple_feat[0] = z1.F
        #tuple_feat[1, :, :z2.F.shape[1]] = z2.F
        #tuple_feat[1, :, :z3.F.shape[1]] = z3.F
        # ------------------------------------BATCH MUL with pad ----------------------------------
        samples = z2.F.shape[0]
        dim_max = z1.F.shape[1]
        padder = torch.zeros(samples,dim_max-z2.F.shape[1], device=self.device)
        #print(padder.shape)
        z2.F = torch.cat([z2.F,padder], dim = 1)
        #print(z2.F.shape)
        padder = torch.zeros(samples,dim_max-z3.F.shape[1], device=self.device)
        z3.F = torch.cat([z3.F,padder], dim = 1)
        tuple_feat = torch.stack((z1.F, z2.F, z3.F))

        #tuple_feat = (z1.F, z2.F, z3.F) #<----- BEFORE

        #out = self.classifier(concat_feat)
        #print("\nOut")
        #print(out.shape)
        
        if train_hd:
            #target = batch_dict['targets'].F.long().cuda(non_blocking=True)

            #coords_xyz = batch_dict['lidar'].C[:, :3].float()
            #offset = batch_dict['offset']
            self.hd_model.train(tuple_feat, batch_dict['targets'].feats)

            #x = input("Enter")

            return {}
        
        else:
            invs = batch_dict['inverse_map']
            all_labels = batch_dict['targets_mapped']
            point_predict = []
            point_labels = []
            hv, sim, pred_label = self.hd_model.forward(tuple_feat)
            #temp[mask_sim] = pred_label
            for idx in range(invs.C[:, -1].max() + 1):
                cur_scene_pts = (x.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = pred_label[cur_scene_pts][cur_inv]
                targets_mapped = all_labels.F[cur_label]
                point_predict.append(outputs_mapped[:batch_dict['num_points'][idx]].cpu().numpy())
                point_labels.append(targets_mapped[:batch_dict['num_points'][idx]].cpu().numpy())

            return {'point_predict': point_predict, 'point_labels': point_labels, 'name': batch_dict['name'], 'lidar':batch_dict, 
            'output': tuple_feat}

    def forward_ensemble(self, batch_dict):
        return self.forward(batch_dict, ensemble=True)