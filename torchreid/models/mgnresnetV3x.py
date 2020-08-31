################################################ MGNetV3x ##################################################
# for resnet50 pool_channels = 2048  
# AdaptiveMaxPool2d could not been supported by ONNX

class MGBranchV3x(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256, onnx_en=False):
        super(MGBranchV3x, self).__init__()
        self.onnx_en = onnx_en
        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
         #   res50 = resnet50_ibn_a(1, pretrained=True)
            private_layer3 = res50.layer3
            private_layer4 = res50.layer4
            private_conv = nn.Sequential(private_layer3, private_layer4)
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            private_layer3 = seres50.base[3]
            private_layer4 = seres50.base[4]
            private_conv = nn.Sequential(private_layer3, private_layer4)
        else:
            pass
        # reset private convolution layers
        self.private_conv = private_conv
        if self.parts_num > 0:
            if backbone=='resnet50':
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[1][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        if self.parts_num > 2:
            if backbone=='resnet50':
                self.private_conv[0][0].conv2.stride = (1, 1)
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[0][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
            self.private_conv[0][0].downsample[0].stride = (1, 1)
        # global embedding
        if onnx_en:
            factor_dict = {0:1, 2:2, 3:4}
            factor = factor_dict[self.parts_num]
            pool_h = int(12 * factor)
            pool_w = int(4 * factor)
            global_pool = nn.MaxPool2d((pool_h, pool_w), stride=(pool_h, pool_w))
        else:
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.embedding = nn.Sequential(
            global_pool,
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.embedding.apply(weights_init_kaiming)
        if not onnx_en:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            if onnx_en:
                p_pool_h = pool_h // int(self.parts_num)
                self.adaptive_pool = nn.MaxPool2d((p_pool_h, pool_w), stride=(p_pool_h, pool_w))
            else:
                self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                if onnx_en:
                    p_conv = nn.Conv2d(pool_channels, embedding_dim, kernel_size=(parts_num,1), bias=False)
                else:
                    p_conv = nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False)
                local_emb = nn.Sequential(
                    p_conv,
                    nn.BatchNorm2d(embedding_dim),
                    #nn.ReLU(inplace=True)
                )
                local_emb.apply(weights_init_kaiming)
                self.local_emb_name[i] = 'local_emb%d'%(i+1)
                setattr(self, self.local_emb_name[i], local_emb)
                if not onnx_en:
                    classifier = nn.Linear(embedding_dim, num_classes)
                    classifier.apply(weights_init_classifier)
                    self.local_cls_name[i] = 'local_cls%d'%(i+1)
                    setattr(self, self.local_cls_name[i], classifier)

    def forward(self, x):
        x = self.private_conv(x)
        global_feat = self.embedding(x)
        if self.training:
            global_feat = global_feat.view(global_feat.size(0), -1)
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            l_pool = self.adaptive_pool(x) 
            for i in range(self.parts_num):
                if self.onnx_en:
                    part = l_pool
                else:
                    part = l_pool[:,:,i].unsqueeze(2)
                extr = getattr(self, self.local_emb_name[i])                                               
                feat = extr(part)
                if self.training:
                    feat = feat.view(feat.size(0), -1)
                    cls = getattr(self, self.local_cls_name[i])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 
                local_feat_group.append(feat)

        if self.training:
            return tuple([global_score] + local_score_group), tuple([global_feat])
        else:
            return tuple([global_feat] + local_feat_group)

        
class MGResNet50V3x(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50V3x, self).__init__()
        self.loss = loss
        self.onnx_en = False 

        res50 = torchvision.models.resnet50(pretrained=True)
  #      res50 = resnet50_ibn_a(1, pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3x(num_classes, parts_num=num, onnx_en=self.onnx_en))

    def forward(self, x):
        x = self.common(x)
        y = [] 
        for i, num in enumerate(self.branch_stripes):
            branch = getattr(self, self.branch_name[i])
            y.append(branch(x))

        if self.training:
            scores = ()
            feats = ()
            for f in y:
                scores += f[0]
                feats += f[1]
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = ()
            for f in y:
                feats += f
            feature = torch.cat(feats, 1)
            if not self.onnx_en:
                feature = feature.view(feature.size(0), -1)
            return feature


class MGSEResNet50V3x(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3x, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3x(num_classes, backbone='seresnet50', parts_num=num))

    def forward(self, x):
        x = self.common(x)
        y = [] 
        for i, num in enumerate(self.branch_stripes):
            branch = getattr(self, self.branch_name[i])
            y.append(branch(x))

        if self.training:
            scores = ()
            feats = ()
            for f in y:
                scores += f[0]
                feats += f[1]
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = ()
            for f in y:
                feats += f
            feature = torch.cat(feats, 1)
            return feature