import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as F
# from captum.attr import FeatureAblation
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# import dropbox

# from SimCLR.models.resnet_simclr import *
'''
This function 
'''

class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights='ResNet18_Weights.DEFAULT', num_classes=out_dim),
                            "resnet50": models.resnet50(weights='ResNet50_Weights.DEFAULT', num_classes=out_dim),
                            "resnet101": models.resnet101(weights='ResNet101_Weights.DEFAULT', num_classes=out_dim),
                            "densenet121": models.densenet121(weights='DenseNet121_Weights.DEFAULT', num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
    
def setup_seed(seed):
    # random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True 
 
def imshow(inp, title=None, mean=np.array([ 0.7013, -0.1607, -0.7902]), std=np.array([0.5904, 0.5008, 0.3771])):
    """Input shound be tensor [3,224,224]"""
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = mean
    # std = std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def imgtensor2array(inp, mean = np.array([ 0.7013, -0.1607, -0.7902]), std = np.array([0.5904, 0.5008, 0.3771])):
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = mean
    # std = std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def imgarray2tensor(inp, mean = np.array([ 0.7013, -0.1607, -0.7902]), std = np.array([0.5904, 0.5008, 0.3771])):
    tf = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return tf(inp)

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight 

def get_sampler_unbalance_set(dataset_train):                                                                         
# For unbalanced dataset we create a weighted sampler                       
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler

def mean_std(loader):
  images, lebels = next(iter(loader))
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

def reshape_transform_vit_huggingface(x):
    activations = x[:, 1:, :]
    activations = activations.view(activations.shape[0],
                                   14, 14, activations.shape[2])
    activations = activations.transpose(2, 3).transpose(1, 2)
    return activations

# Show list of images (in tensor type)
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def get_mask(path, prefix = "./datasets/annotated_eyes/"):
    """
    This function return annotation mask aggregated from multiple medical experts.
    The overlapping between expert's annomations are considered the same important
    as the area not overlapping
    Input:
        path: The path to an image
        prefix:  Prefix until image ID
    Output: 
        aggregate_mask: Annotation mask of an image (have the same height and width with
    original images but color dimension = 1). Values in Mask is true/false, indicate
    which pixel is consider contain glaucoma.
    """
    image_ID = path.split(prefix)[1].split(".jpg")[0]
    # Looking for annotation of images ID
    files = glob.glob(f"./datasets/annotators/OHTS_{image_ID}_*.png")
    sum = 0
    if(files == []):
        return torch.tensor(0)
    for f in files:
        temp = read_image(f)
        sum = sum + temp

    aggregate_mask = (sum[0] != 0)
    return aggregate_mask

class RankNet(torch.nn.Module):
    def __init__(self, input_size = 2048, model = None):
        super(RankNet, self).__init__()
        setup_seed(2023)
        self.model = nn.Sequential(torch.nn.Linear(input_size, 200),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.3),
                              torch.nn.Linear(200, 64),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.3),
                              torch.nn.Linear(64, 1)) if (model == None) else model

    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        # subtract = x1-x2
        # prob = torch.nn.Sigmoid()(subtract)
        return x1, x2

    def get_score(self, x):
        return self.model(x)

class ListNet(torch.nn.Module):
    def __init__(self, input_size = 2048):
        super(ListNet, self).__init__()
        self.f = nn.Sequential(torch.nn.Linear(input_size, 200),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.3),
                                torch.nn.Linear(200, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.3),
                                torch.nn.Linear(64, 1))

    def forward(self, listofx):
        scores = self.f(listofx)
        top1probs = torch.nn.Softmax(dim=0)(scores)
        # top1probs = [(i/torch.sum(torch.FloatTensor(top1probs))) for i in top1probs]
        return top1probs
    
    def score(self, x):
        return self.f(x)
    
    def get_score_function(self):
        return self.f

class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = models.resnet101(pretrained = True)
        self.cnn1.fc = nn.Linear(2048, 3) # mapping input image to a 3 node output

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class RankNet_wresnet(torch.nn.Module):
    def __init__(self, fcnet = None, feature_extractor='resnet50', cotrain=True, simclr=None):
        super(RankNet_wresnet, self).__init__()
        # Init feature extractor
        self.fextractor = get_feature_extractor(feature_extractor, fcnet, cotrain, simclr=simclr)

    def forward(self, x1, x2):
        x1 = self.fextractor(x1)
        x2 = self.fextractor(x2)
        return torch.nn.Sigmoid()(x1-x2)
    
    def get_model(self):
        return self.fextractor

class RankNet_wresnet2(torch.nn.Module):
    def __init__(self, fcnet = None, feature_extractor='resnet50', cotrain=True, ncriteria=10, simclr=None):
        super(RankNet_wresnet2, self).__init__()
        # self.fcnet = nn.Sequential(torch.nn.Linear(2048, 256),
        #                       torch.nn.ReLU(),
        #                       torch.nn.Dropout(0.1),
        #                       torch.nn.Linear(256, 64),
        #                       torch.nn.ReLU(),
        #                       torch.nn.Dropout(0.1),
        #                       torch.nn.Linear(64, ncriteria))
        self.fextractor = get_feature_extractor(feature_extractor, fcnet=fcnet, cotrain = cotrain, model = 'siamese10', ncriteria = ncriteria, simclr=simclr)
        
        self.dense = nn.Sequential(torch.nn.Linear(ncriteria, 4),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.1),
                              torch.nn.Linear(4, 2))

        for param in self.dense.parameters():
            param.requires_grad = True
        
    def forward(self, x1, x2):
        x1 = self.fextractor(x1)
        x2 = self.fextractor(x2)
        return self.dense(torch.nn.Sigmoid()(x1-x2))
    
    def get_model(self):
        return self.fextractor
    
    def get_dense(self):
        return self.dense

class RankNet_wresnet3(torch.nn.Module):
    def __init__(self, fcnet=None, device='cpu', feature_extractor='resnet50', cotrain=True, ncriteria=10):
        super(RankNet_wresnet3, self).__init__()
        self.device = device
        self.ncriteria = ncriteria
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.model.fc = nn.Sequential(torch.nn.Linear(2048, 256),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, ncriteria)
                              )

        self.dense = nn.Sequential(torch.nn.Linear(ncriteria*ncriteria, 256),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, 64),
                              torch.nn.ReLU(),
                              torch.nn.Dropout(0.1),
                              torch.nn.Linear(64, 2))
    def forward(self, x1, x2):
        x1 = self.model(x1).unsqueeze(2)
        x2 = self.model(x2).unsqueeze(2)
        batch_size = x1.shape[0]
        # print(x1.is_cuda)
        # print(batch_size)
        sigm = torch.nn.Sigmoid()(torch.matmul(x1, torch.ones((batch_size, 1, self.ncriteria)).to(self.device)) - torch.matmul(x2, torch.ones((batch_size, 1, self.ncriteria)).to(self.device)))
        return self.dense(torch.flatten(sigm, 1))
    def get_model(self):
        return self.model
    def get_dense(self):
        return self.dense

def get_activation(activation='sigmoid'):
    # ret = Sigmoid()
    if(activation == "Sigmoid"):
        ret = torch.nn.Sigmoid()
    elif(activation == "ReLU"):
        ret = torch.nn.ReLU()
    elif(activation == "GELU"):
        ret = torch.nn.GELU()
    else:
        assert False, "Please choose activation between Sigmoid, ReLU, GELU"
    return ret

# class nComparisonSiamese(Module):
#     def __init__(self, 
#                 fcnet = None, 
#                 feature_extractor='resnet50', 
#                 cotrain=True, 
#                 ncriteria=10, 
#                 simclr=None,
#                 activation='sigmoid'):
#         super(nComparisonSiamese, self).__init__()

#         self.fextractor = get_feature_extractor(feature_extractor, fcnet=fcnet, cotrain = cotrain, model = 'siamese10', ncriteria = ncriteria, simclr=simclr)
#         self.activation = get_activation(activation)
        
#         self.dense = fcnet if fcnet else Sequential(torch.nn.Linear(ncriteria, 4), 
#                                                     torch.nn.ReLU(), 
#                                                     torch.nn.Dropout(0.1), 
#                                                     torch.nn.Linear(4, 2))

#         for param in self.dense.parameters():
#             param.requires_grad = True
        
#     def forward(self, x1, x2):
#         x1 = self.fextractor(x1)
#         x2 = self.fextractor(x2)
#         return self.dense(self.activation(x1-x2))
    
#     def get_model(self):
#         return self.fextractor
    
#     def get_dense(self):
#         return self.dense

def get_feature_extractor(feature_extractor = 'resnet50', fcnet = None, cotrain=True, ncriteria=10, model='siamese1', simclr = None):
    if(feature_extractor == 'resnet50'):    
        fextractor = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        in_features = 2048
        if(simclr):
            print('load simclr resnet')
            ressimclr = ResNetSimCLR('resnet50', 1000)
            state_dict = torch.load(simclr)
            ressimclr.load_state_dict(state_dict['state_dict'])
            fextractor = ressimclr.backbone
        fextractor.fc = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
    elif(feature_extractor == 'resnet101'):    
        fextractor = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        in_features = 2048
        if(simclr):
            print('load simclr resnet')
            ressimclr = ResNetSimCLR('resnet101', 1000)
            state_dict = torch.load(simclr)
            ressimclr.load_state_dict(state_dict['state_dict'])
            fextractor = ressimclr.backbone
        fextractor.fc = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
    elif(feature_extractor == 'densnet121'):
        fextractor = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
        in_features = 1024
        fextractor.classifier = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet
        # fextractor._modules['classifier'] = fextractor._modules.pop('classifier')
    elif(feature_extractor == 'vgg19'):
        fextractor = models.vgg19()
        fextractor.load_state_dict(torch.load('./pretrained/vgg19-dcbb9e9d.pth'))
        in_features = 25088 # https://www.geeksforgeeks.org/vgg-16-cnn-model/ length of vgg19
        fextractor.classifier = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet 
        # fextractor._modules['fc'] = fextractor._modules.pop('classifier')
    elif(feature_extractor == 'vit16'):
        fextractor = models.vit_b_16()
        in_features = 768
        fextractor.load_state_dict(torch.load('./pretrained/vit_b_16-c867db91.pth'))
        fextractor.heads.head = get_default_fc(in_features, model=model, ncriteria=ncriteria) if (fcnet == None) else fcnet 
        # fextractor.classifier = get_default_fc(in_features) if (fcnet == None) else fcnet
    else:
        assert False, 'No feature extractor founded'

    for param in fextractor.parameters():
            param.requires_grad = cotrain
    if(feature_extractor == 'resnet50' or feature_extractor == 'resnet101'):        
        for param in fextractor.fc.parameters():
            param.requires_grad = True
    elif(feature_extractor == 'vit16'):
        for param in fextractor.heads.parameters():
            param.requires_grad = True
    else:
        for param in fextractor.classifier.parameters():
            param.requires_grad = True

    return fextractor

def get_default_fc(in_features=2048, model='siamese1', ncriteria=10):
    if(model=='siamese1'):
        ret =  nn.Sequential(torch.nn.Linear(in_features, 256),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, 1))
    else:
        ret =  nn.Sequential(torch.nn.Linear(in_features, 256),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, 64),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, ncriteria))
    return ret

def get_feature_importance(data, ranknet, device='cpu', type='32x32'):
    model = ranknet.fextractor.to(device)
    x1 = model(data[0].unsqueeze(0).to(device)).unsqueeze(2)
    x2 = model(data[1].unsqueeze(0).to(device)).unsqueeze(2)
    sigm = torch.nn.Sigmoid()(torch.matmul(x1, torch.ones((x1.shape[0],1,32)).to(device)) - torch.matmul(x2, torch.ones((x2.shape[0],1,32)).to(device)))

    if(type == '10'):
        sigm = torch.nn.Sigmoid()(x1-x2)
        
    X_test = torch.flatten(sigm, 1)
    fa = FeatureAblation(ranknet.dense.cpu())
    fa_attr_test = fa.attribute(X_test.cpu(), target=1)
    fa_attr_test_sum = fa_attr_test.detach().numpy().sum(0)
    fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)
    return fa_attr_test_norm_sum

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def find_heatmap(ranknet, imageA, imageB, fA, fB, feature_extractor='resnet50'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    model_pretrained = ranknet.fextractor
    model = model_pretrained
    if(feature_extractor == 'resnet50'):
        target_layers = [model.layer4[-1]]
    elif(feature_extractor == 'vgg19'):
        target_layers = [model.features[-1]]
    elif(feature_extractor == 'vit16'):
        target_layers = [model.encoder.layers.encoder_layer_11.ln_1]
    else:
        assert False, 'No feature_extractor name found'

    # input_tensor = data_transforms['test'](Image.open("./datasets/longtitude/90007-L/90007-LU306-L.jpg")).unsqueeze(0) # Create an input tensor image for your model..
    input_tensorA = imageA.unsqueeze(0)
    input_tensorB = imageB.unsqueeze(0)
    
    # Note: input_tensor can be a batch tensor with several images!
    rgb_imgA = imgtensor2array(input_tensorA[0], mean=mean, std=std)
    rgb_imgB = imgtensor2array(input_tensorB[0], mean=mean, std=std)

    # Construct the CAM object once, and then re-use it on many images:
    if(feature_extractor == 'vit16'):
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    
    grayscale_camA = cam(input_tensor=input_tensorA, targets=[ClassifierOutputTarget(fA)])
    grayscale_camB = cam(input_tensor=input_tensorB, targets=[ClassifierOutputTarget(fB)])
    
    return rgb_imgA, rgb_imgB, grayscale_camA, grayscale_camB 

def calculate_iou(pred_mask, gt_mask, true_pos_only):
    """
    Calculate IoU score between two segmentation masks.

    Args:
        pred_mask (np.array): binary segmentation mask
        gt_mask (np.array): binary segmentation mask
    Returns:
        iou_score (np.float64)
    """
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)

    if true_pos_only:
        if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(union) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))

    return iou_score

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """ 

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        cosine_distance = torch.nn.functional.cosine_similarity(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        # loss_contrastive =  torch.nn.NLLLoss()(cosine_distance)

        return loss_contrastive

class SeverityModel(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SeverityModel, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.bestsimese50simclr = SiameseNetwork101()
        state_dict = torch.load('./pretrained/best-contrastive50.pt')
        self.bestsimese50simclr.load_state_dict(state_dict)
        self.bestsimese50simclr.cnn1.add_module('fc2',
            nn.Sequential(torch.nn.Linear(256, 256),
                          torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(256, 256)))
    
    def forward_once(self, x):
        output = self.bestsimese50simclr.cnn1.fc2(self.bestsimese50simclr.cnn1(x))
        return output

    def forward(self, input1, input2, refinput):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        refinput = self.bestsimese50simclr.cnn1(refinput)
        return output1, output2, refinput

class PreferenceComparisonLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """ 

    def __init__(self, margin=2.0):
        super(PreferenceComparisonLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, ref):
        # euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        # loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
        #                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        cosine_distanceA = torch.nn.functional.cosine_similarity(output1, ref)
        cosine_distanceB = torch.nn.functional.cosine_similarity(output2, ref)
        loss_comparation = torch.nn.NLLLoss()(torch.nn.Sigmoid()(cosine_distanceA - cosine_distanceB), label)

        return loss_comparation

class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = get_feature_extractor(feature_extractor='resnet50', cotrain=False)# , simclr='/mnt/c/Users/PCM/Dropbox/pretrained/SimCLR/checkpoint_10_02102023.pth.tar')
        self.cnn1.fc = nn.Sequential(torch.nn.Linear(2048, 1000),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                                torch.nn.Linear(1000, 256))
    
    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
class SeverityModel(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self, path2pretrained='./pretrained/best-contrastive50.pt'):
        super(SeverityModel, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.bestsimese50simclr = SiameseNetwork101()
        state_dict = torch.load(path2pretrained)
        self.bestsimese50simclr.load_state_dict(state_dict)
        self.bestsimese50simclr.cnn1.add_module('fc2',
            nn.Sequential(torch.nn.Linear(256, 256),
                          torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(256, 256)))
    
    def forward_once(self, x):
        output = self.bestsimese50simclr.cnn1.fc2(self.bestsimese50simclr.cnn1(x))
        return output

    def forward(self, input1, input2, refinput):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        refinput = self.bestsimese50simclr.cnn1(refinput)
        return output1, output2, refinput
