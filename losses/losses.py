from torch import nn, pow, mean, clamp

class ContrastiveLoss(nn.Module):
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
        
        cosine_distance = nn.functional.cosine_similarity(output1, output2)
        loss_contrastive = mean((1-label) * pow(cosine_distance, 2) +
                                      (label) * pow(clamp(self.margin - cosine_distance, min=0.0), 2))
        # loss_contrastive =  torch.nn.NLLLoss()(cosine_distance)

        return loss_contrastive
    

class PreferenceComparisonLoss(nn.Module):
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
        
        cosine_distanceA = nn.functional.cosine_similarity(output1, ref)
        cosine_distanceB = nn.functional.cosine_similarity(output2, ref)
        loss_comparation = nn.NLLLoss()(nn.Sigmoid()(cosine_distanceA - cosine_distanceB), label)

        return loss_comparation