import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import backbones


class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb

    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MDiscriminator(nn.Module):
    def __init__(self, num_domain=1, input_dim=256, hidden_dim=256):
        super(MDiscriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domain)  # قبلاً +1 اضافه بود که اشتباه بود
        )

    def forward(self, x):
        return self.layers(x)



class MAdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''

    def __init__(self, gamma=1.0, max_iter=1000, num_domain=1, use_lambda_scheduler=True):
        super(MAdversarialLoss, self).__init__()
        self.domain_classifier = MDiscriminator(num_domain=num_domain)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)

    def forward(self, feature, label):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        domain_loss = self.get_adversarial_result(feature, label, lamb)
        return domain_loss

    def get_adversarial_result(self, x, label, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        domain_label = torch.zeros_like(domain_pred, dtype=torch.float, device=device)
        domain_label[:, label] = 1
        loss_fn = nn.BCEWithLogitsLoss()
        # domain_label.float().to(device)
        loss_adv = loss_fn(domain_pred, domain_label)
        return loss_adv


class MSVD(nn.Module):
    """
    """
    def __init__(self, args, max_iter=1000):
        super(MSVD, self).__init__()
        self.num_class = args.num_class
        self.num_domain = args.number_domain
        self.base_network, self.encoder_config = backbones.get_backbone(args)
        bottleneck_list = [
            nn.Linear(self.encoder_config.hidden_size, args.bottleneck_width),
            nn.ReLU()
        ]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        feature_dim = args.bottleneck_width
        self.classifier_layer = nn.Linear(feature_dim, self.num_class)
        self.adapt_loss = MAdversarialLoss(num_domain=self.num_domain)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, sources, target, source_labels):
        # extract features
        source_features = []
        for i in range(self.num_domain):
            source_features.append(self.bottleneck_layer(self.base_network(sources[i]).pooler_output))
        target_feature = self.bottleneck_layer(self.base_network(target).pooler_output)
        # classification
        clf_losses = []
        for i in range(self.num_domain):
            source_clf = self.classifier_layer(source_features[i])
            clf_losses.append(self.criterion(source_clf, source_labels[i]))
        # transfer
        transfer_losses = []
        for i in range(self.num_domain):
            transfer_losses.append(self.adapt_loss(source_features[i], i))
        transfer_losses.append(self.adapt_loss(target_feature, self.num_domain-1))
        return torch.stack(clf_losses), torch.stack(transfer_losses)

    def get_parameters(self, args, lr):
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.base_network.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': 0.1 * lr},
            {'params': [p for n, p in self.base_network.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': 0.1 * lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * lr},
            {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * lr},
            {'params': self.adapt_loss.domain_classifier.parameters(), 'lr': 1.0 * lr}
        ]
        # Loss-dependent
        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features.pooler_output)
        clf = self.classifier_layer(x)
        return clf
