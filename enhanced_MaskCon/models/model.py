import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet

dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
}


class ModelBase(nn.Module):
    """
    For small size figures:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, figsize=32, num_classes=10, projection_dim=128, arch=None):
        super(ModelBase, self).__init__()
        resnet_arch = getattr(resnet, arch)

        self.net = resnet_arch(pretrained=True)
        if figsize <= 64:  # adapt to small-size images
            self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.net.maxpool = nn.Identity()
        self.net.fc = nn.Identity()

        self.feat_dim = dim_dict[arch]
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, projection_dim)
        )

        self.classifer = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, feat=False):
        x = self.net(x)
        if feat:
            return x
        else:
            cls, proj = self.classifer(x), self.projector(x)
            return cls, proj


"""### MaskCon backbone"""


class MaskCon(nn.Module):
    def __init__(self, num_classes_coarse=10,  dim=128, K=4096, m=0.9, T1=0.1, T2=0.1, arch='resnet18', mode='mixcon', size=32,attribute_vec_size=312):
        '''
        Modifed based on MoCo framework.

        :param num_classes_coarse: num of coarse classes
        :param dim: dimension of feature projections
        :param K: size of memory bank
        :param m: momentum encoder
        :param T1: temperature of original contrastive loss
        :param T2: temperature for soft labels generation
        :param arch: architecture of encoder
        :param mode: method mode [maskcon, grafit or coins]
        :param size: dataset image size
        '''
        super(MaskCon, self).__init__()
        self.K = K
        self.m = m
        self.T1 = T1
        self.T2 = T2
        self.mode = mode
        # create the encoders
        self.encoder_q = ModelBase(figsize=size, num_classes=num_classes_coarse, projection_dim=dim, arch=arch)
        self.encoder_k = ModelBase(figsize=size, num_classes=num_classes_coarse, projection_dim=dim, arch=arch)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.num_classes_coarse = num_classes_coarse
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("attribute_buffer", torch.randint(0, 2, size=(attribute_vec_size, K), dtype=torch.float32 ))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('coarse_labels', torch.randint(0, num_classes_coarse, [self.K]).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, coarse_labels,binary_vectors):

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        #print(binary_vectors.shape)
        self.attribute_buffer[:, ptr:ptr + batch_size] = binary_vectors.t()  # transpose
        self.coarse_labels[ptr:ptr + batch_size] = coarse_labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]


    def initiate_memorybank(self, dataloader):
        #print('Initiate memory bank!')
        num = 0
        iter_data = iter(dataloader)
        for i in range(self.K):  # update the memory bank with image representation
            if num == self.K:
                break
            # print(num)
            try:
                [im_k, _], coarse_label, _ , binary_vectors = next(iter_data)
            except:
                iter_data = iter(dataloader)
                [im_k, _], coarse_label, _ , binary_vectors = next(iter_data)
            num = num + len(im_k)
            im_k, coarse_label,binary_vectors = im_k.cuda(non_blocking=True), coarse_label.cuda(non_blocking=True),binary_vectors.cuda(non_blocking=True)
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)
            self._dequeue_and_enqueue(k, coarse_label,binary_vectors)

    def forward(self, im_k, im_q, coarse_label, binary_vectors, args, epoch):
        #print(binary_vectors.shape)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        cls_q, q = self.encoder_q(im_q)  # queries:
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # soft-labels
            coarse_z = torch.ones(len(q), self.K).cuda()
            new_label = coarse_label.reshape(-1, 1).repeat(1, self.K)
            memory_labels = self.coarse_labels.reshape(1, -1).repeat(len(q), 1)
            
            coarse_z = coarse_z * (new_label == memory_labels)
            logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
            #print(logits_pd.shape)
            logits_pd /= self.T2
            logits_pd = logits_pd * coarse_z  # mask out non-same-coarse class samples
            logits_pd = logits_pd - logits_pd.max(dim=1, keepdim=True)[0]
            
            
            pseudo_soft_z = logits_pd.exp() * coarse_z
            pseudo_sum = torch.sum(pseudo_soft_z, dim=1, keepdim=True)
            
            ############# cosine distance #################

            # logits_pd_new = torch.einsum('nc,ck->nk', [binary_vectors, self.attribute_buffer.clone().detach()])
            # # logits_pd_new /= self.T2
            # logits_pd_new = logits_pd * coarse_z  # mask out non-same-coarse class samples
            # logits_pd_new = logits_pd - logits_pd.max(dim=1, keepdim=True)[0]
            
            ############# hamming_distances ####################

            # # Adjust attribute_buffer shape and compare
            # attribute_buffer_t = self.attribute_buffer.clone().detach().transpose(0, 1)  # Shape becomes (1024, 312)

            # # Expand binary_vectors for Hamming distance calculation
            # binary_vectors_expanded = binary_vectors.unsqueeze(1)  # Shape becomes (256, 1, 312)
            # attribute_buffer_expanded = attribute_buffer_t.unsqueeze(0)  # Shape becomes (1, 1024, 312)

            # # Ensure both tensors are integers for XOR operation
            # binary_vectors_int = binary_vectors_expanded.to(torch.int)
            # attribute_buffer_int = attribute_buffer_expanded.to(torch.int)

            # # Compute Hamming distance using XOR and sum along the feature dimension
            # hamming_distances = torch.sum(binary_vectors_int ^ attribute_buffer_int, dim=2)  # Resulting shape (256, 1024)
            # similarity_scores = -hamming_distances  # Transform distances into similarity scores

            # # Convert distances to similarity scores (optional)
            # logits_new = similarity_scores.float()  # Convert distances to negative for similarity
            # logits_new = logits_new - logits_new.max(dim=1, keepdim=True)[0]  # Normalize
            # logits_pd_new = logits_new

            ########### Euclidean distance #############

            # Adjust attribute_buffer shape
            attribute_buffer_t = self.attribute_buffer.clone().detach().transpose(0, 1)  # Shape becomes (1024, 312)

            # Expand binary_vectors to compare against attribute_buffer
            binary_vectors_expanded = binary_vectors.unsqueeze(1)  # Shape becomes (256, 1, 312)
            attribute_buffer_expanded = attribute_buffer_t.unsqueeze(0)  # Shape becomes (1, 1024, 312)

            # Compute squared Euclidean distance
            dist_squared = torch.sum((binary_vectors_expanded - attribute_buffer_expanded) ** 2, dim=2)

            # Compute Euclidean similarity using the negative exponential of the distance
            sigma = 1  # You can adjust sigma depending on the scale of distances
            euclidean_similarity = torch.exp(-dist_squared / (2 * sigma ** 2))

            logits_new = euclidean_similarity.float()  # Convert distances to negative for similarity
            logits_new = logits_new - logits_new.max(dim=1, keepdim=True)[0]  # Normalize
            logits_pd_new = logits_new

            ##############################################################


            pseudo_soft_z_new = logits_pd_new.exp() * coarse_z
            pseudo_sum_new = torch.sum(pseudo_soft_z_new, dim=1, keepdim=True)
            
            tmp_new = pseudo_soft_z_new / pseudo_sum_new ## same but attribute na vector mate

            ####################################################################
                        
            maskcon_z = torch.zeros(len(q), self.K + 1).cuda()
            maskcon_z[:, 0] = 1
            tmp = pseudo_soft_z / pseudo_sum     ##ahiya sudhi paper nu euqation 11 thy gyu
            
            ######################################################################


            current_epoch = epoch
            total_epochs = args.epochs

            # # Exponential decay for weight1 and growth for weight2
            # weight2 = 0.8 * math.exp(-2 * (current_epoch / total_epochs)**2)
            # weight1 = 1.0 - weight2  # Ensure they sum to 1
            
            
            #print(weight2)
            
            ######################################################################
            
            # # linear weight changing 
            # weight2 = 0.8 - 0.2 * (current_epoch / total_epochs)  # Linear interpolation from 0.6 to 0.4
            # weight1 = 0.2 + 0.2 * (current_epoch / total_epochs)  # Linear interpolation from 0.4 to 0.6
            
            
            ############################################################################
            
            weight1 = 1.0
            weight2 = 0.0

            # Compute the weighted combination
            tmp_fine = torch.add(torch.mul(tmp, weight1), torch.mul(tmp_new, weight2))

            # rescale by maximum
            tmp_fine = tmp_fine / tmp_fine.max(dim=1, keepdim=True)[0]
            
            tmp = tmp / tmp.max(dim=1, keepdim=True)[0] ## ahiya paper nu equation 13 thyu
            
            maskcon_z[:, 1:] = tmp_fine
            
            # generate weighted inter-sample relations
            maskcon_z = maskcon_z / maskcon_z.sum(dim=1, keepdim=True)

            # self-supervised inter-sample relations
            self_z = torch.zeros(len(q), self.K + 1).cuda()
            self_z[:, 0] = 1.0

            labels = args.w * maskcon_z + (1 - args.w) * self_z ##now exactly evy thy che maskcon ma ahiya already 0.3 maskcon and 0.7 ecilidian che
                 #aa same j method thi krelu che now a thay jay pachi pachu again paper na formula mujab maskcon and self supervised learning ne mix kryu che

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # logits: Nx(1+K)
        logits_all = torch.cat([l_pos, l_neg], dim=1)
        logits_all /= self.T1

        loss = -torch.sum(F.log_softmax(logits_all, 1) * labels.detach(), 1).mean()
        # inside vs outside?
        self._dequeue_and_enqueue(k, coarse_label,binary_vectors)

        return loss

    def forward_explicit(self, im_k, im_q, coarse_label,binary_vectors, args):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        cls_q, q = self.encoder_q(im_q)  # queries:
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            _, k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # supcon: coarse inter-sample relations
            coarse_z = torch.zeros(len(q), self.K + 1).cuda()
            coarse_z[:, 0] = 1.0
            tmp_z = torch.ones(len(q), self.K).cuda()
            new_label = coarse_label.reshape(-1, 1).repeat(1, self.K)
            memory_labels = self.coarse_labels.reshape(1, -1).repeat(len(q), 1)
            tmp_z = tmp_z * (new_label == memory_labels)
            coarse_z[:, 1:] = tmp_z

            # self-supervised inter-sample relations
            self_z = torch.zeros(len(q), self.K + 1).cuda()
            self_z[:, 0] = 1.0

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T1
        loss_selfcon = -torch.sum(F.log_softmax(logits, 1) * self_z.detach(), 1)
        loss_cls = torch.nn.functional.cross_entropy(cls_q, coarse_label, reduction='none')
        loss_supcon = -torch.sum(F.log_softmax(logits, 1) * (coarse_z / coarse_z.sum(dim=1, keepdim=True)).detach(), 1)

        logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        logits_pd /= self.T2
        if self.mode == 'grafit':
            loss = args.w * loss_supcon + (1 - args.w) * loss_selfcon
        else:  # self.mode == 'coins'
            loss = args.w * loss_cls + (1 - args.w) * loss_selfcon
        self._dequeue_and_enqueue(k, coarse_label,binary_vectors)

        return loss.mean()
    

###USE ADAM Optimizer instead of SGD
##o/w use Nadam optimizer
##Introduce regularization
##learning rate scheduler
##Make the weights adaptable
##batch learning
##Do mini-batch
##