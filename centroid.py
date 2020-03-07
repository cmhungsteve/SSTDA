
import torch
import torch.nn as nn


class Centroid(nn.Module):
    def __init__(self, num_f_maps, num_classes):
        super(Centroid, self).__init__()

        self.dim_feat = num_f_maps
        self.num_classes = num_classes
        self.register_buffer('centroid_s', torch.zeros(num_classes, num_f_maps))  # easier to convert devices
        self.register_buffer('centroid_t', torch.zeros(num_classes, num_f_maps))

    def update_centroids(self, feat_s, feat_t, y_s, y_t, method_centroid, ratio_ma):
        # get labels (source: ground truth / target: select highest probability)
        label_source = y_s.detach()
        if method_centroid == 'prob_hard':
            label_target = torch.max(y_t, 1)[1].detach()

        # initialize the centroid for each class
        centroid_source = torch.zeros(self.num_classes, self.dim_feat, device=feat_s.device)
        centroid_target = torch.zeros(self.num_classes, self.dim_feat, device=feat_t.device)

        for i in range(self.num_classes):
            # select features for the current class
            feat_source_select = feat_s[label_source == i]
            feat_target_select = feat_t[label_target == i]

            # get the current class centroids (also deal w/ zero-case)
            centroid_source_current = feat_source_select.mean(0) if feat_source_select.size(0) > 0 else torch.zeros_like(feat_s[0])
            centroid_target_current = feat_target_select.mean(0) if feat_target_select.size(0) > 0 else torch.zeros_like(feat_t[0])

            # moving centroid
            centroid_source[i] = ratio_ma * self.centroid_s[i] + (1 - ratio_ma) * centroid_source_current
            centroid_target[i] = ratio_ma * self.centroid_t[i] + (1 - ratio_ma) * centroid_target_current

        return centroid_source, centroid_target

    def forward(self):  # not really use it
        return self.centroid_s, self.centroid_t

