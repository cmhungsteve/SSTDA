
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
from loss import *
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.ce_d = nn.CrossEntropyLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def adapt_weight(self, iter_now, iter_max_default, iter_max_input, weight_loss, weight_value=10.0, high_value=1.0, low_value=0.0):
        # affect adaptive weight value
        iter_max = iter_max_default
        if weight_loss < -1:
            iter_max = iter_max_input

        high = high_value
        low = low_value
        weight = weight_value
        p = float(iter_now) / iter_max
        adaptive_weight = (2. / (1. + np.exp(-weight * p)) - 1) * (high-low) + low
        return adaptive_weight

    def train(self, model, model_dir, results_dir, batch_gen_source, batch_gen_target, device, args):
        # ====== collect arguments ====== #
        verbose = args.verbose
        num_epochs = args.num_epochs
        batch_size = args.bS
        num_f_maps = args.num_f_maps
        learning_rate = args.lr
        alpha = args.alpha
        tau = args.tau
        use_target = args.use_target
        ratio_source = args.ratio_source
        ratio_label_source = args.ratio_label_source
        resume_epoch = args.resume_epoch
        # tensorboard
        use_tensorboard = args.use_tensorboard
        epoch_embedding = args.epoch_embedding
        stage_embedding = args.stage_embedding
        num_frame_video_embedding = args.num_frame_video_embedding
        # adversarial loss
        DA_adv = args.DA_adv
        DA_adv_video = args.DA_adv_video
        iter_max_beta_user = args.iter_max_beta
        place_adv = args.place_adv
        beta = args.beta
        # multi-class adversarial loss
        multi_adv = args.multi_adv
        weighted_domain_loss = args.weighted_domain_loss
        ps_lb = args.ps_lb
        # semantic loss
        method_centroid = args.method_centroid
        DA_sem = args.DA_sem
        place_sem = args.place_sem
        ratio_ma = args.ratio_ma
        gamma = args.gamma
        iter_max_gamma_user = args.iter_max_gamma
        # entropy loss
        DA_ent = args. DA_ent
        place_ent = args.place_ent
        mu = args.mu
        # discrepancy loss
        DA_dis = args.DA_dis
        place_dis = args.place_dis
        nu = args.nu
        iter_max_nu_user = args.iter_max_nu
        # ensemble loss
        DA_ens = args.DA_ens
        place_ens = args.place_ens
        dim_proj = args.dim_proj
        # self-supervised learning for videos
        SS_video = args.SS_video
        place_ss = args.place_ss
        eta = args.eta

        # multi-GPU
        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        model.train()
        model.to(device)
        if resume_epoch > 0:
            model.load_state_dict(torch.load(model_dir + "/epoch-" + str(resume_epoch) + ".model"))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # determine batch size
        batch_size_source = batch_size
        batch_size_target = max(int(batch_gen_target.num_examples/batch_gen_source.num_examples*batch_size_source), 1)
        num_iter_epoch = np.ceil(batch_gen_source.num_examples / batch_size_source)

        acc_best_source = 0.0  # store the best source acc
        acc_best_target = 0.0  # store the best target acc
        if use_tensorboard:
            writer = SummaryWriter(results_dir + '/tensorboard')  # for tensorboardX

        for epoch in range(resume_epoch, num_epochs):
            epoch_loss = 0
            correct_source = 0
            total_source = 0
            correct_target = 0
            total_target = 0
            iter_batch = 0

            start_iter = epoch * num_iter_epoch
            iter_max_default = num_epochs * num_iter_epoch  # affect adaptive weight value

            # initialize the embedding (for tensorboardX)
            if use_tensorboard and (epoch_embedding == epoch+1 or epoch_embedding == -1):
                feat_source_display = None
                label_source_display = None

                feat_target_display = None
                label_target_display = None

            # start training
            while batch_gen_source.has_next():
                # adaptive weight for adversarial loss
                iter_now = iter_batch + start_iter
                adaptive_beta_0 = self.adapt_weight(iter_now, iter_max_default, iter_max_beta_user[0], beta[0])
                adaptive_beta_1 = self.adapt_weight(iter_now, iter_max_default, iter_max_beta_user[1], beta[1]) / 10.0
                adaptive_gamma = self.adapt_weight(iter_now, iter_max_default, iter_max_gamma_user, gamma)
                adaptive_nu = self.adapt_weight(iter_now, iter_max_default, iter_max_nu_user, nu)
                beta_in_0 = adaptive_beta_0 if beta[0] < 0 else beta[0]
                beta_in_1 = adaptive_beta_1 if beta[1] < 0 else beta[1]
                beta_in = [beta_in_0, beta_in_1]
                gamma_in = adaptive_gamma if gamma < 0 else gamma
                nu_in = adaptive_nu if nu < 0 else nu

                # ====== Feed-forward data ====== #
                # prepare inputs
                input_source, label_source, mask_source = batch_gen_source.next_batch(batch_size_source, 'source')
                input_source, label_source, mask_source = input_source.to(device), label_source.to(device), mask_source.to(device)

                # drop some source frames (including labels) for semi-supervised learning
                input_source, label_source, mask_source = self.ctrl_video_length(input_source, label_source, mask_source, ratio_source)

                # drop source labels only
                label_source_new, mask_source_new = self.ctrl_video_label_length(label_source, mask_source, ratio_label_source)

                input_target, label_target, mask_target = batch_gen_target.next_batch(batch_size_target, 'target')
                input_target, label_target, mask_target = input_target.to(device), label_target.to(device), mask_target.to(device)

                # forward-pass data
                # label: (batch, frame#)
                # pred: (batch, stage#, class#, frame#)
                # feat: (batch, stage#, dim, frame#)
                # pred_d: (batch x frame#, stage#, class#, 2)
                # pred_d_video: (batch x seg#, stage#, 2)
                pred_source, prob_source, feat_source, pred_target, prob_target, feat_target, \
                pred_d, pred_d_video, label_d, label_d_video, \
                pred_source_2, prob_source_2, pred_target_2, prob_target_2 \
                    = model(input_source, input_target, mask_source, mask_target, beta_in, reverse=False)

                num_stages = pred_source.shape[1]

                # ------ store the embedding ------ #
                # only store the frame-level features ==> need to reshape
                if use_tensorboard and (epoch_embedding == epoch+1 or epoch_embedding == -1):
                    id_source = self.select_id_embedding(mask_source, num_frame_video_embedding)  # sample frame indices

                    feat_source_reshape = feat_source[:, stage_embedding, :, id_source].detach().transpose(1, 2).reshape(-1, num_f_maps)
                    feat_source_display = feat_source_reshape if iter_batch == 0 else torch.cat((feat_source_display, feat_source_reshape), 0)
                    label_source_reshape = label_source[:, id_source].detach().reshape(-1)
                    label_source_display = label_source_reshape if iter_batch == 0 else torch.cat((label_source_display, label_source_reshape), 0)

                    id_target = self.select_id_embedding(mask_target, num_frame_video_embedding)  # sample frame indices

                    feat_target_reshape = feat_target[:, stage_embedding, :, id_target].detach().transpose(1, 2).reshape(-1, num_f_maps)
                    feat_target_display = feat_target_reshape if iter_batch == 0 else torch.cat((feat_target_display, feat_target_reshape), 0)
                    label_target_reshape = label_target[:, id_target].detach().reshape(-1)
                    label_target_display = label_target_reshape if iter_batch == 0 else torch.cat((label_target_display, label_target_reshape), 0)

                # ------ Classification loss ------ #
                loss = 0
                for s in range(num_stages):
                    p = pred_source[:, s, :, :]  # select one stage --> (batch, class#, frame#)
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), label_source_new.view(-1))
                    loss += alpha * torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=tau ** 2) * mask_source_new[:, :, 1:])
                    if DA_ens == 'MCD' or DA_ens == 'SWD' and use_target != 'none':
                        p_2 = pred_source_2[:, s, :, :]  # select one stage --> (batch, class#, frame#)
                        loss += self.ce(p_2.transpose(2, 1).contiguous().view(-1, self.num_classes), label_source_new.view(-1))
                        loss += alpha * torch.mean(torch.clamp(self.mse(F.log_softmax(p_2[:, :, 1:], dim=1), F.log_softmax(p_2.detach()[:, :, :-1], dim=1)), min=0, max=tau ** 2) * mask_source_new[:, :, 1:])

                # ====== Domain Adaptation ====== #
                if use_target != 'none':
                    num_class_domain = pred_d.size(2)

                    if DA_ens != 'none':  # get multiple target outputs
                        _, _, _, _, prob_target, _, _, _, _, _, _, _, _, prob_target_2 \
                            = model(input_source, input_target, mask_source, mask_target, beta_in, reverse=True)

                    for s in range(num_stages):
                        # --- select data for the current stage --- #
                        # masking class prediction
                        pred_select_source, prob_select_source, prob_select_source_2, feat_select_source, label_select_source, classweight_stage_select_source \
                            = self.select_data_stage(s, pred_source, prob_source, prob_source_2, feat_source, label_source)

                        pred_select_target, prob_select_target, prob_select_target_2, feat_select_target, label_select_target, classweight_stage_select_target \
                            = self.select_data_stage(s, pred_target, prob_target, prob_target_2, feat_target, label_target)

                        # masking domain prediction
                        pred_d_stage, pred_d_video_stage, label_d_stage, label_d_video_stage \
                            = self.select_data_domain_stage(s, pred_d, pred_d_video, label_d, label_d_video)

                        # concatenate class probability masks
                        classweight_stage = torch.cat((classweight_stage_select_source, classweight_stage_select_target), 0)
                        classweight_stage_hardmask = classweight_stage == classweight_stage.max(dim=1, keepdim=True)[0]  # highest prob: 1, others: 0
                        classweight_stage_hardmask = classweight_stage_hardmask.float()

                        # ------ Adversarial loss ------ #
                        if DA_adv == 'rev_grad':
                            if place_adv[s] == 'Y':
                                # calculate loss
                                loss_adv = 0
                                for c in range(num_class_domain):
                                    pred_d_class = pred_d_stage[:, c, :]  # (batch x frame#, 2)
                                    label_d_class = label_d_stage[:, c]  # (batch x frame#)

                                    loss_adv_class = self.ce_d(pred_d_class, label_d_class)
                                    if weighted_domain_loss == 'Y' and multi_adv[1] == 'Y':  # weighted by class prediction
                                        if ps_lb == 'soft':
                                            loss_adv_class *= classweight_stage[:, c].detach()
                                        elif ps_lb == 'hard':
                                            loss_adv_class *= classweight_stage_hardmask[:, c].detach()

                                    loss_adv += loss_adv_class.mean()

                                loss += loss_adv

                                if 'rev_grad' in DA_adv_video:
                                    loss_adv_video = self.ce_d(pred_d_video_stage, label_d_video_stage)
                                    loss += loss_adv_video.mean()

                        # ------ Discrepancy loss ------ #
                        if DA_dis == 'JAN':
                            if place_dis[s] == 'Y':
                                # calculate loss
                                size_loss = min(prob_select_source.size(0), prob_select_target.size(0))  # choose the smaller number
                                size_loss = min(512, size_loss)  # avoid "out of memory" issue
                                # random indices
                                id_rand_source = torch.randperm(prob_select_source.size(0))
                                id_rand_target = torch.randperm(prob_select_target.size(0))
                                feat_source_sel = [feat_select_source[id_rand_source[:size_loss]], prob_select_source[id_rand_source[:size_loss]]]
                                feat_target_sel = [feat_select_target[id_rand_target[:size_loss]], prob_select_target[id_rand_target[:size_loss]]]

                                loss_dis = loss_jan(feat_source_sel, feat_target_sel)

                                loss += nu_in * loss_dis

                        # ------ Semantic loss between centroids ------ #
                        if method_centroid != 'none':
                            if place_sem[s] == 'Y':
                                # update centroids: (num_classes, num_f_maps)
                                centroid_source, centroid_target \
                                    = model.centroids[s].update_centroids(feat_select_source, feat_select_target, label_select_source,
                                                                          prob_select_target, method_centroid, ratio_ma)

                                # calculate semantic loss from centroids
                                if DA_sem == 'mse':
                                    loss_sem = self.mse(centroid_target, centroid_source).mean()
                                    loss += gamma_in * loss_sem

                                model.centroids[s].centroid_s = centroid_source.detach()
                                model.centroids[s].centroid_t = centroid_target.detach()

                        # ------ Ensemble loss ------ #
                        if DA_ens != 'none':
                            if place_ens[s] == 'Y':
                                loss_ens = 0
                                # calculate loss
                                if DA_ens == 'MCD':
                                    loss_ens = -dis_mcd(prob_select_target, prob_select_target_2)
                                elif DA_ens == 'SWD':
                                    loss_ens = -dis_swd(prob_select_target, prob_select_target_2, dim_proj)

                                loss += loss_ens

                        # ------ Entropy loss ------ #
                        if DA_ent == 'target':
                            if place_ent[s] == 'Y':
                                # calculate loss
                                loss_ent = cross_entropy_soft(pred_select_target)
                                loss += mu * loss_ent
                        elif DA_ent == 'attn':
                            if place_ent[s] == 'Y':
                                # calculate loss
                                loss_ent = 0
                                for c in range(num_class_domain):
                                    pred_d_class = pred_d_stage[:, c, :]  # (batch x frame#, 2)

                                    loss_ent_class = attentive_entropy(torch.cat((pred_select_source, pred_select_target), 0), pred_d_class)
                                    if weighted_domain_loss == 'Y' and multi_adv[1] == 'Y':  # weighted by class prediction
                                        if ps_lb == 'soft':
                                            loss_ent_class *= classweight_stage[:, c].detach()
                                        elif ps_lb == 'hard':
                                            loss_ent_class *= classweight_stage_hardmask[:, c].detach()

                                    loss_ent += loss_ent_class.mean()
                                loss += mu * loss_ent

                        # ------ Adversarial loss ------ #
                        if SS_video == 'VCOP':
                            if place_ss[s] == 'Y':
                                loss_ss_video = self.ce_d(pred_d_video_stage, label_d_video_stage)
                                loss += eta * loss_ss_video.mean()

                # training
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()  # record the epoch loss

                # prediction
                _, pred_id_source = torch.max(pred_source[:, -1, :, :].data, 1)  # predicted indices (from the last stage)
                correct_source += ((pred_id_source == label_source_new).float() * mask_source_new[:, 0, :].squeeze(1)).sum().item()
                total_source += torch.sum(mask_source_new[:, 0, :]).item()
                _, pred_id_target = torch.max(pred_target[:, -1, :, :].data, 1)  # predicted indices (from the last stage)
                correct_target += ((pred_id_target == label_target).float() * mask_target[:, 0, :].squeeze(1)).sum().item()
                total_target += torch.sum(mask_target[:, 0, :]).item()

                iter_batch += 1

            batch_gen_source.reset()
            batch_gen_target.reset()
            random.shuffle(batch_gen_source.list_of_examples)
            random.shuffle(batch_gen_target.list_of_examples)  # use it only when using target data

            torch.save(model.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + ".opt")

            acc_epoch_source = float(correct_source) / total_source
            acc_epoch_target = float(correct_target) / total_target
            # update the "best" model (best training acc)
            if acc_epoch_source > acc_best_source:
                acc_best_source = acc_epoch_source
                torch.save(model.state_dict(), model_dir + "/acc_best_source.model")
                torch.save(optimizer.state_dict(), model_dir + "/acc_best_source.opt")

            if acc_epoch_target > acc_best_target:
                acc_best_target = acc_epoch_target
                torch.save(model.state_dict(), model_dir + "/acc_best_target.model")
                torch.save(optimizer.state_dict(), model_dir + "/acc_best_target.opt")

            if verbose:
                print("[epoch %d]: epoch loss = %f,   acc (S) = %f,   acc (T) = %f,   beta = (%f, %f),   nu = %f" % (epoch + 1, epoch_loss / num_iter_epoch, acc_epoch_source, acc_epoch_target, beta_in[0], beta_in[1], nu_in))  # uncomment for debugging

            # ------ update the embedding every epoch ------ #
            if use_tensorboard and (epoch_embedding == epoch+1 or epoch_embedding == -1):
                # generate domain labels
                label_source_domain_display = torch.full_like(feat_source_display[:, 0], 0)
                label_target_domain_display = torch.full_like(feat_target_display[:, 0], 1)

                # mix source and target
                feat_all_display = torch.cat((feat_source_display, feat_target_display), 0)
                label_all_class_display = torch.cat((label_source_display, label_target_display), 0)
                label_all_domain_display = torch.cat((label_source_domain_display, label_target_domain_display), 0)
                label_all_display = list(zip(label_all_class_display, label_all_domain_display))
                writer.add_embedding(feat_all_display, metadata=label_all_display, metadata_header=['class', 'domain'], global_step=iter_now)

        if use_tensorboard:
            writer.close()

    def select_data_stage(self, s, pred, prob, prob_2, feat, label):
        dim_feat = feat.size(2)

        # features & prediction
        feat_stage = feat[:, s, :, :]  # select one stage --> (batch, dim, frame#)
        feat_frame = feat_stage.transpose(1, 2).reshape(-1, dim_feat)
        pred_stage = pred[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        pred_frame = pred_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_stage = prob[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_frame = prob_stage.transpose(1, 2).reshape(-1, self.num_classes)
        prob_2_stage = prob_2[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        prob_2_frame = prob_2_stage.transpose(1, 2).reshape(-1, self.num_classes)

        # select the masked frames & labels
        label_vector = label.reshape(-1).clone()
        feat_select = feat_frame[label_vector != -100]
        pred_select = pred_frame[label_vector != -100]
        label_select = label_vector[label_vector != -100]
        prob_select = prob_frame[label_vector != -100]
        prob_2_select = prob_2_frame[label_vector != -100]

        # class probability as class weights
        classweight_stage = prob[:, s, :, :]  # select one stage --> (batch, class#, frame#)
        classweight_stage = classweight_stage.transpose(1, 2).reshape(-1, self.num_classes)  # (batch x frame#, class#)

        # mask frames
        classweight_stage_select = classweight_stage[label_vector != -100]

        return pred_select, prob_select, prob_2_select, feat_select, label_select, classweight_stage_select

    def select_data_domain_stage(self, s, pred_d, pred_d_video, label_d, label_d_video):

        # domain predictions & labels (frame-level)
        pred_d_select = pred_d[:, s, :, :]  # select one stage --> (batch x frame#, class#, 2)
        label_d_select = label_d[:, s, :]  # select one stage --> (batch x frame#, class#)

        # domain predictions & labels (video-level)
        pred_d_select_seg = pred_d_video[:, s, :]  # select one stage --> (batch x seg#, 2)
        label_d_select_video = label_d_video[:, s]  # select one stage --> (batch x seg#)

        return pred_d_select, pred_d_select_seg, label_d_select, label_d_select_video

    def select_id_embedding(self, mask, num_frame_select):
        # sample frame indices
        num_frame_min = mask[:, 0, :].sum(-1).min()  # length of shortest video
        if num_frame_min.item() < num_frame_select:
            raise ValueError('space between frames should be at least 1!')
        index = torch.tensor(np.linspace(0, num_frame_min.item()-1, num_frame_select).tolist()).long()
        if mask.get_device() >= 0:
            index = index.to(mask.get_device())

        return index

    def ctrl_video_length(self, input_data, label, mask, ratio_length):
        # shapes:
        # input_data: (batch, dim, frame#)
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)

        # get the indices of the frames to keep
        num_frame = input_data.size(-1)  # length of video
        num_frame_drop = (1 - ratio_length) * num_frame
        id_drop = np.floor(np.linspace(0, num_frame-1, num_frame_drop)).tolist()
        id_keep = list(set(range(num_frame)) - set(id_drop))
        id_keep = torch.tensor(id_keep).long()
        if input_data.get_device() >= 0:
            id_keep = id_keep.to(input_data.get_device())

        # filter the inputs w/ the above indices
        input_data_filtered = input_data[:, :, id_keep]
        label_filtered = label[:, id_keep]
        mask_filtered = mask[:, :, id_keep]

        return input_data_filtered, label_filtered, mask_filtered

    def ctrl_video_label_length(self, label, mask, ratio_length):
        # shapes:
        # label: (batch, frame#)
        # mask: (batch, class#, frame#)
        mask_new = mask.clone()
        label_new = label.clone()

        # get the indices of the frames to keep
        num_frame = mask.size(-1)  # length of video
        num_frame_drop = (1 - ratio_length) * num_frame
        id_drop = np.floor(np.linspace(0, num_frame-1, num_frame_drop)).tolist()
        id_drop = torch.tensor(id_drop).long()
        if mask.get_device() >= 0:
            id_drop = id_drop.to(mask.get_device())

        # assign 0 to the above indices
        mask_new[:, :, id_drop] = 0
        label_new[:, id_drop] = -100  # class id -100 won't be calculated in cross-entropy

        return label_new, mask_new
