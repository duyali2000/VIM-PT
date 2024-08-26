# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time

import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import numpy as np
import random

postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "Javascript":"js"}


def is_nan(num):
    return num != num

class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """
    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None, args=None):
        super(Seq2Seq, self).__init__()
        self.lingual_number = args.lingual_number
        self.encoder = encoder
        self.decoder = nn.ModuleList([copy.deepcopy(decoder) for _ in range(self.lingual_number)])
        #self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.lingual_number)])
        self.lm_head = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(self.lingual_number)])
        self.lsm = nn.ModuleList([nn.LogSoftmax(dim=-1) for _ in range(self.lingual_number)])
        self.tie_weights()

        self.beam_size=beam_size
        self.max_length=max_length
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.similarity_metric = "l2"
        # size of memory-bank
        self.K = 256

        self.q_x_fc_mu = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(inplace=False)
        ) for _ in range(self.lingual_number)])
        self.q_x_fc_var = copy.deepcopy(self.q_x_fc_mu)
        self.r_x_fc_mu = copy.deepcopy(self.q_x_fc_mu)
        self.r_x_fc_var = copy.deepcopy(self.q_x_fc_mu)
        self.q_s_fc_mu = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(inplace=False)
        )
        self.q_s_fc_var = copy.deepcopy(self.q_s_fc_mu)

        self.p_projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LeakyReLU(inplace=False)
        ) for _ in range(self.lingual_number)])


        # loss
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        # queue
        # create the queue
        self.register_buffer("queue", torch.randn(self.K, self.lingual_number, args.max_source_length))
        self.train_ids_queue = nn.functional.normalize(self.queue, dim=0).cuda()
        self.train_masks_queue = nn.functional.normalize(self.queue, dim=0).cuda()
        self.register_buffer("queue_labels", torch.randn(self.K, self.lingual_number))
        self.train_labels_queue = nn.functional.normalize(self.queue_labels, dim=0).cuda()

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))



    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_ids, key_masks, key_labels):
        # gather keys before updating queue
        batch_size = key_ids.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.train_ids_queue[ptr: ptr + batch_size,:] = key_ids
        self.train_masks_queue[ptr: ptr + batch_size,:] = key_masks
        self.train_labels_queue[ptr: ptr + batch_size,:] = key_labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr


    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight


    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        for i in range(self.lingual_number):
            self._tie_or_clone_weights(self.lm_head[i],
                                   self.encoder.embeddings.word_embeddings)


    def test_unit(self, lingual_num, encoder_output, source_mask):
        # Predict
        decoder = self.decoder[lingual_num]
        #decoder = self.decoder
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        for i in range(source_mask.shape[0]):
            context = encoder_output[:, i:i + 1]
            context_mask = source_mask[i:i + 1, :]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context = context.repeat(1, self.beam_size, 1)
            context_mask = context_mask.repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break
                decoder_attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                # print(tgt_embeddings.shape, context.shape,decoder_attn_mask.shape)
                out = decoder(tgt_embeddings, context, tgt_mask=decoder_attn_mask,
                                   memory_key_padding_mask=(1 - context_mask).bool())
                out = torch.tanh(self.dense[lingual_num](out))
                hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                out = self.lsm[lingual_num](self.lm_head[lingual_num](hidden_states)).data
                beam.advance(out)

                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                    pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))
        preds = torch.cat(preds, 0)
        return preds

    def sample(self, source_ids, source_masks, labels):
        output_ids = copy.deepcopy(source_ids)
        output_masks = copy.deepcopy(source_masks)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                # Check that the label is 0, and if it is, select a sample location at random and assign its corresponding location's source_ids and source_masks to the current location
                if labels[i, j] == torch.tensor(0):
                    while True:
                        # random select a sample
                        random_idx = random.randint(0, self.K - 1)
                        if(random_idx == i):
                            continue
                        # If the label for the random sample location is not 0, assign the value to the source_ids and source_masks for the current location
                        if self.train_labels_queue[random_idx, j] != torch.tensor(0):
                            output_ids[i, j, :] = self.train_ids_queue[random_idx, j, :]
                            output_masks[i, j, :] = self.train_masks_queue[random_idx, j, :]
                            break  

        return output_ids, output_masks


    def encoding(self, input_ids, input_mask):
        encoder_outputs = self.encoder(input_ids, attention_mask=input_mask)
        encoder_output = encoder_outputs[0].permute([1, 0, 2]).contiguous()
        return encoder_output


    def q_inferrer(self, lingual_num, features):
        # Check that lingual_num is valid
        if lingual_num < 0 or lingual_num >= self.lingual_number:
            raise ValueError("Invalid lingual_num")

        fc_mu = self.q_x_fc_mu[lingual_num]
        fc_var = self.q_x_fc_var[lingual_num]
        
        mu = fc_mu(features)
        log_var = fc_var(features)
        return [mu, log_var]


    def r_inferrer(self, lingual_num, features):
        # Check that lingual_num is valid
        if lingual_num < 0 or lingual_num >= self.lingual_number:
            raise ValueError("Invalid lingual_num")

        fc_mu = self.r_x_fc_mu[lingual_num]
        fc_var = self.r_x_fc_var[lingual_num]

        mu = fc_mu(features)
        log_var = fc_var(features)
        return [mu, log_var]

    def p_inferrer(self, lingual_num, specific_feature, shared_feature):
        if lingual_num < 0 or lingual_num >= self.lingual_number:
            raise ValueError("Invalid lingual_num")
        fused_feature = torch.mean(torch.stack((specific_feature, shared_feature), dim=0), dim=0)
        projector = self.p_projectors[lingual_num]
        encoded_feature = projector(fused_feature)
        return encoded_feature

    def fuse(self, p_features, token_features):
        fused_feature = torch.cat((p_features, token_features), dim=0).cuda()
        return fused_feature

    def replace_pseduo(self, special_length, input_labels, source_ids, source_masks, p_features_list):
        # [max_length, batchsize, embedding]
        encoder_outputs = self.encoder(source_ids, attention_mask=source_masks)
        encoder_output = encoder_outputs[0].permute([1, 0, 2]).contiguous()
        input_embeding = encoder_output[:special_length,:,:]
        for idx in range(input_labels.shape[0]):
            if (input_labels[idx] == torch.tensor(0)):
                #print(input_embeding.shape, p_features_list.shape)
                input_embeding[:, idx, :] = p_features_list[:, idx, :]
        return input_embeding

    def q_shared_inferrer(self, input_embedings):
        # [batchsize, lingual_num, embedding]
        mean_features = torch.mean(input_embedings, dim=0).cuda()
        
        fc_mu = self.q_s_fc_mu
        fc_var = self.q_s_fc_var
        
        mu = fc_mu(mean_features)
        log_var = fc_var(mean_features)
        return [mu, log_var]


    def compute_kl_loss(self, mean, logv, normalization=1):
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_loss = kl_loss / normalization
        return kl_loss


    def decoding(self, args, lingual_num, encoder_output, target_ids, target_mask, label):
        code_length = args.max_source_length - args.max_comment_length
        target_code_ids = target_ids[:, args.special_length:code_length]
        target_code_mask = target_mask[:, args.special_length:code_length]

        tgt_embeddings = self.encoder.embeddings(target_code_ids).permute([1, 0, 2]).contiguous()
        decoder_attn_mask = -1e4 * (1 - self.bias[:target_code_ids.shape[1], :target_code_ids.shape[1]])
        decoder = self.decoder[lingual_num]
        decoder_output = decoder(tgt_embeddings, encoder_output, tgt_mask=decoder_attn_mask,
                                 memory_key_padding_mask=(1 - target_mask).bool())
        hidden_states = torch.tanh(self.dense[lingual_num](decoder_output)).permute([1, 0, 2]).contiguous()

        lm_logits = self.lm_head[lingual_num](hidden_states)
       

        loss_ = 0.0
        for i in range(label.shape[0]):
            if(label[i]!=torch.tensor(0)):
                active_loss = target_code_mask[i, 1:].ne(0).view(-1) == 1
                #shift_logits = lm_logits[i, :-1, :].contiguous()
                shift_logits = lm_logits[i, :-1, :].contiguous()
                shift_labels = target_code_ids[i, 1:].contiguous()
                # Flatten the tokens
                loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                     shift_labels.view(-1)[active_loss])
                if (~is_nan(loss)):
                    loss_ += loss * label[i]
        return decoder_output, loss_

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def compute_mse_loss(self, encoder_output_list, p_features_list, label):
        loss_ = 0.0
        for idx in range(label.shape[0]):
            for lingual in range(label.shape[1]):
                if (label[idx, lingual] != torch.tensor(0)):
                    # Flatten the tokens
                    loss = self.loss_mse(encoder_output_list[lingual, idx, :], p_features_list[lingual, idx, :])
                    loss_ += loss
        return loss_


    def forward(self, state, args, source_ids, source_masks, labels, source_lingual=None, warm_up_step=None):
        if(state == 'initialize'):
            self._dequeue_and_enqueue(source_ids, source_masks, labels)
        elif (state == 'auto'):
            loss_auto = 0
            source_ids, source_masks = self.sample(source_ids, source_masks, labels)
            for lingual in range(self.lingual_number):
                input_ids = source_ids[:, lingual, :]
                input_masks = source_masks[:, lingual, :]
                encoder_output = self.encoding(input_ids, input_masks)
                decoder_input = encoder_output
                target_ids = source_ids[:, lingual, :]
                target_masks = source_masks[:, lingual, :]
                labels = torch.ones(labels.shape[0])
                decoder_output, loss = self.decoding(args, lingual, decoder_input, target_ids, target_masks, labels)
                loss_auto += loss
            return (1+self.lamda) * loss_auto

        elif (state == 'train' or state == 'finetune'):
            source_ids, source_masks = self.sample(source_ids, source_masks, labels)
            loss_tran, loss_mse, loss_KL_z_x, loss_KL_z_s, loss_KL_z_r = 0, 0, 0, 0, 0
            encoder_output_list = torch.zeros((self.lingual_number, args.max_source_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            q_x_features_list = torch.zeros((self.lingual_number, args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            r_mu_list = torch.zeros((self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            r_var_list = torch.zeros((self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()

            #for lingual in range(self.lingual_number):
            input_ids = source_ids[:,source_lingual,:]
            input_masks = source_masks[:,source_lingual,:]
            encoder_output = self.encoding(input_ids, input_masks)
            [q_mu, q_log_var] = self.q_inferrer(source_lingual, encoder_output[:args.special_length,:,:])
            [r_mu, r_log_var] = self.r_inferrer(source_lingual, encoder_output[:args.special_length,:,:])
            q_x_features = self.reparameterize(q_mu, q_log_var)
            encoder_output_list[source_lingual,:,:,:] = encoder_output
            q_x_features_list[source_lingual,:,:,:] = q_x_features
            r_mu_list[source_lingual,:,:,:] = r_mu
            r_var_list[source_lingual,:,:,:] = r_log_var

            # get r(zs|x) of source domain
            z_s_r_mu = r_mu_list[source_lingual,:,:,:]
            z_s_r_var = r_var_list[source_lingual,:,:,:]
            z_s_r = self.reparameterize(z_s_r_mu, z_s_r_var)
           

            p_features_list = torch.zeros((self.lingual_number, args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for lingual in range(self.lingual_number):
                p_features = self.p_inferrer(lingual, q_x_features_list[lingual,:,:,:], z_s_r)
                p_features_list[lingual,:,:,:] = p_features
            
            loss_mse += self.compute_mse_loss(encoder_output_list[:, :args.special_length, :, :], p_features_list, labels)

            
            second_q_x_features_list = torch.zeros((self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            input_embeddings = torch.zeros((self.lingual_number, args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for lingual in range(self.lingual_number):
                input_ids = source_ids[:, lingual, :]
                input_masks = source_masks[:, lingual, :]
                input_labels = labels[:,lingual]
                # if loss, use pseudo sample complete
                input_embedding = self.replace_pseduo(args.special_length, input_labels, input_ids, input_masks, p_features_list[lingual,:, :, :])
                input_embeddings[lingual, :,:, :] = input_embedding
                #encoder_output = self.second_encoding(input_ids, input_masks, input_labels, pseduo_embeddings[lingual,:,:,:])
                #input_embedding = input_embeddings[lingual,:,:]
                [q_mu, q_log_var] = self.q_inferrer(lingual, input_embedding)
                q_x_features = self.reparameterize(q_mu, q_log_var)
                second_q_x_features_list[lingual, :,:, :] = q_x_features
                kl_z_x = self.compute_kl_loss(q_mu, q_log_var)
                loss_KL_z_x += kl_z_x

            [q_share_mu, q_share_log_var] = self.q_shared_inferrer(input_embeddings)
            q_shared_features = self.reparameterize(q_share_mu, q_share_log_var)
            second_p_features_list = torch.zeros((self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            kl_z_s = self.compute_kl_loss(q_share_mu, q_share_log_var)
            loss_KL_z_s += kl_z_s

            for lingual in range(self.lingual_number):
                p_features = self.p_inferrer(lingual, second_q_x_features_list[lingual, :,:, :], q_shared_features)
                second_p_features_list[lingual, :,:, :] = p_features

            encoder_output = encoder_output_list[source_lingual, :, :, :]
            for lingual in range(self.lingual_number):
                p_features = second_p_features_list[lingual, :,:, :]
                decoder_input = self.fuse(p_features, encoder_output[args.special_length:,: :])
                target_ids = source_ids[:, lingual, :]
                target_masks = source_masks[:, lingual, :]
                mask_labels = labels[:, source_lingual] & labels[:, lingual]
                decoder_output, loss = self.decoding(args, lingual, decoder_input, target_ids, target_masks, mask_labels)
                loss_tran += loss

                z_s_r_mu = r_mu_list[lingual,:,:,:]
                z_s_r_var = r_var_list[lingual,:,:,:]
                kl_loss = 0.5 * (-1 + (z_s_r_var - q_share_log_var) + (q_share_log_var.exp() + (q_share_mu - z_s_r_mu).pow(2)) / z_s_r_var.exp()).sum(1).mean()
                loss_KL_z_r += kl_loss


            # warm_up_step 0 -> 1
            beta = torch.tensor(0.0001 * warm_up_step, requires_grad=False)
            return (1+self.lamda) * (loss_tran) + (1+self.lamda) * (loss_mse) + beta * loss_KL_z_s + beta * (1+self.lamda) * loss_KL_z_x + beta * self.lamda * loss_KL_z_r
           
        else:
            # Predict
            preds_dict = {}
            #for source_lingual in range(self.lingual_number):
            source_labels = torch.zeros(labels.shape)
            source_labels[:, source_lingual] = torch.tensor(1)
            pseduo_source_ids, pseduo_source_masks = self.sample(source_ids, source_masks, source_labels)
            encoder_output_list = torch.zeros((self.lingual_number, args.max_source_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            q_features_list = torch.zeros((self.lingual_number, args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for tgt_lingual in range(self.lingual_number):
                input_ids = pseduo_source_ids[:, tgt_lingual, :]
                input_masks = pseduo_source_masks[:, tgt_lingual, :]
                encoder_output = self.encoding(input_ids, input_masks)
                if (tgt_lingual == source_lingual):
                    [r_mu, r_log_var] = self.r_inferrer(tgt_lingual, encoder_output[:args.special_length,:, :])
                    r_features = self.reparameterize(r_mu, r_log_var)
                encoder_output_list[tgt_lingual, :, :, :] = encoder_output
                [q_mu, q_log_var] = self.q_inferrer(tgt_lingual, encoder_output[:args.special_length, :, :])
                q_x_features = self.reparameterize(q_mu, q_log_var)
                q_features_list[tgt_lingual, :,:, :] = q_x_features


            # get p(x|q_x, r_s) for each domain
            p_features_list = torch.zeros((self.lingual_number, args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for tgt_lingual in range(self.lingual_number):
                q_x_feature = q_features_list[tgt_lingual,:, :, :]
                p_features = self.p_inferrer(tgt_lingual, q_x_feature, r_features)
                p_features_list[tgt_lingual,:, :, :] = p_features

            # Second round, all the missing instances have been filled in
            second_q_x_features_list = torch.zeros(
                (self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            input_embeddings = torch.zeros(
                (self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for lingual in range(self.lingual_number):
                input_ids = source_ids[:, lingual, :]
                input_masks = source_masks[:, lingual, :]
                input_labels = source_labels[:, lingual]
                # 如果缺失，则用pseduo补全
                input_embedding = self.replace_pseduo(args.special_length, input_labels, input_ids, input_masks,
                                                      p_features_list[lingual, :, :])
                input_embeddings[lingual,:, :, :] = input_embedding
                [q_mu, q_log_var] = self.q_inferrer(lingual, input_embedding)
                q_x_features = self.reparameterize(q_mu, q_log_var)
                second_q_x_features_list[lingual, :,:, :] = q_x_features

            [q_share_mu, q_share_log_var] = self.q_shared_inferrer(input_embeddings)
            q_shared_features = self.reparameterize(q_share_mu, q_share_log_var)
            second_p_features_list = torch.zeros((self.lingual_number,args.special_length, source_ids.shape[0], self.config.hidden_size)).cuda()
            for tgt_lingual in range(self.lingual_number):
                p_features = self.p_inferrer(tgt_lingual, second_q_x_features_list[tgt_lingual, :,:, :], q_shared_features)
                # print(p_features.shape)
                second_p_features_list[tgt_lingual,:, :, :] = p_features

            source_mask = source_masks[:, source_lingual, :]
            encoder_output_ = encoder_output_list[source_lingual, :, :, :]
            for tgt_lingual in range(self.lingual_number):
                if (source_lingual == tgt_lingual):
                    continue
                p_features = second_p_features_list[tgt_lingual, :,:, :]
                decoder_input = self.fuse(p_features, encoder_output_[args.special_length:,: :])
                preds = self.test_unit(tgt_lingual, decoder_input, source_mask)
                preds_dict[str(source_lingual)+'-'+str(tgt_lingual)] = preds
                
            return preds_dict

    def score(self, Q, D):
        if self.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        elif self.similarity_metric == 'l2':
            #return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)
            return (-1.0 * ((Q.expand(D.shape) - D) ** 2).sum(-1)).sum(-1)


class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        #prevK = bestScoresId // numWords
        prevK = (bestScoresId / numWords).long()
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

