import random

import torch
import torch.nn.functional as F
import numpy as np
import wandb

from utils.metrics import AverageMeterSet


class ValidationMetricsCalculator:
    def __init__(self, composed_query_features,composed_query_features2,composed_query_features3,composed_query_features4,
                  test_features, test_features2,test_features3,test_features4, attribute_matching_matrix: np.array,
                  ref_attribute_matching_matrix: np.array, top_k: tuple,
                  all_ref_attributes, all_query_attributes, all_test_attributes, all_modifiers):

    #def __init__(self, composed_query_features,
                 #test_features, attribute_matching_matrix: np.array,
                 #ref_attribute_matching_matrix: np.array, top_k: tuple,
                 #all_ref_attributes, all_query_attributes, all_test_attributes, all_modifiers):
        self.composed_query_features = composed_query_features
        self.composed_query_features2 = composed_query_features2
        self.composed_query_features3 = composed_query_features3
        self.composed_query_features4 = composed_query_features4
        self.test_features = test_features
        self.test_features2 = test_features2
        self.test_features3 = test_features3
        self.test_features4 = test_features4
        self.top_k = top_k
        self.attribute_matching_matrix = attribute_matching_matrix
        self.ref_attribute_matching_matrix = ref_attribute_matching_matrix
        self.num_query_features = composed_query_features.size(0)
        self.num_test_features = test_features.size(0)
        self.similarity_matrix = torch.zeros(self.num_query_features, self.num_test_features)
        self.top_scores = torch.zeros(self.num_query_features, max(top_k))
        self.most_similar_idx = torch.zeros(self.num_query_features, max(top_k))
        self.recall_results = {}
        self.recall_positive_queries_idxs = {k: [] for k in top_k}
        #self.recall_negative_queries_idxs = {k: [] for k in top_k}
        self.similarity_matrix_calculated = False
        self.top_scores_calculated = False
        self.all_ref_attributes = np.array(all_ref_attributes)
        self.all_query_attributes = np.array(all_query_attributes)
        self.all_test_attributes = np.array(all_test_attributes)
        self.all_modifiers = np.array(all_modifiers)

    def __call__(self):
        self._calculate_similarity_matrix()
        # Filter query_feat == target_feat
        assert self.similarity_matrix.shape == self.ref_attribute_matching_matrix.shape
        self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
        return self._calculate_recall_at_k()

    def return_similarity(self):
        return self.similarity_matrix

    def _calculate_similarity_matrix(self) -> torch.tensor:
        """
        query_features = torch.tensor. Size = (N_test_query, Embed_size)
        test_features = torch.tensor. Size = (N_test_dataset, Embed_size)
        output = torch.tensor, similarity matrix. Size = (N_test_query, N_test_dataset)
        """
        if not self.similarity_matrix_calculated:
            self.similarity_matrix = self.composed_query_features.mm(self.test_features.t())
            self.similarity_matrix2 = self.composed_query_features2.mm(self.test_features2.t())
            self.similarity_matrix3 = self.composed_query_features3.mm(self.test_features3.t())
            self.similarity_matrix4 = self.composed_query_features4.mm(self.test_features4.t())

            tmp_similarity1 = self.similarity_matrix.clone()
            tmp_similarity2 = self.similarity_matrix2.clone()
            tmp_similarity3 = self.similarity_matrix3.clone()
            tmp_similarity4 = self.similarity_matrix4.clone()

            self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
            layer4_result = self._calculate_recall_at_k()
            self.similarity_matrix = self.similarity_matrix2
            self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
            layer2_result = self._calculate_recall_at_k()
            self.similarity_matrix = self.similarity_matrix3
            self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
            layer3_result = self._calculate_recall_at_k()
            self.similarity_matrix = self.similarity_matrix4
            self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min()
            layert_result = self._calculate_recall_at_k()

            self.similarity_matrix = tmp_similarity1
            self.similarity_matrix2 = tmp_similarity2
            self.similarity_matrix3 = tmp_similarity3
            self.similarity_matrix4 = tmp_similarity4
            print('layer4_result:{}'.format(layer4_result))
            print('layert_result:{}'.format(layer2_result))
            print('layert3_result:{}'.format(layert_result))
            print('layer3_result:{}'.format(layer3_result))

            self.similarity_matrix = 1*self.similarity_matrix + 0.5*self.similarity_matrix3 + .5*self.similarity_matrix2+.5*self.similarity_matrix4

            self.similarity_matrix_calculated = True

    def _calculate_recall_at_k(self):
        average_meter_set = AverageMeterSet()
        self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))
        #_, self.most_similar_idx = self.rank_matrix.topk(max(self.top_k))
        self.top_scores_calculated = True
        topk_attribute_matching = np.take_along_axis(self.attribute_matching_matrix, self.most_similar_idx.numpy(),
                                                     axis=1)

        for k in self.top_k:
            query_matched_vector = topk_attribute_matching[:, :k].sum(axis=1).astype(bool)
            self.recall_positive_queries_idxs[k] = list(np.where(query_matched_vector > 0)[0])

            if k == 10:
                recall_negative_queries_idxs = list(np.where(query_matched_vector > 0)[0]) #wrong ref index
                most_similar_idx = self.most_similar_idx
                orderd_target_attribute = self.all_test_attributes[most_similar_idx]
                wrong_target_attribute = orderd_target_attribute[recall_negative_queries_idxs]
                wrong_query_attribute = self.all_query_attributes[recall_negative_queries_idxs]
                wrong_ref_attribute = self.all_ref_attributes[recall_negative_queries_idxs]
                wrong_modifiers = self.all_modifiers[recall_negative_queries_idxs]
                wrongs = []
                for i in range(len(wrong_ref_attribute)):
                    key = wrong_ref_attribute[i]
                    modifier = wrong_modifiers[i]
                    targ = wrong_query_attribute[i]
                    wronglst = wrong_target_attribute[i][:5]
                    wrongs.append([modifier]+[key]+[targ]+wronglst.tolist())
                f = open('wronglst.txt', 'w')
                for l in wrongs:
                    f.write(str(l)+'\n')
                f.close()

            num_correct = query_matched_vector.sum()
            num_samples = len(query_matched_vector)
            average_meter_set.update('recall_@{}'.format(k), num_correct, n=num_samples)
        recall_results = average_meter_set.averages()
        return recall_results

    def get_positive_sample_info(self, num_samples, num_imgs_per_sample, positive_at_k):
        info = []
        num_samples = min(num_samples, len(self.recall_positive_queries_idxs[positive_at_k]))
        for ref_idx in random.sample(self.recall_positive_queries_idxs[positive_at_k], num_samples):
            targ_img_ids = self.most_similar_idx[ref_idx, :num_imgs_per_sample].tolist()
            targ_scores = self.top_scores[ref_idx, :num_imgs_per_sample].tolist()
            gt_idx = np.where(self.attribute_matching_matrix[ref_idx, :] == True)[0]
            gt_score = self.similarity_matrix[ref_idx, gt_idx[0]].item()
            info.append(
                {'ref_idx': ref_idx, 'targ_idxs': targ_img_ids, 'targ_scores': targ_scores, 'gt_score': gt_score})
        return info

    def get_negative_sample_info(self, num_samples, num_imgs_per_sample, negative_at_k):
        info = []
        negative_idxs_list = list(
            set(range(self.num_query_features)) - set(self.recall_positive_queries_idxs[negative_at_k]))
        num_samples = min(num_samples, len(negative_idxs_list))
        for ref_idx in random.sample(negative_idxs_list, num_samples):
            targ_img_ids = self.most_similar_idx[ref_idx, :num_imgs_per_sample].tolist()
            targ_scores = self.top_scores[ref_idx, :num_imgs_per_sample].tolist()
            gt_idx = np.where(self.attribute_matching_matrix[ref_idx, :] == True)[0]
            gt_score = self.similarity_matrix[ref_idx, gt_idx[0]].item()
            info.append(
                {'ref_idx': ref_idx, 'targ_idxs': targ_img_ids, 'targ_scores': targ_scores, 'gt_score': gt_score})
        return info

    def VOCap(rec, prec):
        mrec = np.append(0, rec)
        mrec = np.append(mrec, 1)

        mpre = np.append(0, prec)
        mpre = np.append(mpre, 0)

        for ii in range(len(mpre)-2,-1,-1):
            mpre[ii] = max(mpre[ii], mpre[ii+1])

        msk = [i!=j for i,j in zip(mrec[1:], mrec[0:-1])]
        ap = np.sum((mrec[1:][msk]-mrec[0:-1][msk])*mpre[1:][msk])
        return ap

    def eval_AP_inner(inst_id, scores, gt_labels, top=None):
        pos_flag = gt_labels == inst_id
        tot = scores.shape[0]
        tot_pos = np.sum(pos_flag)

        sort_idx = np.argsort(-scores)
        tp = pos_flag[sort_idx]
        fp = np.logical_not(tp)

        if top is not None:
            top = min(top, tot)
            tp = tp[:top]
            fp = fp[:top]
            tot_pos = min(top, tot_pos)

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / tot_pos
        prec = tp / (tp + fp)

        ap = VOCap(rec, prec)
        return ap

    def get_similarity_histogram(self, negative_hist_topk=10) -> (wandb.Histogram, wandb.Histogram, wandb.Histogram):
        self._calculate_similarity_matrix()
        sim_matrix_np = self.similarity_matrix.numpy()
        original_features_sim_matrix_np = self.original_query_features.mm(self.test_features.t()).numpy()

        if not self.top_scores_calculated:
            self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))

        # Get the scores of negative images that are in topk=negative_hist_topk
        hardest_k_negative_mask = np.zeros_like(self.attribute_matching_matrix)
        np.put_along_axis(hardest_k_negative_mask, self.most_similar_idx[:, :negative_hist_topk].numpy(), True, axis=1)
        hardest_k_negative_mask = hardest_k_negative_mask & ~self.attribute_matching_matrix

        composed_positive_score_distr = sim_matrix_np[self.attribute_matching_matrix]
        composed_negative_score_distr = sim_matrix_np[hardest_k_negative_mask]
        original_positive_score_distr = original_features_sim_matrix_np[self.attribute_matching_matrix]

        composed_pos_histogram = wandb.Histogram(np_histogram=np.histogram(composed_positive_score_distr, bins=200))
        composed_neg_histogram = wandb.Histogram(np_histogram=np.histogram(composed_negative_score_distr, bins=200))
        original_pos_histogram = wandb.Histogram(np_histogram=np.histogram(original_positive_score_distr, bins=200))

        return composed_pos_histogram, composed_neg_histogram, original_pos_histogram

    @staticmethod
    def _multiple_index_from_attribute_list(attribute_list, indices):
        attributes = []
        for idx in indices:
            attributes.append(attribute_list[idx.item()])
        return attributes
