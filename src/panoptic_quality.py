import torch


class PanopticQuality:
    def __init__(self, device):
        # initialization of a list of dictionaries with len = num classes
        self.panoptic_qualities = {}
        self.device = device

    def compute_pq_single_class(self, prediction, groundtruth):
        numerator_iou = torch.tensor(0.).to(self.device)
        class_matches = []
        unmatched_class_predictions = []

        # take the non-zero INSTANCE labels for both prediction and groundtruth
        labels = torch.unique(prediction)
        gt_labels = torch.unique(groundtruth)
        labels = labels[labels > 0]
        gt_labels = gt_labels[gt_labels > 0]
        for label in labels:    # for each predicted label
            iou = torch.tensor(0.)
            best_gt_label = -1
            for gt_label in gt_labels:
                # compute iou with all instance gt labels and store the best
                intersection = (((prediction == label) & (groundtruth == gt_label)).sum()).float()
                union = ((prediction == label).sum() + (groundtruth == gt_label).sum() - intersection).float()
                iou_tmp = intersection / union
                if iou_tmp > iou:
                    iou = iou_tmp
                    best_gt_label = gt_label
            # if the best iou is above 0.5, store the match pred_label-gt_label-iou
            if iou > 0.5:
                result = {'pred_label': label.item(), 'gt_label': best_gt_label.item(), 'iou': iou.item()}
                class_matches.append(result)
                numerator_iou += iou
            # else, the predicted label remains unmatched
            else:
                unmatched_class_predictions.append(label.item())

        true_positives = len(class_matches)     # TP = number of matches
        false_positives = len(unmatched_class_predictions)  # FP = number of unmatched predictions
        false_negatives = len(gt_labels) - len(class_matches)   # FN = number of unmatched gt labels
        panoptic_quality_one_class = numerator_iou / (true_positives + 0.5 * false_positives + 0.5 * false_negatives)

        return panoptic_quality_one_class, class_matches

    def compute_pq(self, semantic_pred, semantic_gt, instance_pred, instance_gt):
        # take the semantic labels (except 0 = void label)
        semantic_gt_labels = torch.unique(semantic_gt).int()
        semantic_gt_labels = semantic_gt_labels[semantic_gt_labels > 0].int()

        for gt_label in semantic_gt_labels:
            # take pred and gt where they have sem_class = label
            semantic_tmp = (semantic_pred == gt_label).int()
            semantic_gt_tmp = (semantic_gt == gt_label).int()

            # take pred and gt instances corresponding to the considered semantic class
            instance_tmp = (instance_pred * semantic_tmp).int()
            instance_gt_tmp = (instance_gt * semantic_gt_tmp).int()

            pq, _ = self.compute_pq_single_class(instance_tmp, instance_gt_tmp)

            # update: count = number of images that contains the class, pq is the overall panoptic quality of the images
            if gt_label.item() not in self.panoptic_qualities.keys():
                self.panoptic_qualities[gt_label.item()] = {}
                self.panoptic_qualities[gt_label.item()]['count'] = 1
                self.panoptic_qualities[gt_label.item()]['pq'] = pq.item()
            else:
                self.panoptic_qualities[gt_label.item()]['count'] += 1
                self.panoptic_qualities[gt_label.item()]['pq'] = ((self.panoptic_qualities[gt_label.item()]['pq'] * (self.panoptic_qualities[gt_label.item()]['count']-1) + pq) / self.panoptic_qualities[gt_label.item()]['count']).item()

        panoptic_quality = self.average_pq(self.panoptic_qualities)

        return panoptic_quality, self.panoptic_qualities

    def average_pq(self, panoptic_qualities):
        # overall panoptic quality averaged from all classes
        pq = 0.
        n = 0
        for item in panoptic_qualities.keys():
            pq += panoptic_qualities[item]['pq']
            n += 1
        pq /= n
        return pq

    def panoptic_quality_forward(self, batch_semantic_pred, batch_semantic_gt, batch_instance_pred, batch_instance_gt):
        batch_size = batch_semantic_pred.shape[0]
        for item in range(batch_size):
            semantic_gt = batch_semantic_gt[item]
            semantic_pred = batch_semantic_pred[item]
            instance_gt = batch_instance_gt[item]
            instance_pred = batch_instance_pred[item]
            # for each batch item
            pq, pqs = self.compute_pq(semantic_pred, semantic_gt, instance_pred, instance_gt)
        return pq, pqs


def main():
    sem_gt = torch.tensor([[1, 1, 1], [2, 2, 2], [0, 0, 1]]).unsqueeze(0)
    ins_gt = torch.tensor([[1, 1, 1], [3, 3, 4], [12, 13, 2]]).unsqueeze(0)
    sem_pr = torch.tensor([[0, 0, 1], [2, 2, 1], [2, 2, 1]]).unsqueeze(0)
    ins_pr = torch.tensor([[1, 1, 1], [3, 3, 5], [3, 4, 2]]).unsqueeze(0)
    shape = (2, 1, 1)
    sem_gt =  sem_gt.repeat(shape)
    sem_gt[0, 2, :2] = 3
    ins_gt = ins_gt.repeat(shape)
    sem_pr = sem_pr.repeat(shape)
    ins_pr = ins_pr.repeat(shape)
    sem_pr[0] = torch.tensor([[1, 1, 1], [2, 2, 2], [0, 0, 1]])
    PQ = PanopticQuality()

    pqs = PQ.panoptic_quality_forward(sem_pr, sem_gt, ins_pr, ins_gt)
    print(pqs)


if __name__ == "__main__":
    main()