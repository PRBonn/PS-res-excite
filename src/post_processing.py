import torch


def get_instance(sem_seg, center_pred, embedding_pred):
    # rearrange embeddings
    embeddings = embedding_pred.squeeze().permute(1, 2, 0)

    # filter out prediction noise
    center_confidences = (center_pred >= 0.5)

    # filter out center of non-things classes
    semantic = torch.clone(sem_seg)
    semantic[sem_seg == 1.] = 0.
    semantic[sem_seg == 2.] = 0.
    center_confidences = center_confidences * (semantic > 0)

    # save coordinates of center candidates
    x_centers = (center_confidences > 0.).nonzero(as_tuple=True)[0]
    y_centers = (center_confidences > 0.).nonzero(as_tuple=True)[1]

    # assign instance id to each center candidate
    counter = 1
    center_mask = torch.zeros_like(sem_seg, dtype=int)
    for i in range(len(x_centers)):
        # if the center has not been already assigned to an instance
        if center_mask[x_centers[i], y_centers[i]] == 0:
            emb = embeddings[x_centers[i], y_centers[i]]  # compute embedding of center
            dist = torch.norm(embeddings - emb, dim=2)  # compute emb-distance between center and pixels
            mask = torch.zeros_like(center_mask, dtype=int)
            mask[dist < 0.5] = counter  # if they're close, same instance
            mask[dist > 0.5] = 0
            mask = mask * center_confidences  # keep centers only (not unique! still blobs)
            center_mask += mask
            counter += 1

    # check for having dense ids and not skip any int
    inst_labels = torch.unique(center_mask)
    for i, label in enumerate(inst_labels):
        center_mask[center_mask == label] = i

    # number of instance is the max id thanks to the lines above
    n_inst = torch.max(center_mask)
    center_confidences_new = torch.zeros_like(center_pred)
    instance_mask = torch.zeros_like(sem_seg)

    if n_inst > 0:
        unique_labels = torch.unique(center_mask)
        unique_labels = unique_labels[unique_labels > 0]

        # probability mask of the centers that we have up to now (remember: still blobs)
        center_confidences_refined = (center_mask > 0) * center_pred

        for label in unique_labels:
            # extracting the candidate centers/blobs of each instance
            center_help = (center_mask == label)
            # probabilities
            center_confidences_help = center_confidences_refined * center_help
            # coordinates of the highest-probability center candidate
            x_center = (center_confidences_help == torch.max(center_confidences_help)).nonzero(as_tuple=True)[0]
            y_center = (center_confidences_help == torch.max(center_confidences_help)).nonzero(as_tuple=True)[1]
            # save the "high-quality" center in the new mask
            center_confidences_new[x_center, y_center] = center_pred[x_center, y_center]

        # instance mask for the high quality centers only
        center_instance_mask = center_mask * (center_confidences_new > 0.)

        # coordinates of all HQ centers
        x_centers = (center_instance_mask > 0.).nonzero(as_tuple=True)[0]
        y_centers = (center_instance_mask > 0.).nonzero(as_tuple=True)[1]

        # dictionary with keys=instance-id, value=center-embedding
        comparison = {}
        for i in range(len(x_centers)):
            comparison[center_instance_mask[x_centers[i], y_centers[i]]] = embeddings[x_centers[i], y_centers[i]]

        # tensor of shape H x W x #CENTERS. Each channel is the embedding distance between all pixels and a center
        distances = torch.zeros((center_mask.shape[0], center_mask.shape[1], len(x_centers)))
        for key in comparison:
            # for all centers, extract the semantic class
            center_semantic = sem_seg[x_centers[key - 1], y_centers[key - 1]]
            # penalty if semantic class is not the same, or if it's unlabeled/stuff
            semantic_help = ((sem_seg - center_semantic) != 0) * 1e9
            semantic_help[sem_seg == 0] = 1e9
            semantic_help[sem_seg == 1] = 1e9
            semantic_help[sem_seg == 2] = 1e9
            # distance is the difference between all embeddings and the center embedding + penalty
            # if a point is not of the same semantic class as the center, they do NOT belong to the same instance
            distances_with_semantic = torch.norm(embeddings - comparison[key], dim=2) + semantic_help
            distances[:, :, key - 1] = distances_with_semantic

        # take the minimal distance for every pixel and the channel from which it comes from
        values, instance_mask = torch.min(distances, axis=2)
        instance_mask += 1

        # post-processing: if a pixel has an embedding that is far-away from all the others, it is not assigned to any
        # instance
        instance_mask[values > 0.5] = 0.

    return center_confidences_new, instance_mask, n_inst