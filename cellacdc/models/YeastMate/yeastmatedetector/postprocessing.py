import numpy as np
from scipy.optimize import linear_sum_assignment

def _match_scores(scores, composition, optional_object_score_threshold=0.25):
    
    # build score matrix:
    # rows -> objects, columns -> class (repeated if multiple instances can be present)
    match_mat = []
    for mask_class, (min_instances, max_instances) in composition.items():
        match_mat.append(scores.take([mask_class] * max_instances, axis=1))
    match_mat = np.concatenate(match_mat, axis=1)
    
    # find optimal match
    ri, ci = linear_sum_assignment(match_mat, True)
    
    possibilities_checked = 0    
    res_objs = []
    res_cls = []
    
    for mask_class, (min_instances, max_instances) in composition.items():
        
        # get all matchings to current class
        ri_slice = ri[(ci >= possibilities_checked) & (ci < possibilities_checked + max_instances)]
        ci_slice = ci[(ci >= possibilities_checked) & (ci < possibilities_checked + max_instances)]
        score_slice = match_mat[ri_slice, ci_slice]
        
        # not enough matches to satisfy minimal mumber of instances
        n_matches = len(score_slice)
        if n_matches < min_instances:
            return None
        
        # sort matches ascending by score
        # take at least min_instances + optional instances with score over threshold
        idxs = np.argsort(score_slice)[::-1]
        idxs = idxs[(np.arange(n_matches) < min_instances) |
                    (score_slice[idxs] >= optional_object_score_threshold)]
    
        res_objs.extend(ri_slice[idxs])
        res_cls.extend([mask_class] * len(idxs))
    
        possibilities_checked += n_matches
        
    return res_objs, res_cls


def _resolve_subobjects(parent_class, scores, possible_compositions,
                       optional_object_score_threshold=0.25, parent_override_thresh=2.0):
    
    # TODO: cleaner handling of edge cases
    
    if not int(parent_class) in possible_compositions:
        raise ValueError(f'must provide a possible subobject composition for parent class {parent_class}')
    
    # get matching for proposed parent class
    main_match = _match_scores(scores, possible_compositions[int(parent_class)], optional_object_score_threshold)
    if main_match is not None:
        res_objs, res_cls = main_match
        score_i = np.mean(scores[res_objs, res_cls])
        main_hypothesis = res_objs, res_cls, score_i
    else:
        main_hypothesis = None
        
    # check matchings for alternative parent classes
    alt_hypothesis = None    
    for cls, comp in possible_compositions.items():
        if cls == parent_class:
            continue
            
        match = _match_scores(scores, comp, optional_object_score_threshold)
        
        if match is None:
            continue
        
        res_objs, res_cls = match
        score_i = np.mean(scores[res_objs, res_cls])
        
        # accept an alternative parent class if it has a much higher score
        if main_hypothesis is None or score_i > main_hypothesis[2] * parent_override_thresh:
            # we have no alternative or this one is even better
            if alt_hypothesis is None or score_i > alt_hypothesis[2]:
                alt_hypothesis = res_objs, res_cls, score_i, cls
    
    if main_hypothesis is None and alt_hypothesis is None:
        return None
    
    final_parent_class = parent_class if alt_hypothesis is None else alt_hypothesis[3]
    subobject_assignment = ((main_hypothesis if alt_hypothesis is None else alt_hypothesis)[0],
                            (main_hypothesis if alt_hypothesis is None else alt_hypothesis)[1])
    
    return final_parent_class, subobject_assignment


def postproc_multimask(inst, possible_compositions,
                       optional_object_score_threshold=0.25, parent_override_thresh=2.0,
                       cls_offset = {1:1, 2:3}, score_thresholds = {0:0.9, 1:0.5, 2:0.5}, single_cell_mask_treshold=0.5):
    # TODO: remove cls_offset?
    
    # make sure keys in score_thresholds are int
    # (might be str if we read directly from JSON)
    score_thresholds = {int(k): v for k, v in score_thresholds.items()}

    boxes = list(inst.pred_boxes)
    boxes = [tuple(box.cpu().numpy()) for box in boxes]

    masks = list(inst.pred_masks)
    masks = [mask.cpu().numpy() for mask in masks]

    scores = list(inst.scores)
    scores = [float(score.cpu().numpy()) for score in scores]

    classes = list(inst.pred_classes)
    labels = [cls.cpu().numpy() for cls in classes]
            
    tensorboxes = np.stack(list(boxes))

    x_centroids = tensorboxes[:,0] + (tensorboxes[:,2] - tensorboxes[:,0])//2
    y_centroids = tensorboxes[:,1] + (tensorboxes[:,3] - tensorboxes[:,1])//2

    pts = np.stack([x_centroids, y_centroids], axis=1)

    detections = {}
    output_mask = np.zeros(masks[0].shape[1:], dtype=np.uint16)

    if len(boxes) == 0:
        return detections, output_mask

    # keep track of already assigned subobjects
    # -> we do not want to assign them to another compond object
    assigned_subobject_idxs = set()

    for n in range(len(boxes)):
        if labels[n] == 0 and scores[n] > score_thresholds[0]:

            x0, y0, x1, y1 = map(int, boxes[n])
            single_cell_mask_bin = masks[n][1][y0:y1, x0:x1] > single_cell_mask_treshold

            # do not add cells with empty mask after thresholding
            if single_cell_mask_bin.sum() == 0:
                continue

            output_mask[y0:y1, x0:x1][single_cell_mask_bin] = n+1
            
            detection = {'box': boxes[n], 'class': [str(0)], 'score': [scores[n]], 'id':str(n+1), 'links':[]}
            detections[str(n+1)] = detection

    for n in range(len(boxes)):
        if int(labels[n]) in possible_compositions:

            # score of compound object not high enough -> skip
            if scores[n] < score_thresholds[int(labels[n])]:
                continue

            detection = {'box': boxes[n], 'class': [str(int(labels[n]))], 'score': [scores[n]], 'links':[], 'id':str(n+1)}

            box_min = [boxes[n][0],boxes[n][1]]
            box_max = [boxes[n][2],boxes[n][3]]
                        
            inidx = np.nonzero(np.all(np.logical_and(box_min <= pts, pts <= box_max), axis=1))

            # ignore already assigned subobjects
            # inidx = np.array([idx for idx in inidx[0] if idx not in assigned_subobject_idxs])

            # NB: nonzero return size-1 tuple
            inidx = inidx[0]

            # collect mean mask scores for each class
            mask_scores = []
            accepted_box_idxs = []
            
            for j, boxidx in enumerate(inidx):
                if scores[boxidx] > score_thresholds[0] and labels[boxidx] == 0:

                    x0, y0, x1, y1 = map(int, boxes[boxidx])

                    multicellular_mask_slice = masks[n][:, y0:y1, x0:x1]
                    single_cell_mask_bin = masks[boxidx][1][y0:y1, x0:x1] > single_cell_mask_treshold

                    # ignore single cells with empty masks
                    # (would lead to NaNs, exceptions during LAP assignment)
                    if single_cell_mask_bin.sum() == 0:
                        continue

                    k = np.mean(multicellular_mask_slice[:, single_cell_mask_bin],axis=(1))
                    mask_scores.append(k)
                    accepted_box_idxs.append(j)
            
            if len(mask_scores) < 1:
                continue
                
            mask_scores = np.stack(mask_scores, axis=0)

            res = _resolve_subobjects(int(labels[n]), mask_scores, possible_compositions,
                                      optional_object_score_threshold=optional_object_score_threshold,
                                      parent_override_thresh=parent_override_thresh)

            if res is None:
                continue # TODO: handle
                
            final_parent_class, subobject_assignment = res

            # check compound object score again as class might have changed and we thus have different threshold
            if scores[n] < score_thresholds[int(final_parent_class)]:
                continue
            
            detection['class'] = [str(final_parent_class)]

            links = []
            for obj, cls in zip(*subobject_assignment):
                
                off = 0 if final_parent_class not in cls_offset else cls_offset[final_parent_class]

                sub_id = inidx[accepted_box_idxs[obj]]

                # mark to ignore this object for further assignments
                # as it is already in this compound object
                assigned_subobject_idxs.add(sub_id)

                class_idx = str(final_parent_class) + '.' + str(int(cls-off))

                # get existing single cell object
                sub_detection = detections[str(sub_id+1)]

                # add new class and score and link to compound object
                # NB: the score here is the score of parent compound object
                sub_detection['class'].append(class_idx)
                sub_detection['score'].append(scores[n])
                sub_detection['links'].append(str(n+1))

                detections[str(sub_id+1)] = sub_detection
                links.append(str(sub_id+1))

            detection['links'] = links
            detections[str(n+1)] = detection

    _cleanup_multiple_assignments(detections, possible_compositions, cls_offset)
    
    for v in detections.values():
        v['box'] = tuple(map(int, v['box']))
    
    return detections, output_mask

def _cleanup_multiple_assignments(detections, possible_compositions, cls_offset = {1:1, 2:3}):
    compound_objects_to_delete = set()
    for k in list(detections.keys()):
        detection = detections[k]

        for cls in set(detection['class']):

            classes_remaining = detection['class'][1:]
            scores_remaining = detection['score'][1:]
            links_remaining = detection['links']

            # get all occurrences of the class, sorted descending by score
            sorted_links = sorted(
                [(idx, (cl, sc, li)) for idx, (cl, sc, li) in enumerate(zip(classes_remaining, scores_remaining, links_remaining)) if cl == cls],
                key = lambda x: x[1][1],
                reverse=True
            )

            # remove all occurrences of the class except the first
            detection['class'] = [c for i,c in enumerate(detection['class']) if not i-1 in [idx for idx, (cl, sc, li) in sorted_links[1:]]]
            detection['score'] = [s for i,s in enumerate(detection['score']) if not i-1 in [idx for idx, (cl, sc, li) in sorted_links[1:]]]
            detection['links'] = [l for i,l in enumerate(detection['links']) if not i in [idx for idx, (cl, sc, li) in sorted_links[1:]]]
            
            for (idx, (cl, sc, li)) in sorted_links[1:]:

                # 1) remove k from parent li
                parent = detections[li]
                parent['links'].remove(k)

                # 2) check if parent still conforms to possible_comp
                children_classes = [detections[child]['class'][detections[child]['links'].index(li) + 1] for child in parent['links']]
                children_class_comp = [int(c.split('.')[1]) + cls_offset[int(parent['class'][0])] for c in children_classes]

                valid = True
                for c, (mi, ma) in possible_compositions[int(parent['class'][0])].items():
                    valid &= (children_class_comp.count(c) >= mi) & (children_class_comp.count(c) <= ma)

                # 3) remove parent and links in other children of necessary.
                if not valid:
                    for child in parent['links']:
                        link_idx = detections[child]['links'].index(li)
                        detections[child]['links'].remove(li)
                        detections[child]['class'].pop(link_idx + 1)
                        detections[child]['score'].pop(link_idx + 1)

                    compound_objects_to_delete.add(li)

    # delete compound objects marked for deletion
    for k in compound_objects_to_delete:
        detections.pop(k)
