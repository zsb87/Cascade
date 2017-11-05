from wardmetrics.core_methods import eval_segments
from wardmetrics.core_methods import eval_events
from wardmetrics.visualisations import *
from wardmetrics.utils import *
from datetime import datetime, timedelta
from intervaltree import Interval, IntervalTree


def get_overlap(a, b):
    '''
    Given two pairs a and b,
    find the intersection between them
    could be either number or datetime objects
    '''

    tmp = min(a[1], b[1]) - max(a[0], b[0])
    
    if isinstance(tmp, timedelta):
        zero_value = timedelta(seconds=0)
    else:
        zero_value = 0
    
    return max(zero_value, tmp)



def get_union(a, b):
    '''
    Given two pairs a and b,
    assume a and b have overlap
    find the union between them
    could be either number or datetime objects
    '''

    tmp = max(a[1], b[1]) - min(a[0], b[0])
    # if a and b have no overlap
    # todo
    return tmp


def get_overlap_by_union_ratio(gt, pred):
    one_ground_truth_test = get_index_intersect_interval(gt, pred)
    if one_ground_truth_test == []:
        return 0
    one_ground_truth_test = [one_ground_truth_test[0][0] , one_ground_truth_test[0][1]]
    return float(get_overlap(one_ground_truth_test, pred[0]))/float(get_union(one_ground_truth_test, pred[0]))


def get_index_intersect_interval(gt, pred):
    '''
    Efficient algorithm to find which intervals intersect

    Handles both unix timestamp or datetime object

    Return:
    -------

    prediction_gt: 
        array with same size as prediction,
        will be 1 if there's an overlapping label
        0 if not
    recall:
        recall percentage of labels
    overlap:
        how much overlap between label and prediction
    '''

    # gt = kwargs['groundtruth']
    # pred = kwargs['prediction']

    total_overlap = None
    missed = None
    false_alarm = None

    # calculate recall
    tree = IntervalTree()
    for segment in pred:
        tree.add(Interval(segment[0],segment[1]))

    TP = 0
    for segment in gt:
        overlap = tree.search(segment[0], segment[1])

        if len(overlap) != 0:
            TP += 1

    recall = TP/len(gt)

    # calculate precision
    tree = IntervalTree()
    for segment in gt:
        tree.add(Interval(segment[0],segment[1]))

    prediction_gt = []

    segment = pred[0]
    overlap = tree.search(segment[0], segment[1])

    return list(overlap)




if __name__ == "__main__":

    # ground_truth_test = [(1,4),(12,13)]
    # detection_test = [(1,3)]

    ground_truth_test = [[199, 241], [530, 639], [1271, 1334], [1442, 1507], [1536, 1569], [1673, 1773], [1862, 1904], [1942, 1976], [2082, 2156], [2361, 2556], [2583, 2651], [2823, 2962], [7678, 7705], [7718, 7756], [7817, 7912], [8043, 8154], [8264, 8318], [8424, 8526], [8605, 8659], [8675, 8708], [8879, 8958], [9160, 9219], [9338, 9424], [9515, 9563], [10060, 10105], [10212, 10248], [10444, 10506], [11475, 11510], [11560, 11606], [11778, 11865], [12019, 12096], [12313, 12360], [12562, 12631], [13013, 13087], [13346, 13393], [13484, 13566], [13736, 13784], [13944, 13999], [14178, 14224], [14310, 14347], [14438, 14480], [14627, 14669], [14767, 14795], [14995, 15021], [15192, 15232], [15283, 15317], [15354, 15384], [15415, 15446]]
    detection_test = [(200, 240)]
    # get_index_intersect_interval(ground_truth_test, detection_test)

    print(get_overlap_by_union_ratio(ground_truth_test, detection_test))
