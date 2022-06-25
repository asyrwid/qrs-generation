import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from qrs.structures import ClusterQrsRecord, QrsRecord


def cluster_qrs_records(qrs_records: list[QrsRecord], denoised: bool = True) -> list[ClusterQrsRecord]:
    """
    Choose clustering method between clustering based on denoised signal QRSs
    or raw signal QRSs. This is done by setting denoised=True in the first case
    and denoised=False otherwise.
    """

    records = group_by_labels(qrs_records)
    clusters = []

    for grouped_qrs_records in records:
        signals = []
        for qrs_record in grouped_qrs_records:
            if denoised:
                qrs_signal = qrs_record.denoised_qrs_signal
            else:
                qrs_signal = qrs_record.qrs_signal

            signals.append(qrs_signal)

        # clustering
        if len(signals) < 2:  # no clustering when having single signal
            labels = np.array([1])
        else:
            labels = find_clusters(signals)

        n_clusters = labels.max()

        for it in range(n_clusters):
            cluster_qrs_records = []
            cluster_qrs_signals = []
            for idx, label in enumerate(labels):
                if label == it + 1:
                    cluster_qrs_records.append(grouped_qrs_records[idx])
                    cluster_qrs_signals.append(signals[idx])

            average_signal = np.sum(cluster_qrs_signals, axis=0)
            average_signal = average_signal / len(cluster_qrs_signals)
            cluster_record = ClusterQrsRecord(
                qrs_records=cluster_qrs_records,
                average_signal=average_signal
            )
            clusters.append(cluster_record)

    return clusters


def group_by_labels(qrs_records: list[QrsRecord]):

    unique_labels = np.unique([qrs.qrs_features.label for qrs in qrs_records])
    records = []
    for unique_label in unique_labels:
        same_label_records = []
        for qrs_record in qrs_records:
            if qrs_record.qrs_features.label == unique_label:
                same_label_records.append(qrs_record)
        records.append(same_label_records)

    return records


def find_clusters(signals: np.array):

    matrix_corrcoef = np.corrcoef(signals)
    np.fill_diagonal(matrix_corrcoef, 1)
    dissimilarity = 1 - matrix_corrcoef
    # force symmetry
    for it in range(len(dissimilarity)):
        for jt in range(it+1, len(dissimilarity)):
            dissimilarity[it][jt] = dissimilarity[jt][it]
    # generate linkage matrix
    hierarchy = linkage(squareform(dissimilarity), method='average')
    # employing the linkage matrix form flat clusters
    # for the chosen criterion 'distance':
    # t plays a role of 'distance treshold' for forming clusters
    labels = fcluster(hierarchy, t=0.25, criterion='distance')

    return labels
