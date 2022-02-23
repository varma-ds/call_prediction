import labels
import numpy as np
import torch
from sklearn.metrics import accuracy_score

label2id = {}
for i, cls in labels.id2label.items():
    label2id[cls] = i


def utt_reasoncode_accuracy(outputs, data):
    y_bar = torch.argmax(outputs['logits'], axis=1)
    y_true = data['targets']
    return accuracy_score(y_true, y_bar)


def utt_marked_accuracy(outputs, data):
    m_bar = torch.round(torch.sigmoid(outputs['marked']))
    m_true = data['marked']
    return accuracy_score(m_true, m_bar)


def call_reason_accuracy(outputs, data):
    y_true_list = []
    y_bar_list = []
    No_label_id = label2id['No Label']
    for i in range(data['targets'].shape[0]):
        y_true1 = np.where(data['targets'][i] == 1)[1].reshape(-1)
        y_bar1 = torch.argmax(outputs['finalreason_code'][i][:len(y_true1)],
                              axis=-1).reshape(-1).cpu().numpy()

        y_true2 = y_true1[y_true1 != No_label_id]

        if len(y_bar1[y_bar1 != No_label_id]) > 0:
            y_bar2 = y_bar1[y_bar1 != No_label_id][:1]
        else:
            y_bar2 = np.array([No_label_id])

        if len(y_true2) > 0:
            y_true_list.extend(y_true2)
            y_bar_list.extend(y_bar2)

    return accuracy_score(y_true_list, y_bar_list)


def marked_accuracy(outputs, data):
    m_true_list = []
    m_bar_list = []
    for i in range(data['targets'].shape[0]):
        m_true = torch.argmax(data['marked'], axis=-1)
        m_bar = torch.argmax(outputs['marked'].squeeze(-1), axis=-1).cpu().numpy()
        m_true_list.extend(m_true.tolist())
        m_bar_list.extend(m_bar.tolist())

    return accuracy_score(m_true_list, m_bar_list)


