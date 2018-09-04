import csv
import re
import matplotlib.pyplot as plt
from matplotlib import pyplot 
import numpy as np 
import os
import itertools

def _read(log):
    with open(log) as f:
        lines = f.readlines()
    return lines

def _group_results(lines, groundtruth):
    regex = '\d:\w+|-\w+'
    samples = {}
    counter = 0
    for line in lines:
        m = re.search(regex, line)
        if m is None or '=' in line:
            continue
        else:
            data = line.split()
            sample = data[0]  # this also contains track id
            class_id = data[2]
            if len(data) == 6:
                predicted_label = data[3]
            else:
                predicted_label = class_id

            #inference_time = data[5]
            samples[counter] = {
                'sample': sample,
                'groundtruth': groundtruth,
                'prediction': predicted_label,

            }
            counter += 1

    return samples


def _pretty_print(metrics, ispostive=True):
    print("\n----------------------------------------")
    print("Ground Truth:\t{}".format(metric['groundtruth']))
    print("Total samples:\t{}".format(metric['Total']))
    if ispostive:
        print("True Positive :\t{}".format(metric['TP']))
        print("False Positive:\t{}".format(metric['FP']))
        print("Sensitivity:\t{}".format(metric['Sensitivity']))
        print("Precision:\t{}".format(metric['Precision']))
        print("The others: \t{}".format(metric['FN']))
    else:
        print("True Negative:\t{}".format(metric['TN']))
        print("False Negative:\t{}".format(metric['FN']))
        print("Specificity:\t{}".format(metric['Specificity']))
        print("Negative Predictive Value:\t{}".format(metric['NPV']))
        print("The others:\t{}".format(metric['FP']))
    print("----------------------------------------\n")


def plot_histogram(classes,title="Sensitivity Plot"):
    fig, ax = plt.subplots()
    ax.grid(zorder=5)
    plot = ax.bar(classes.keys(), classes.values(), color='b')
    ax.set_xlabel('Sensitivity (%)')
    ax.set_title(title)
    # fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join("D:\\viNet\\RnD\\plots",title+".jpg"))


def _removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def plot_multihistogram(x_label, labels, data):
    if len(data) != 2: 
        raise ValueError("Multiplot data exceeds x-lables. Only 2 allowed.")
    
    n_groups = 5

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.35

    opacity = 1.0
    error_config = {'ecolor': '0.3'}
    rects1 = ax.bar(index, data[0], bar_width,
                    alpha=opacity, color='b',
                     error_kw=error_config,
                    label=labels[0],zorder=3)

    rects2 = ax.bar(index + bar_width, data[1], bar_width,
                    alpha=opacity, color='r',
                   error_kw=error_config,
                    label=labels[1],zorder=3)

    ax.grid(zorder=3)
    ax.set_ylabel('Sensitivity (%)')
    ax.set_title('Sensitivity plot')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(x_label)
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = 5000
    for i, j in itertools.product(range(np.array(cm).shape[0]), range(np.array(cm).shape[1])):
        plt.text(j, i, format(cm[i][j], fmt),
                 horizontalalignment="center",
                 color="red" if cm[i][j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calculate_performance_metric_e3(log, gt, ispostive=True):
    class_predictions = {
        'Red-Kite': 0,
        'White-Tailed-Eagle': 0,
        'Buzzard': 0,
        'Other': 0,
        'Harrier': 0
    }
    classes = ['Red-Kite', 'White-Tailed-Eagle', 'Buzzard', 'Other', 'Harrier']

    results = _group_results(_read(log), gt)
    total_samples = len(results)
    TP = 0
    FP = 0
    metric = {}

    potential_predictions = _removekey(class_predictions, gt)
    classes.remove(gt)
    potential_classes = classes

    for index in range(total_samples):
        result = results[index]
        predicted = result['prediction']
        groundtruth = result['groundtruth']
        if groundtruth in predicted:
            TP += 1
        if groundtruth not in predicted:
            FP += 1
            for x in potential_classes:
                if x in predicted:
                    potential_predictions[x] = potential_predictions.get(
                        x, 0) + 1

    assert TP + FP == total_samples, "The sum of TP and FP doesn't equal total samples!"

    sensitivity = TP / total_samples
    precision = TP / (TP + FP)

    if ispostive:
        metric = {
            "Total": total_samples,
            "groundtruth": gt,
            'TP': TP,
            'FP': FP,
            'Sensitivity': sensitivity,
            'Precision': precision,
            'FN': potential_predictions
        }
    else:
        metric = {
            "Total": total_samples,
            "groundtruth": gt,
            'TN': TP,
            'FN': FP,
            'Specificity': sensitivity,
            'NPV': precision,
            'FP': potential_predictions
        }

    return metric, sensitivity

def calculate_performance_metric_avangrid(log, gt, ispostive=True):
    class_predictions = {
        'Bald-Eagle': 0,
        'Golden-Eagle': 0,
        'Hawk': 0,
        'Raven': 0,
        'Turkey-Vulture': 0
    }
    classes = ['Bald-Eagle', 'Golden-Eagle', 'Hawk', 'Raven', 'Turkey-Vulture']

    results = _group_results(_read(log), gt)
    total_samples = len(results)
    TP = 0
    FP = 0
    metric = {}

    potential_predictions = _removekey(class_predictions, gt)
    classes.remove(gt)
    potential_classes = classes

    for index in range(total_samples):
        result = results[index]
        predicted = result['prediction']
        groundtruth = result['groundtruth']
        if groundtruth in predicted:
            TP += 1
        if groundtruth not in predicted:
            FP += 1
            for x in potential_classes:
                if x in predicted:
                    potential_predictions[x] = potential_predictions.get(
                        x, 0) + 1

    assert TP + FP == total_samples, "The sum of TP and FP doesn't equal total samples!"

    sensitivity = TP / total_samples
    precision = TP / (TP + FP)

    if ispostive:
        metric = {
            "Total": total_samples,
            "groundtruth": gt,
            'TP': TP,
            'FP': FP,
            'Sensitivity': sensitivity,
            'Precision': precision,
            'FN': potential_predictions
        }
    else:
        metric = {
            "Total": total_samples,
            "groundtruth": gt,
            'TN': TP,
            'FN': FP,
            'Specificity': sensitivity,
            'NPV': precision,
            'FP': potential_predictions
        }

    return metric, sensitivity

if __name__ == '__main__':
    positive_classes = {"Red-Kite", "White-Tailed-Eagle"}
    # wte_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\e3_wte_128_300iter.log"
    # harrier_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\e3_harrier_128_300iter.log"
    # buzzard_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\e3_buzzard_128_300iter.log"
    redkite_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\E3_v2_redkite_138.log"
    # others_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\e3_other_128_300iter.log"

    be_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\be_128_300iter_adam.log"
    ge_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\ge_128_300iter_adam.log"
    hawk_log= "C:\\svn\\tools\\viNetProfiler\\x64\Release\hawk_128_300iter_adam.log"
    raven_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\raven_128_300iter_adam.log"
    tv_log = "C:\\svn\\tools\\viNetProfiler\\x64\Release\\tv_128_300iter_adam.log"

    e3_data = {
        # 'White-Tailed-Eagle': wte_log,
        # "Harrier": harrier_log,
        # "Buzzard": buzzard_log,
        "Red-Kite": redkite_log,
        # "Other": others_log
    }
    avangrid_totw_data = {
        'Bald-Eagle': be_log,
        "Golden-Eagle": ge_log,
        "Hawk": hawk_log,
        "Raven": raven_log,
        "Turkey-Vulture": tv_log
    }
    sensitivity_perclass_avangrid_totw = {
        'Bald-Eagle': 0,
        "Golden-Eagle": 0,
        "Turkey-Vulture": 0,
        "Hawk": 0,
        "Raven": 0,
    }

    sensitivity = 0
    for k, v in avangrid_totw_data.items():
        if k not in positive_classes:
            metric, sensitivity = calculate_performance_metric_avangrid(v, k, False)
            _pretty_print(metric, False)
            sensitivity_perclass_avangrid_totw[k] = round(sensitivity, 2)
        else:
            metric, sensitivity = calculate_performance_metric_avangrid(v, k)
            _pretty_print(metric)
            sensitivity_perclass_avangrid_totw[k] = round(sensitivity, 2)

    plot_histogram(sensitivity_perclass_avangrid_totw, "viNet_V.2.2.0_Avangrid_TOTW_ADAM Sensitivity (%)")

    """
    [
        [BE,GE,HAWK,RAVEN,TV]
    ]
    """
    class_names = ['BE','GE','HAWK','RAVEN','TV']
    cm = [
        [18285,5,575,355,0],
        [526,25847,54,171,238],
        [9,47,18340,14,50],
        [5,70,482,23293,42],
        [299,1305,28,82,18508]
    ]
    # plt.figure()
    # plot_confusion_matrix(cm, classes=class_names,
    #                   title='viNet_V.2.2.0_Avangrid_TOTW Confusion matrix, without normalization')
   
    # plt.savefig("D:\\viNet\\RnD\\plots\\viNet_V.2.2.0_Avangrid_TOTW Confusion Matrix.jpg")
    # plt.show()
    x_label = ['BE', 'GE','TV','Hawk','Raven']
    data = [[95.1,96.3,91.5,99.3,97.5],[95.8,97.4,96,93,94.5]]
    labels = ['V1-SGD','V2-ADAM-New']
    plot_multihistogram(x_label,labels,data)
