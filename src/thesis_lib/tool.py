import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def display_confusion_matrix(confusion_matrix, target_names=None, fname=None, normalization=True, title='Confusion matrix', cmap=plt.cm.OrRd):
    """
    Display the confusion matrix (either save to a file or show).
    
    Parameters
    ----------
    confusion matrix: 2D array (M * M)
    target_names: name of each element of the matrix. If it is None, it doesn't display the name.
    path: string containing a path to a filename. If it's None, it display the image.
    normalization: normalization of the confusion matrix to get value in [0, 1] for each line
    titel: title of the plot
    cmap: color map of the confusion matrix
    
    References
    ----------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    """
    if normalization:
        # Normalize the confusion matrix by row
        # (i.e by the number of samples in each class)
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusion_matrix
    print("Confusion matrix")
    print(cm)
    plt.figure()
    plt.title(title)
    plt.matshow(cm, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
