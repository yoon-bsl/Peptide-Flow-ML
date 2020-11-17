from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from itertools import combinations
import pandas
import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def runSVM(dataFile, saveModel=False):
    dataset = pandas.read_csv(dataFile, header=0)
    dataFrame = pandas.DataFrame(dataset)
    columns = []
    for col in dataFrame.columns:
        columns.append(col)
    columns = columns[1:]

    y = dataFrame['Class']
    x = dataFrame.loc[:, dataFrame.columns != 'Class']
    x = x.values

    sscaler = StandardScaler()
    mscaler = MinMaxScaler()

    sscaler.fit(x)
    mscaler.fit(x)

    sx = sscaler.transform(x)
    mx = mscaler.transform(x)

    dependent = [x, sx, mx]
    dependentNames = ['None', 'StandardScaler', 'MinMaxScaler']

    maxScore = 0
    bestGamma = None
    bestC = None
    bestKernel = None
    bestX = None
    bestI = 0

    for i in range(0, 3):
        for kernel in ['rbf', 'linear', 'poly']:
            for c in [.001, .01, .1, 1, 10, 100]:
                for gamma in [.001, .01, .1, 1, 10, 100]:
                    print('Running model for {} kernel, {} c, and {} gamma'.format(kernel, str(c), str(gamma)))
                    model = SVC(kernel=kernel, C=c, gamma=gamma)
                    tempScore = cross_val_score(model, dependent[i], y).mean()

                    if tempScore > maxScore:
                        maxScore = tempScore
                        bestGamma = gamma
                        bestC = c
                        bestKernel = kernel
                        bestX = dependentNames[i]
                        bestI = i

    print('Best score: {}'.format(maxScore))
    print('Best gamma: {}'.format(bestGamma))
    print('Best C: {}'.format(bestC))
    print('Best Kernel Type: {}'.format(bestKernel))
    print('Best Scaler: {}'.format(bestX))

    bestModel = SVC(kernel=bestKernel, gamma=bestGamma, C=bestC)
    if saveModel:
        modelSVM = bestModel.fit(dependent[bestI], y)
        return modelSVM
    pred = cross_val_predict(bestModel, dependent[bestI], y)

    cm = confusion_matrix(y, pred)
    plt.figure()
    plot_confusion_matrix(cm, title='Sangsik Sample Confusion Matrix')
    plt.show()

    combos = list(combinations(columns, 2))
    plt.figure()
    plt.suptitle('Boundary Decision Function Shown in All Dimension Combinations (Scaler Used: {}'.format(dependentNames[bestI]))
    for i in range(0, len(combos)):
        print('Running combination: {}'.format(combos[i]))
        try:
            aa, bb = np.meshgrid(np.arange(dependent[bestI][:, int(combos[i][0]) - 1].min() - 0.01, dependent[bestI][:, int(combos[i][0]) - 1].max() + 0.01, .001),
                                np.arange(dependent[bestI][:, int(combos[i][1]) - 1].min() - 0.01, dependent[bestI][:, int(combos[i][1]) - 1].max() + 0.01, .001))
            modelSVM2D = bestModel.fit(dependent[bestI][:, [int(combos[i][0]) - 1, int(combos[i][1]) - 1]], y)
            dec = modelSVM2D.predict(np.c_[aa.ravel(), bb.ravel()])
        except MemoryError:
            aa, bb = np.meshgrid(np.arange(dependent[bestI][:, int(combos[i][0]) - 1].min() - 0.01, dependent[bestI][:, int(combos[i][0]) - 1].max() + 0.01, .01),
                            np.arange(dependent[bestI][:, int(combos[i][1]) - 1].min() - 0.01, dependent[bestI][:, int(combos[i][1]) - 1].max() + 0.01, .01))
            modelSVM2D = bestModel.fit(dependent[bestI][:, [int(combos[i][0]) - 1, int(combos[i][1]) - 1]], y)
            dec = modelSVM2D.predict(np.c_[aa.ravel(), bb.ravel()])
        if len(combos) == 6:
            plt.subplot(3, 2, i + 1)
        else:
            plt.subplot(7, 3, i + 1)
        plt.subplots_adjust(wspace=0.3, hspace=0.25)
        plt.contourf(aa, bb, dec.reshape(aa.shape),
                    cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(dependent[bestI][:, int(combos[i][0]) - 1], dependent[bestI][:, int(combos[i][1]) - 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('Feature {}'.format(combos[i][0]))
        plt.ylabel('Feature {}'.format(combos[i][1]))
        plt.xticks(())
        plt.yticks(())

    plt.show()

def predictFromModel(model):
    print('Input new data (BP, D2, D3, D4, D, RE, RI)')
    print('to allow for prediction from best model:')
    print('')
    bp = float(input('BP: '))
    d2 = float(input('D2: '))
    d3 = float(input('D3: '))
    d4 = float(input('D4: '))
    d = float(input('D: '))
    re = float(input('RE: '))
    ri = float(input('RI: '))
    print('The model predicts that this data is class:')
    print(model.predict([[bp, d2, d3, d4, d, re, ri]]))
    print('-------------------------------------------')

if __name__ == '__main__':
    bestModel = runSVM('Prima_postech(remove_I)&D_v2(addedNewControl).csv', saveModel=True)
    while True:
        predictFromModel(bestModel)
