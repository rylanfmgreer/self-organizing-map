import pandas as pd
import numpy as np
import seaborn as sns

def check_SOM(SOM, X, y):
    predictions = SOM.evaluate(X)
    x_pred = predictions[0]
    y_pred = predictions[1]
    results = pd.DataFrame(y)
    results.columns = ['truth']
    results['x'] = x_pred
    results['y'] = y_pred

    # jitter
    results['x'] += np.random.randn( len(results['x'])) / 10
    results['y'] += np.random.randn( len(results['y'])) / 10

    data_to_plot = results
    sns.scatterplot(data=data_to_plot, x='x', y='y', hue='truth')


def create_fake_data(n_classes=5, n_per_class=300, easiness=1):
    y = []
    X = np.random.randn(30, n_per_class * n_classes)

    for i in range(n_classes):
        a = i * n_per_class
        b = a + n_per_class
        X[:, a:b] += np.random.randn(30, 1) * easiness
        y += [i] * n_per_class

    return X, y