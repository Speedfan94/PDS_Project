from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# TODO: very slow and only a first prototype
# TODO: Add docstring
def train_pred(p_df):
    X = p_df[['Duration', 'Dist_start']]
    y = p_df['Direction']
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print('Test score (mean accuracy)', clf.score(X_test, y_test))
