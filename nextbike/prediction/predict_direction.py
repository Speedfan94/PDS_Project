import geopy.distance as geodis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# TODO: Find another place for feature creation
def quick_create_dist(pDf):
    uni = (49.452210, 11.079575)
    pDf['Dist_start'] = pDf.apply(lambda x: geodis.distance((x['Latitude_start'], x['Longitude_start']), uni).km, axis=1)
    pDf['Dist_end'] = pDf.apply(lambda x: geodis.distance((x['Latitude_end'], x['Longitude_end']), uni).km, axis=1)
    pDf['Direction'] = pDf['Dist_start'] > pDf['Dist_end']  # to uni: True, away: False
    return pDf


#TODO: very slow and only a first prototype
def train_pred(df):
    df = quick_create_dist(df)

    X = df[['Duration', 'Dist_start']]
    y = df['Direction']
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print('Test score (mean accuracy)', clf.score(X_test, y_test))
