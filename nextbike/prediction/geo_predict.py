import geopy.distance as geodis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# TODO: Find another place for feature creation
# TODO: Add docstring
def quick_create_dist(p_df):
    uni = (49.452210, 11.079575)
    p_df['Dist_start'] = p_df.apply(lambda x: geodis.distance((x['Latitude_start'],
                                                               x['Longitude_start']), uni).km, axis=1)
    p_df['Dist_end'] = p_df.apply(lambda x: geodis.distance((x['Latitude_end'],
                                                             x['Longitude_end']), uni).km, axis=1)
    p_df['Direction'] = p_df['Dist_start'] > p_df['Dist_end']  # to uni: True, away: False
    return p_df


# TODO: very slow and only a first prototype
# TODO: Add docstring
def train_pred(p_df):
    p_df = quick_create_dist(p_df)

    X = p_df[['Duration', 'Dist_start']]
    y = p_df['Direction']
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print('Test score (mean accuracy)', clf.score(X_test, y_test))
