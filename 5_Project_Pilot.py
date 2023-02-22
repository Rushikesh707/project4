
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)






import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clean Data
music_data = pd.read_csv(r'D:\1_AJINKYA_PAWAR\3_SOFTWARE JOB\1_CDAC\1_CCEE_PLACEMENT\12_Project\4_Spotify.csv')
music_data.dropna(inplace=True)
music_data.drop_duplicates(inplace=True)

X = music_data.drop(columns = ["track_id", "artists", "album_name", "track_name", "track_genre"])
y = music_data["track_genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)

