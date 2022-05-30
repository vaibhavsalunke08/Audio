import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import streamlit as st

from librosa import display
from librosa.feature.spectral import mfcc
from sklearn.model_selection import train_test_split



# header of the page

html_temp = """
    <div style ="background-color:powderblue;padding:13px">
    <h1 style ="color:black;text-align:center;">Predicting Hit Songs Using Repeated Chorus </h1>
    </div>
    """
      
st.markdown(html_temp, unsafe_allow_html = True)



    

# loading in the model to predict on the data
#pickle_in = open('C:/Users/Admin/Desktop/Intern/class.pkl', 'rb')
#classifier = pickle.load(pickle_in)


###################################################          Loading the Final Data   #####################################################

# Importing the dataset
data=pd.read_csv("C:/Users/Admin/Desktop/Intern/Final Data.csv")
df=pd.DataFrame(data)




df1=df.iloc[:,5:] # contains all numeric columns



# ### Outlier Removal



feature = df1 # one or more.
for cols in feature:
    Q1 = df1[cols].quantile(0.25)
    Q3 = df1[cols].quantile(0.75)
    IQR = Q3 - Q1
    dff = df1[((df1[cols] < (Q1 - 1.5 * IQR)) |(df1[cols] > (Q3 + 1.5 * IQR)))]
f=dff.index
new_df = df.iloc[f]





x = new_df.iloc[:,5:]# feature columns
y = new_df['Label'] # target column




# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=0)
print(" Training data: ",X_train.shape)
print(" Testing data:  ",X_test.shape)


# ### Feature Selection



from sklearn.feature_selection import mutual_info_classif
mutual_info=mutual_info_classif(X_train,y_train)
mutual_info=pd.Series(mutual_info)
mutual_info.index=X_train.columns
s=mutual_info.sort_values(ascending=False)[:25]
s.plot.bar(figsize=(10,6))
f=s.index




x1 = new_df[f]
y1 = new_df["Label"]



# separate dataset into train and test
X1_train, X1_test, y1_train, y1_test = train_test_split(
    x1,
    y1,
    test_size=0.3,
    random_state=0)
print(" Training data: ",X1_train.shape)
print(" Testing data:  ",X1_test.shape)




from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X1_train, y_train)



# separate dataset into train and test
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0)
print(" Training data: ",X2_train.shape)
print(" Testing data:  ",X2_test.shape)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_norm1 = scaler.fit_transform(X2_train)
X_test_norm1 = scaler.transform(X2_test)


###########################################                     Model Development                  ########################################



#### Random Forest Classifier

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
# creating a RF classifier
clf = RandomForestClassifier() 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_norm1, y2_train)
# performing predictions on the test dataset
y_pred2 = clf.predict(X_test_norm1)
y_pred3 = clf.predict(X_train_norm1)
score = metrics.accuracy_score(y2_test, y_pred2)*100
print("Test accuracy:", round(score,2))
score = metrics.accuracy_score(y2_train, y_pred3)*100
print("Training accuracy:", round(score,2))



#### Cross validating Random Forest


#evaluating the model using repeated k-fold cross-validation
from numpy import mean
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

#prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=3, n_repeats=7, random_state=1)
# evaluate model
scores = cross_val_score(clf, X_train_norm1, y2_train, scoring='accuracy', cv=cv, n_jobs=-1)
#report performance
print('Training Accuracy:',round(mean(scores),2)*100)





######################################                  Feature Extraction                              ###################################

data=[]



import librosa
import numpy as np
from scipy.stats import skew , kurtosis

def statistics(list, feature, columns_name, data):
    i = 0
    for ele in list:
        _skew = skew(ele)
        columns_name.append(f'{feature}_kew_{i}')
        min = np.min(ele)
        columns_name.append(f'{feature}_min_{i}')
        max = np.max(ele)
        columns_name.append(f'{feature}_max_{i}')
        std = np.std(ele)
        columns_name.append(f'{feature}_std_{i}')
        mean = np.mean(ele)
        columns_name.append(f'{feature}_mean_{i}')
        median = np.median(ele)
        columns_name.append(f'{feature}_median_{i}')
        _kurtosis = kurtosis(ele)
        columns_name.append(f'{feature}_kurtosis_{i}')

        i += 1
        data.append(_skew)
        data.append(min)
        data.append(max)
        data.append(std)
        data.append(mean)
        data.append(median)
        data.append(_kurtosis)
    return data




def extract_features(audio_path, title):

  
    columns_name = ['title']
    data.append(title)
    x , sr = librosa.load(audio_path)

    chroma_stft = librosa.feature.chroma_stft(x, sr)
    stft = statistics(chroma_stft, 'chroma_stft', columns_name, data)
    

    chroma_cqt = librosa.feature.chroma_cqt(x, sr)
    cqt = statistics(chroma_cqt, 'chroma_cqt', columns_name, data)
    

    chroma_cens = librosa.feature.chroma_cens(x, sr)
    cens = statistics(chroma_cens, 'chroma_cens', columns_name, data)
    

    mfcc = librosa.feature.mfcc(x, sr)
    mf = statistics(mfcc, 'mfcc', columns_name, data)
    

    rms = librosa.feature.rms(x, sr)
    rm = statistics(rms, 'rms', columns_name, data)
    

    spectral_centroid = librosa.feature.spectral_centroid(x, sr)
    centroid = statistics(spectral_centroid, 'spectral_centroid', columns_name, data)
    

    spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr)
    bandwidth = statistics(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)
    

    spectral_contrast = librosa.feature.spectral_contrast(x, sr)
    contrast = statistics(spectral_contrast, 'spectral_contrast', columns_name, data)
    

    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr)
    rolloff = statistics(spectral_rolloff, 'spectral_rolloff', columns_name, data)
    

    tonnetz = librosa.feature.tonnetz(x, sr)
    tonnetz = statistics(tonnetz, 'tonnetz', columns_name, data)
    

    zero_crossing_rate = librosa.feature.zero_crossing_rate(x, sr)
    zero = statistics(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)
   

    return data , columns_name






#####################################################                User Interface              ##########################################


# user input file

file = st.file_uploader("Upload Audio Files", type=["wav"])


# button modify

st.markdown("""
<style>
div.stButton > button:first-child {
    border-radius: 20%;
    height: 4em;
    width: 8em; 
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    border-radius: 20%;
    height: 4em;
    width: 8em; 
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)


# button alignment

col1, col2, col3 = st.columns([1,0.5,1])

col4, col5, col6 = st.columns([1,0.5,1])



if col2.button("PREDICT"):
    data = []
    columns_name = []
    ca = [file.name] # considering the name of file

    for i , chorus in enumerate(ca):
        data , columns_name = extract_features(f"C:/Users/Admin/Desktop/Intern/{chorus}", chorus)

    
    # combining list of row values (data) and list of columns (columns_name)
    nnn = []
    for i in range(0, len(data), len(columns_name)):
        nnn.append(data[i:i + 519])

    # creating dataframe
    df2 = pd.DataFrame(nnn, columns=columns_name)
    df2 = df2.drop(["title"],axis=1)
    x2 = df2[f]

    # transforming the data
    d1 =np.array(x2)
    data1 = scaler.transform(d1)
    hit_prediction = clf.predict(data1)


    if hit_prediction == 1:
        st.write("The predicted class is popular")

    elif hit_prediction == 0:
        st.write("The predicted class is unpopular")
