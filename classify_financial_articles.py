from gravityai import gravityai as grav
import pickle 
import pandas as pd

model= pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb')) # TF-IDF: Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rb'))


def process(inPath, outPath): 
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize input data
    features = tfidf_vectorizer.transform(input_df['body'])
    # predict the classes 
    predictions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    #save results to csv 
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outPath, index=False)

grav.wait_for_requests(process)