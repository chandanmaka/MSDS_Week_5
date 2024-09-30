import pandas as pd
from pycaret.classification import predict_model, load_model

model = load_model('trained_model')

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    print('Data loaded successfully!!')
    return df


def make_predictions(df, threshold=0.5):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    Rounds up to 1 if greater than or equal to the threshold.
    """
    predictions = predict_model(model, data=df)
    # predictions['Churn_prediction'] = (predictions['Score'] >= threshold)
    # predictions['Churn_prediction'].replace({True: 'Churn', False: 'No churn'}, inplace=True)
    predictions['prediction_label'].replace({1: 'Churn', 0: 'No churn'}, inplace=True)
    drop_cols = predictions.columns.tolist()
    drop_cols.remove('prediction_label')
    # drop_cols.remove('Churn_prediction')
    return predictions.drop(drop_cols, axis=1)


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)