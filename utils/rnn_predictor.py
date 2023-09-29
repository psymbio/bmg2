import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
from utils.dataframes import dfs_tabs
from sklearn.metrics import r2_score

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class LSTMClassifierBidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
def save_model_and_see_stats(model, model_file_name, X_test, y_test, max_factor):
    """
    model : pytroch model that has been trained
    model_file_name : string "rnn_model_method_x" without extension
    X_test : pytorch tensor
    y_test : pytorch tensor
    max_factor : 
    """
    torch.save(model.state_dict(), model_file_name + ".pt")
    model.eval()
    with torch.no_grad():
        predicted_labels = model(X_test)
    r2 = r2_score(np.array(y_test)*max_factor, np.array(predicted_labels).reshape(-1)*max_factor)
    print(f"R2 Score: {r2:.4f}")
    sns.regplot(x=np.array(y_test)*max_factor, y=np.array(predicted_labels).reshape(-1)*max_factor)
    
def save_predictions(alloy_to_vectorized_tensor, model, max_factor, df_train, df_test, output_file_name):
    """
    alloy_to_vectorized_tensor : function
        converts the string alloy to a vectorized tensor
    model : trained pytorch model
    max_factor : training_set max factor
    df_train : pd.DataFrame()
    df_test : pd.DataFrame()
    output_file_name : string without extension
        was "dataset_rnn"
    """
    all_rnn_train_output = []
    for i in range(df_train.shape[0]):
        alloy_tensor = alloy_to_vectorized_tensor(df_train.loc[i, "bmg_alloy"])
        output = model(alloy_tensor.unsqueeze(0))
        all_rnn_train_output.append(output.squeeze().item() * max_factor)

    all_rnn_test_output = []
    for i in range(df_test.shape[0]):
        alloy_tensor =  alloy_to_vectorized_tensor(df_test.loc[i, "bmg_alloy"])
        output = model(alloy_tensor.unsqueeze(0))
        all_rnn_test_output.append(output.squeeze().item() * max_factor)

    new_train = df_train
    new_test = df_test

    new_train["rnn_encoding"] = all_rnn_train_output
    new_test["rnn_encoding"] = all_rnn_test_output

    dfs = [new_train, new_test]
    sheets = ["train", "test"]
    dfs_tabs(dfs, sheets, output_file_name + ".xlsx")
    