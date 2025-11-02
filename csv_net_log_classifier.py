
import pandas as pd
from pandas import DataFrame
from pycaret.classification import *

def pre_process_data(df_in: DataFrame)-> DataFrame:

    #filter out unwanted columns

    wanted_columns = ['dur',
 'proto',
 'service',
 'state',
 'spkts',
 'dpkts',
 'sbytes',
 'dbytes',
 'rate',
 'sttl',
 'dttl',
 'sload',
 'dload',
 'sloss',
 'dloss',
 'sinpkt',
 'dinpkt',
 'sjit',
 'djit',
 'swin',
 'stcpb',
 'dtcpb',
 'dwin',
 'tcprtt',
 'synack',
 'ackdat',
 'smean',
 'dmean',
 'trans_depth',
 'response_body_len',
 'ct_srv_src',
 'ct_state_ttl',
 'ct_dst_ltm',
 'ct_src_dport_ltm',
 'ct_dst_sport_ltm',
 'ct_dst_src_ltm',
 'is_ftp_login',
 'ct_ftp_cmd',
 'ct_flw_http_mthd',
 'ct_src_ltm',
 'ct_srv_dst',
 'is_sm_ips_ports']

    df_in = df_in[wanted_columns]

    return df_in.copy()


def get_predictions(model, df_testing_data:DataFrame, required_accuracy:float=0.75) -> DataFrame:

    #required_accuracy is minimum prediction accuracy to return, default = 75%
    predictions_out = predict_model(model, data=df_testing_data)
    predictions_out = predictions_out[predictions_out.prediction_score >= 0.75] #only use entries with a pridiction score of atleast 75%
    return predictions_out


def get_predictions_dictionary(df_prediction:DataFrame):
    predictions_slim = df_prediction.prediction_label.value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    results = [{i:y} for i,y in predictions_slim.items()]
    out = {}
    for i in results:

        for j in i:
            out[j] = i[j]

    return out

def get_classification_results(raw_test_data:DataFrame)->dict:
    model = load_model('netlog_RFC_classification_model')
    df_predict = pre_process_data(df_in=raw_test_data)
    predictions = get_predictions(model, df_predict)
    predictions_dict = get_predictions_dictionary(predictions)
    return predictions_dict

def classifyUploadedFile(uploaded_file)->dict:
    df_in = pd.read_csv(uploaded_file)
    df_in = pre_process_data(df_in)
    classification_results = get_classification_results(df_in)
    return classification_results


def classify_csv_file(csv_df:DataFrame, model_path:str)->DataFrame:
    model = load_model('classifier/netlog_pycaret_classification_model')
    csv_df = pre_process_data(df_in=csv_df)
    csv_df = get_predictions(model, csv_df)
    csv_df = csv_df.rename(columns={'prediction_label': 'attack_cat'})
    return csv_df


if __name__ == '__main__':
    print('Running as main')



