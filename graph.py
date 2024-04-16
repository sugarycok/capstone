from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 백그라운드에서 Matplotlib 사용
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import os

plt.style.use('ggplot')

app = Flask(__name__, static_folder='static', template_folder='templates')

# CORS 헤더 추가
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # 모든 도메인에서 요청 허용
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'  # GET과 POST 요청 허용
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Content-Type 헤더 허용
    return response

# 홈 페이지 렌더링
@app.route('/')
def home():
    return render_template('index.html')

# 주가 예측 실행 및 결과 반환
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json  # JSON 형식으로 전송된 데이터를 읽음
        print(data)  # 데이터 확인용
        stock_code = data.get('stockCode')  # stockCode 키를 사용하여 주식 코드 추출
        if stock_code:
            result_file_path = run_stock_prediction(stock_code)
            return jsonify({'result_file': result_file_path})  # 그래프 파일 경로를 JSON 형식으로 반환
        else:
            return jsonify({'error': "Stock code is missing."})  # 오류 메시지를 JSON 형식으로 반환

# 주가 예측 함수...
def run_stock_prediction(stock_code):
    if stock_code.endswith('.NY'):
        stock_code = stock_code[:-3]
        df = fdr.DataReader(stock_code, exchange='NYSE')

    elif stock_code.endswith('.NSQ'):
        stock_code = stock_code[:-4]
        df = fdr.DataReader(stock_code, exchange='NASDAQ')

    else:
        df = fdr.DataReader(stock_code)

    ma5 = df['Close'].rolling(window=5).mean()
    ma3 = df['Close'].rolling(window=3).mean()

    df['MA5'] = ma5
    df['MA3'] = ma3

    file_name = stock_code + '.csv'
    plot_file_name = 'static/Result_' + stock_code + '.png'

    # 파일 저장 경로가 이미 존재하는지 확인하고 처리
    if os.path.exists(plot_file_name):
        # 이미 파일이 존재할 경우 기존 파일을 덮어쓰기
        os.remove(plot_file_name)

    df.to_csv(file_name, date_format='%Y-%m-%d', index=True)

    raw_df = pd.read_csv(file_name)

    raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)
    raw_df = raw_df.dropna()

    scaler = MinMaxScaler()
    scale_cols = ['Open', 'High', 'Low', 'Close', 'MA3', 'MA5', 'Volume']
    scaled_df = scaler.fit_transform(raw_df[scale_cols])
    scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

    feature_cols = ['MA3', 'MA5', 'Close']
    label_cols = ['Close']

    feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
    label_df = pd.DataFrame(scaled_df, columns=label_cols)

    feature_np = feature_df.to_numpy()
    label_np = label_df.to_numpy()

    def make_sequence_dataset(feature, label, window_size):
        feature_list = []
        label_list = []
        for i in range(len(feature) - window_size):
            feature_list.append(feature[i:i+window_size])
            label_list.append(label[i+window_size])
        return np.array(feature_list), np.array(label_list)

    window_size = 40
    X, Y = make_sequence_dataset(feature_np, label_np, window_size)

    split = -120
    x_train = X[:split]
    y_train = Y[:split]
    x_test = X[split:]
    y_test = Y[split:]

    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=100, batch_size=16,
              callbacks=[early_stop])

    pred = model.predict(x_test)

    y_test_inverse = y_test * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]
    pred_inverse = pred * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]

    future_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=30, freq='D')
    last_sequence = x_test[-1]

    future_predictions = []
    for i in range(30):
        future_pred = model.predict(last_sequence.reshape(1, window_size, len(feature_cols)))
        future_predictions.append(future_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, -1] = future_pred

    future_predictions_inverse = np.array(future_predictions) * (scaler.data_max_[3] - scaler.data_min_[3]) + scaler.data_min_[3]

    mape = np.sum(abs(y_test_inverse - pred_inverse) / y_test_inverse) / len(x_test)
    print("MAPE:", mape)

    plt.figure(figsize=(12, 6))
    plt.title('3MA + 5MA + Adj Close, window_size=40')
    plt.ylabel('Price')
    plt.xlabel('Date')

    num_test_samples = len(y_test)
    date_range = pd.date_range(end=datetime.now(), periods=num_test_samples, freq='D')
    plt.plot(date_range, y_test_inverse, label='actual')
    plt.plot(date_range, pred_inverse, label='prediction')

    plt.plot(future_dates, future_predictions_inverse, label='future prediction')

    plt.grid()
    plt.legend(loc='best')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.savefig(plot_file_name)
    plt.close()  # 그래프를 저장한 후에는 메모리에서 그래프를 해제합니다.

    return plot_file_name

if __name__ == '__main__':
    app.run(debug=True, port=8000)  # 포트 변경
