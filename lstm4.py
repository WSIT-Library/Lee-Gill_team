import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시 설정
import os
import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# 모델 저장 경로 설정
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# 데이터 로드
data_file_path = '실내환경데이터.xlsx'
data = pd.read_excel(data_file_path)

# 필요한 열 선택
data = data[["년월일", "소리 값", "온도 (°C)", "습도 (%)", "미세먼지 (μg/m³)"]]

# 데이터 전처리
data['년월일'] = pd.to_datetime(data['년월일'])
data.set_index('년월일', inplace=True)

# 결측치 처리
data.fillna(data.mean(), inplace=True)

# 소음 수준 및 미세먼지 상태 계산 함수
def classify_sound_level(sound_value):
    if sound_value < 200:
        return "조용함"
    elif sound_value < 500:
        return "일반적인 소음"
    elif sound_value < 800:
        return "큰 소음"
    else:
        return "매우 큰 소음"

def classify_dust_level(dust_value):
    if dust_value < 16:
        return "좋음"
    elif dust_value < 36:
        return "보통"
    elif dust_value < 76:
        return "나쁨"
    else:
        return "매우 나쁨"

# 소음 수준과 미세먼지 상태 추가
data['소리 강도'] = data['소리 값'].apply(classify_sound_level)
data['미세먼지 상태'] = data['미세먼지 (μg/m³)'].apply(classify_dust_level)

# 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["온도 (°C)", "습도 (%)"]])

# 시퀀스 생성 함수
def create_dataset(data, time_step=1):
    X, y_temp, y_hum = [], [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y_temp.append(data[i + time_step, 0])  # 온도
        y_hum.append(data[i + time_step, 1])   # 습도
    return np.array(X), np.array(y_temp), np.array(y_hum)

# 10초 단위로 1시간(360개) 데이터로 나누기
time_step = 360  # 1시간의 데이터 수
X, y_temp, y_hum = create_dataset(scaled_data, time_step)

# 데이터 분할 (훈련, 테스트)
train_size = int(len(X) * 0.8)
if train_size == 0 or train_size >= len(X):
    raise ValueError("훈련 데이터 크기가 유효하지 않습니다.")

X_train, X_test = X[:train_size], X[train_size:]
y_temp_train, y_temp_test = y_temp[:train_size], y_temp[train_size:]
y_hum_train, y_hum_test = y_hum[:train_size], y_hum[train_size:]

# LSTM 모델 정의
def build_lstm_model(units=100, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # 온도 또는 습도만 예측
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 모델 파일 경로
model_file_temp = os.path.join(model_dir, 'temperature_model.keras')
model_file_hum = os.path.join(model_dir, 'humidity_model.keras')

# 기존 모델 로드 또는 새로 생성
if os.path.exists(model_file_temp) and os.path.exists(model_file_hum):
    temperature_model = load_model(model_file_temp)
    humidity_model = load_model(model_file_hum)
    print("기존 모델을 로드했습니다.")
else:
    # 모델을 새로 생성 및 학습
    temperature_model = build_lstm_model()
    humidity_model = build_lstm_model()
    print("새 모델을 생성했습니다.")

    # 모델 학습
    temperature_model.fit(X_train, y_temp_train, epochs=100, batch_size=32, verbose=1)
    humidity_model.fit(X_train, y_hum_train, epochs=100, batch_size=32, verbose=1)

    # 모델 저장
    temperature_model.save(model_file_temp)
    humidity_model.save(model_file_hum)
    print("모델이 저장되었습니다.")

# 앙상블 예측 함수
def ensemble_predict(models, X):
    predictions = np.zeros((X.shape[0], len(models)))  # 2차원 배열로 선언
    for i, model in enumerate(models):
        # 예측 결과를 1차원 배열로 변환하여 저장
        predictions[:, i] = model.predict(X).flatten()  # flatten을 사용하여 1차원으로 변환
    return np.mean(predictions, axis=1)  # 평균을 내어 1차원 배열 반환

# 앙상블 예측
temp_train_pred = ensemble_predict([temperature_model], X_train)
hum_train_pred = ensemble_predict([humidity_model], X_train)

temp_test_pred = ensemble_predict([temperature_model], X_test)
hum_test_pred = ensemble_predict([humidity_model], X_test)

# RMSE 계산
train_temp_rmse = np.sqrt(mean_squared_error(y_temp_train, temp_train_pred))
train_hum_rmse = np.sqrt(mean_squared_error(y_hum_train, hum_train_pred))
test_temp_rmse = np.sqrt(mean_squared_error(y_temp_test, temp_test_pred))
test_hum_rmse = np.sqrt(mean_squared_error(y_hum_test, hum_test_pred))

print(f"훈련 데이터 RMSE (온도): {train_temp_rmse:.2f}")
print(f"훈련 데이터 RMSE (습도): {train_hum_rmse:.2f}")
print(f"테스트 데이터 RMSE (온도): {test_temp_rmse:.2f}")
print(f"테스트 데이터 RMSE (습도): {test_hum_rmse:.2f}")

# 30분 후 예측을 위한 데이터 준비
def predict_future_values(model_temp, model_hum, last_data, steps=180):
    future_temp = []
    future_hum = []

    current_data = last_data

    for _ in range(steps):
        # 예측
        temp_pred = model_temp.predict(current_data.reshape(1, time_step, 2))
        hum_pred = model_hum.predict(current_data.reshape(1, time_step, 2))

        future_temp.append(temp_pred[0, 0])
        future_hum.append(hum_pred[0, 0])

        # 현재 데이터 업데이트 (예측된 값을 포함)
        # 차원을 맞춰서 append
        current_data = np.append(current_data[1:], [[temp_pred[0, 0], hum_pred[0, 0]]], axis=0)

    return np.array(future_temp), np.array(future_hum)

# 마지막 훈련 데이터를 가져와서 30분 후 예측
if len(scaled_data) >= time_step:
    last_data = scaled_data[-time_step:]
    future_temperatures, future_humidities = predict_future_values(temperature_model, humidity_model, last_data)

    # 예측 결과 역정규화
    future_temperatures = scaler.inverse_transform(np.hstack((future_temperatures.reshape(-1, 1), np.zeros((future_temperatures.shape[0], 1)))))
    future_humidities = scaler.inverse_transform(np.hstack((np.zeros((future_humidities.shape[0], 1)), future_humidities.reshape(-1, 1)))) 

    # 결과 출력
    future_results = pd.DataFrame({
        '예측 온도': future_temperatures[:, 0],
        '예측 습도': future_humidities[:, 1]
    })

    # 기존 데이터에 앙상블 예측 결과 추가
    data['예측 온도'] = np.nan
    data['예측 습도'] = np.nan

    # 예측 결과를 기존 데이터에 추가
    data.loc[data.index[-len(future_temperatures):], '예측 온도'] = future_temperatures[:, 0]
    data.loc[data.index[-len(future_humidities):], '예측 습도'] = future_humidities[:, 1]

    # 수정된 데이터를 새로운 엑셀 파일로 저장
    data.to_excel('업데이트된_실내환경데이터.xlsx')
    print("앙상블 예측 결과가 '업데이트된_실내환경데이터.xlsx'에 저장되었습니다.")

    # 예측 결과 시각화
    plt.figure(figsize=(14, 6))

    # 마지막 1시간의 실제 온도 데이터
    last_hour_index = data.index[-(360 + len(future_temperatures)):]  # 마지막 1시간 데이터 + 예측 길이
    plt.plot(last_hour_index[:360], data['온도 (°C)'].iloc[-360:], label='실제 온도', color='red', linestyle='--')

    # 예측 온도
    plt.plot(data.index[-len(future_temperatures):], future_temperatures[:, 0], label='예측 온도', color='orange')

    # 마지막 1시간의 실제 습도 데이터
    plt.plot(last_hour_index[:360], data['습도 (%)'].iloc[-360:], label='실제 습도', color='green', linestyle='--')

    # 예측 습도
    plt.plot(data.index[-len(future_humidities):], future_humidities[:, 1], label='예측 습도', color='blue')

    # 그래프 제목과 레이블
    plt.title('미래 예측 온도 및 습도')
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.grid()

    plt.show()

else:
    print("데이터가 부족하여 예측을 수행할 수 없습니다.")
