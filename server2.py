import time
import atexit
import signal
import keyboard  # 추가된 라이브러리
from flask import Flask, render_template, send_file, jsonify
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os
import sys
from io import BytesIO

app = Flask(__name__)

# 엑셀 파일 경로
data_file = '실내환경데이터.xlsx'
prediction_file_path = '예측된_온도_습도.xlsx'

# 종료 시 ngrok 종료 함수 등록
atexit.register(lambda: None)  # ngrok 관련 종료 함수 제거

# Shift + ESC 키를 눌렀을 때 Flask 서버 종료
def on_shift_esc_pressed(event):
    print("Shift + ESC pressed, shutting down Flask server...")
    # 서버 종료를 위한 예외 발생
    raise SystemExit

# Shift + ESC 키에 대한 이벤트 리스너 등록
keyboard.add_hotkey('shift+esc', on_shift_esc_pressed)

# 데이터 파일 유효성 검사 함수
def validate_data_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 파일을 찾을 수 없습니다.")
    df = pd.read_excel(file_path)
    if df.empty:
        raise ValueError(f"{file_path} 파일이 비어 있습니다.")
    return df

# 온도, 습도 그래프 생성 함수
def generate_temp_hum_graph():
    df = validate_data_file(data_file)

    fig = go.Figure()

    # 실제 온도
    fig.add_trace(go.Scatter(
        x=df.index[-360:], 
        y=df['온도 (°C)'].iloc[-360:], 
        mode='lines', 
        name='실제 온도', 
        line=dict(color='red', dash='dash')
    ))

    # 실제 습도
    fig.add_trace(go.Scatter(
        x=df.index[-360:], 
        y=df['습도 (%)'].iloc[-360:], 
        mode='lines', 
        name='실제 습도', 
        line=dict(color='green', dash='dash')
    ))

    # 예측 온도
    if os.path.exists(prediction_file_path):
        pred_df = pd.read_excel(prediction_file_path)
        fig.add_trace(go.Scatter(
            x=pred_df['예측 시간'],
            y=pred_df['예측 온도'],
            mode='lines',
            name='예측 온도',
            line=dict(color='orange')
        ))

        # 예측 습도
        fig.add_trace(go.Scatter(
            x=pred_df['예측 시간'],
            y=pred_df['예측 습도'],
            mode='lines',
            name='예측 습도',
            line=dict(color='blue')
        ))

    fig.update_layout(
        title='미래 예측 온도 및 습도',
        xaxis_title='시간',
        yaxis_title='값',
        xaxis=dict(tickformat='%H:%M'),  # 소수점 초를 없애고 시:분 형식으로 표시
        template='plotly_white',
        dragmode='zoom'  # 팬 기능을 활성화
    )

    img = BytesIO()
    try:
        pio.write_image(fig, img, format='png')
    except Exception as e:
        print(f"Error generating image: {e}")
        raise
    img.seek(0)

    return img

# 예측 온도 및 습도 그래프 생성 함수
def generate_prediction_graph():
    pred_df = validate_data_file(prediction_file_path)

    fig = go.Figure()

    # 예측 온도
    fig.add_trace(go.Scatter(
        x=pred_df['예측 시간'],
        y=pred_df['예측 온도'],
        mode='lines',
        name='예측 온도',
        line=dict(color='orange')
    ))

    # 예측 습도
    fig.add_trace(go.Scatter(
        x=pred_df['예측 시간'],
        y=pred_df['예측 습도'],
        mode='lines',
        name='예측 습도',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='미래 예측 온도 및 습도',
        xaxis_title='시간',
        yaxis_title='값',
        xaxis=dict(tickformat='%H:%M'),  # 소수점 초를 없애고 시:분 형식으로 표시
        template='plotly_white',
        dragmode='zoom'  # 팬 기능을 활성화
    )

    img = BytesIO()
    try:
        pio.write_image(fig, img, format='png')
    except Exception as e:
        print(f"Error generating image: {e}")
        raise
    img.seek(0)

    return img

# 미세먼지 그래프 생성 함수
def generate_dust_graph():
    df = validate_data_file(data_file)

    fig = go.Figure()

    # 실제 미세먼지
    fig.add_trace(go.Scatter(
        x=df.index[-360:], 
        y=df['미세먼지 (μg/m³)'].iloc[-360:], 
        mode='lines', 
        name='실제 미세먼지', 
        line=dict(color='purple', dash='dash')
    ))

    fig.update_layout(
        title='미세먼지 수치',
        xaxis_title='시간',
        yaxis_title='미세먼지 (㎍/m³)',
        xaxis=dict(tickformat='%H:%M'),  # 소수점 초를 없애고 시:분 형식으로 표시
        template='plotly_white',
        dragmode='zoom'
    )

    img = BytesIO()
    try:
        pio.write_image(fig, img, format='png')
    except Exception as e:
        print(f"Error generating image: {e}")
        raise
    img.seek(0)

    return img

# 소음 그래프 생성 함수
def generate_noise_graph():
    df = validate_data_file(data_file)

    fig = go.Figure()

    # 실제 소음
    fig.add_trace(go.Scatter(
        x=df.index[-360:], 
        y=df['소리 값'].iloc[-360:], 
        mode='lines', 
        name='실제 소리 값', 
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title='소음 수치',
        xaxis_title='시간',
        yaxis_title='소음 (dB)',
        xaxis=dict(tickformat='%H:%M'),  # 소수점 초를 없애고 시:분 형식으로 표시
        template='plotly_white',
        dragmode='zoom'
    )

    img = BytesIO()
    try:
        pio.write_image(fig, img, format='png')
    except Exception as e:
        print(f"Error generating image: {e}")
        raise
    img.seek(0)

    return img

@app.route('/data')
def data():
    try:
        df = validate_data_file(data_file)

        labels = df.index[-360:]
        temperature_data = df['온도 (°C)'].iloc[-360:].tolist()
        humidity_data = df['습도 (%)'].iloc[-360:].tolist()
        dust_data = df['미세먼지 (μg/m³)'].iloc[-360:].tolist()
        noise_data = df['소리 값'].iloc[-360:].tolist()

        # 평균 계산
        temperature_avg = df['온도 (°C)'].iloc[-360:].mean()
        humidity_avg = df['습도 (%)'].iloc[-360:].mean()
        dust_avg = df['미세먼지 (μg/m³)'].iloc[-360:].mean()
        noise_avg = df['소리 값'].iloc[-360:].mean()

        # 예측 데이터 가져오기
        prediction_labels = []
        prediction_temperature = []
        prediction_humidity = []

        if os.path.exists(prediction_file_path):
            pred_df = pd.read_excel(prediction_file_path)

            prediction_labels = pred_df['예측 시간'].astype(str).tolist()
            prediction_temperature = pred_df['예측 온도'].tolist()
            prediction_humidity = pred_df['예측 습도'].tolist()
        else:
            raise FileNotFoundError(f"{prediction_file_path} 파일을 찾을 수 없습니다.")

        response_data = {
            'labels': labels.astype(str).tolist(),
            'temperature': temperature_data,
            'humidity': humidity_data,
            'dust': dust_data,
            'noise': noise_data,
            'temperature_avg': temperature_avg,
            'humidity_avg': humidity_avg,
            'dust_avg': dust_avg,
            'noise_avg': noise_avg,
            'prediction_labels': prediction_labels,
            'prediction_temperature': prediction_temperature,
            'prediction_humidity': prediction_humidity
        }

        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 루트 페이지: 그래프를 보여주는 HTML 페이지
@app.route('/')
def index():
    return render_template('index.html')

# 온도 및 습도 예측 페이지
@app.route('/fu')
def fu():
    return render_template('fu.html')

# 온도 및 습도 그래프 이미지 반환
@app.route('/temp_hum_plot.png')
def temp_hum_plot_png():
    img = generate_temp_hum_graph()
    return send_file(img, mimetype='image/png')

# 미세먼지 그래프 이미지 반환
@app.route('/dust_plot.png')
def dust_plot_png():
    img = generate_dust_graph()
    return send_file(img, mimetype='image/png')

# 소음 그래프 이미지 반환
@app.route('/noise_plot.png')
def noise_plot_png():
    img = generate_noise_graph()
    return send_file(img, mimetype='image/png')

# favicon 요청 무시
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content

# 온도 및 습도 페이지
@app.route('/temperature_humidity')
def temperature_humidity():
    return render_template('temperature_humidity.html')

# 소음 페이지
@app.route('/noise')
def noise():
    return render_template('noise.html')

# 미세먼지 페이지
@app.route('/dust')
def dust():
    return render_template('dust.html')

# 애플리케이션 실행
if __name__ == '__main__':
    try:
        # Flask 서버를 메인 스레드에서 실행
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting Flask server: {e}")

