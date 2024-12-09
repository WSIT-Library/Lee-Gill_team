import serial
import time
import pandas as pd
import os
import json
from datetime import datetime
import subprocess
import logging
import shutil
from serial.tools import list_ports  # 추가된 부분
import asyncio
from bleak import BleakScanner, BleakClient  # Bleak 라이브러리 추가

# Bluetooth 동글 연결 포트와 보드레이트 설정
baud_rate = 9600
data_file_path = '실내환경데이터.xlsx'  # 저장할 Excel 파일 경로
backup_folder = '백업'  # 백업 파일을 저장할 폴더
최소_데이터_수 = 360  # 모델 학습을 위한 최소 데이터 수
백업_주기 = 120  # 백업 주기 (데이터 수)
최대_백업_개수 = 5  # 최대 백업 파일 개수

# 데이터 기록을 위한 빈 DataFrame 생성
data_columns = ["년월일", "소리 값", "소리 백분율", "온도 (°C)", "습도 (%)", "미세먼지 (μg/m³)"]

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 시리얼 연결 함수 (자동 포트 탐색 및 재시도)
def connect_serial():
    while True:
        ports = list_ports.comports()  # 연결된 포트 리스트 가져오기
        bluetooth_port = None

        # Bluetooth 포트 찾기
        for port in ports:
            if 'Bluetooth' in port.description or 'bluetooth' in port.device:
                bluetooth_port = port.device
                break

        if bluetooth_port:
            try:
                ser = serial.Serial(bluetooth_port, baud_rate, timeout=1)
                time.sleep(2)
                logging.info(f"Bluetooth 동글 연결 성공: {bluetooth_port} 포트에 연결되었습니다.")
                return ser
            except serial.SerialException as e:
                logging.error(f"Bluetooth 동글 연결 실패: {e}, 재시도 중...")
        else:
            logging.error("Bluetooth 포트를 찾을 수 없습니다. 5초 후 재시도...")
        
        time.sleep(5)  # 5초 후 재시도

# Bluetooth 장치 검색 함수
async def search_bluetooth_devices():
    logging.info("블루투스 장치 검색 중...")
    devices = await BleakScanner.discover()

    # 연결된 장치만 필터링
    connected_devices = []
    for device in devices:
        try:
            async with BleakClient(device.address) as client:
                connected_devices.append(device)
                logging.info(f"연결된 장치: {device.name} ({device.address})")
        except Exception as e:
            logging.warning(f"{device.name} ({device.address})에 연결할 수 없습니다: {e}")

    if not connected_devices:
        logging.info("연결된 블루투스 장치가 없습니다.")
    
    return connected_devices

# Bluetooth 장치에 연결하는 함수
async def connect_to_bluetooth_device(address):
    logging.info(f"{address}에 연결 중...")
    try:
        async with BleakClient(address) as client:
            logging.info(f"Bluetooth 연결 성공: {address}")
            return client
    except Exception as e:
        logging.error(f"Bluetooth 연결 실패: {e}")
        return None

# 데이터 저장 함수
def save_data(new_entry):
    while True:
        try:
            if os.path.exists(data_file_path):
                existing_data = pd.read_excel(data_file_path, engine='openpyxl')  # 엔진 지정
                combined_data = pd.concat([existing_data, new_entry], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=["년월일", "소리 값", "온도 (°C)", "습도 (%)", "미세먼지 (μg/m³)"], keep='last')
                combined_data.to_excel(data_file_path, index=False, engine='openpyxl')  # 엔진 지정
            else:
                # 새로운 파일을 생성할 때 빈 DataFrame으로 초기화
                new_entry = pd.DataFrame(columns=data_columns)  # 빈 DataFrame 생성
                new_entry.to_excel(data_file_path, index=False, engine='openpyxl')  # 엔진 지정
                save_data(new_entry)  # 새로 생성한 빈 파일에 데이터 저장

            logging.info("데이터가 Excel 파일에 저장되었습니다.")
            break  # 저장이 성공하면 루프 종료
        except PermissionError:
            logging.warning("파일이 다른 프로그램에서 열려 있습니다. 잠시 후 다시 시도합니다...")
            time.sleep(5)  # 5초 대기 후 재시도

# 데이터 백업 함수
def backup_data():
    try:
        # 백업 폴더가 없다면 생성
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        # 최대 백업 개수 관리
        backup_files = sorted(os.listdir(backup_folder), key=lambda x: os.path.getmtime(os.path.join(backup_folder, x)))
        
        # 최대 개수를 초과하면 가장 오래된 백업 파일 삭제
        while len(backup_files) >= 최대_백업_개수:
            oldest_file = os.path.join(backup_folder, backup_files[0])
            os.remove(oldest_file)
            logging.info(f"오래된 백업 파일 삭제됨: {oldest_file}")
            backup_files.pop(0)  # 리스트에서 삭제된 파일 제거

        # 백업 파일 이름 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file_path = os.path.join(backup_folder, f'실내환경데이터_백업_{timestamp}.xlsx')  # 백업 파일 이름

        # 백업 파일 복사
        shutil.copy(data_file_path, backup_file_path)
        logging.info(f"백업 생성됨: {backup_file_path}")

    except Exception as e:
        logging.error(f"백업 중 오류 발생: {e}")

# 데이터 수집 및 저장 함수
async def data_collection():
    # Bluetooth 동글 연결 시도
    ser = connect_serial()  # Bluetooth 동글 연결

    if ser:  # 동글이 성공적으로 연결되면
        logging.info("Bluetooth 동글을 통해 데이터 수집을 시작합니다.")
        
        while True:
            try:
                # 동글을 통해 데이터 수신
                line = ser.readline().decode('utf-8').strip()  # 동글로부터 데이터 읽기
                if not line:
                    time.sleep(0.5)
                    continue

                # 수신된 데이터 로그 추가
                logging.info(f"수신된 데이터: {line}")

                # 데이터가 JSON 형식인지 확인
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON 파싱 오류 발생: {e}")
                        logging.error(f"유효하지 않은 JSON 데이터: {line}")
                        continue

                    # 데이터 유효성 검사
                    required_keys = ["sound_value", "sound_percentage", "temperature", "humidity", "dust_density"]
                    if all(key in data for key in required_keys):
                        timestamp = datetime.now()
                        new_entry = pd.DataFrame({
                            "년월일": [timestamp.strftime("%Y-%m-%d %H:%M:%S")],
                            "소리 값": [data["sound_value"]],
                            "소리 백분율": [data["sound_percentage"]],
                            "온도 (°C)": [data["temperature"]],
                            "습도 (%)": [data["humidity"]],
                            "미세먼지 (μg/m³)": [data["dust_density"]],
                        })

                        # 신규 데이터 추가 및 엑셀에 즉시 저장
                        if not new_entry.empty:
                            if new_entry.isnull().values.any():
                                logging.warning("신규 데이터에 NaN 값이 포함되어 있습니다: %s", new_entry)
                            else:
                                save_data(new_entry)

                                # 현재 데이터 개수 확인
                                total_data_count = len(pd.read_excel(data_file_path))

                                # 저장된 데이터의 개수가 360개일 때 백업 및 LSTM 모델 학습 신호 보내기
                                if total_data_count % 최소_데이터_수 == 0 and total_data_count > 0:
                                    logging.info("모델 학습을 위한 데이터가 충분히 수집되었습니다. LSTM 모델 학습 신호를 보냅니다.")
                                    
                                    # lstm5.py 파일 존재 여부 확인 후 실행 (비동기 처리)
                                    lstm_script_path = 'lstm5.py'
                                    if os.path.exists(lstm_script_path):
                                        subprocess.Popen(['python', lstm_script_path])  # lstm5.py 비동기로 실행
                                        logging.info(f"{lstm_script_path} 실행 중...")
                                    else:
                                        logging.error(f"오류: {lstm_script_path} 파일이 존재하지 않습니다.")

                                # 백업 주기마다 백업 수행
                                if total_data_count % 백업_주기 == 0 and total_data_count > 0:
                                    backup_data()

                    else:
                        missing_keys = [key for key in required_keys if key not in data]
                        logging.error(f"수신된 데이터에 필요한 키가 누락되었습니다: {missing_keys}")
                else:
                    logging.error(f"유효하지 않은 데이터 수신: {line}")

            except Exception as e:
                logging.error(f"데이터 처리 중 오류 발생: {e}")
                continue  # 예외 발생 시 다음 루프 실행

    else:
        # 동글 연결 실패 시 Bluetooth 장치 검색
        logging.info("Bluetooth 동글 연결 실패. Bluetooth 장치 검색을 시작합니다.")
        devices = await search_bluetooth_devices()  # Bluetooth 장치 검색
        if not devices:
            logging.error("블루투스 장치를 찾을 수 없습니다. 프로그램을 종료합니다.")
            return

        # 사용자가 연결할 장치 선택
        for idx, device in enumerate(devices):
            logging.info(f"{idx + 1}: {device.name} ({device.address})")
        
        selected_device = int(input("연결할 블루투스 장치 번호를 입력하세요: ")) - 1
        selected_address = devices[selected_device].address

        # 선택한 Bluetooth 장치에 연결
        client = await connect_to_bluetooth_device(selected_address)
        if not client:
            return

        new_data = pd.DataFrame(columns=data_columns)  # 데이터 초기화

        while True:
            try:
                line = await client.read_gatt_char('YOUR_CHARACTERISTIC_UUID')  # UUID에 맞는 특성 읽기
                line = line.decode('utf-8').strip()  # Bluetooth 소켓에서 데이터 읽기
                if not line:
                    time.sleep(0.5)
                    continue

                # 수신된 데이터 로그 추가
                logging.info(f"수신된 데이터: {line}")

                # 데이터가 JSON 형식인지 확인
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON 파싱 오류 발생: {e}")
                        logging.error(f"유효하지 않은 JSON 데이터: {line}")
                        continue

                    # 데이터 유효성 검사
                    required_keys = ["sound_value", "sound_percentage", "temperature", "humidity", "dust_density"]
                    if all(key in data for key in required_keys):
                        timestamp = datetime.now()
                        new_entry = pd.DataFrame({
                            "년월일": [timestamp.strftime("%Y-%m-%d %H:%M:%S")],
                            "소리 값": [data["sound_value"]],
                            "소리 백분율": [data["sound_percentage"]],
                            "온도 (°C)": [data["temperature"]],
                            "습도 (%)": [data["humidity"]],
                            "미세먼지 (μg/m³)": [data["dust_density"]],
                        })

                        # 신규 데이터 추가 및 엑셀에 즉시 저장
                        if not new_entry.empty:
                            if new_entry.isnull().values.any():
                                logging.warning("신규 데이터에 NaN 값이 포함되어 있습니다: %s", new_entry)
                            else:
                                save_data(new_entry)

                                # 현재 데이터 개수 확인
                                total_data_count = len(pd.read_excel(data_file_path))

                                # 저장된 데이터의 개수가 360개일 때 백업 및 LSTM 모델 학습 신호 보내기
                                if total_data_count % 최소_데이터_수 == 0 and total_data_count > 0:
                                    logging.info("모델 학습을 위한 데이터가 충분히 수집되었습니다. LSTM 모델 학습 신호를 보냅니다.")
                                    
                                    # lstm5.py 파일 존재 여부 확인 후 실행 (비동기 처리)
                                    lstm_script_path = 'lstm5.py'
                                    if os.path.exists(lstm_script_path):
                                        subprocess.Popen(['python', lstm_script_path])  # lstm5.py 비동기로 실행
                                        logging.info(f"{lstm_script_path} 실행 중...")
                                    else:
                                        logging.error(f"오류: {lstm_script_path} 파일이 존재하지 않습니다.")

                                # 백업 주기마다 백업 수행
                                if total_data_count % 백업_주기 == 0 and total_data_count > 0:
                                    backup_data()

                    else:
                        missing_keys = [key for key in required_keys if key not in data]
                        logging.error(f"수신된 데이터에 필요한 키가 누락되었습니다: {missing_keys}")
                else:
                    logging.error(f"유효하지 않은 데이터 수신: {line}")

            except Exception as e:
                logging.error(f"데이터 처리 중 오류 발생: {e}")
                continue  # 예외 발생 시 다음 루프 실행

# 데이터 수집 시작
async def main():
    await data_collection()

if __name__ == "__main__":
    asyncio.run(main())
