import pandas as pd #đọc dữ liệu
import numpy as np #xử lý dữ liệu
import matplotlib.pyplot as plt #vẽ biểu đồ
from sklearn.preprocessing import MinMaxScaler #chuẩn hóa dữ liệu
from keras.callbacks import ModelCheckpoint #lưu lại huấn luyện tốt nhất
from keras.models import load_model
from vnstock import *
import pandas_ta as ta
#các lớp để xây dựng mô hình
from keras.models import Sequential #đầu vào
from keras.layers import LSTM #học phụ thuộc
from keras.layers import Dropout #tránh học tủ
from keras.layers import Dense #đầu ra

#kiểm tra độ chính xác của mô hình
from sklearn.metrics import r2_score #đo mức độ phù hợp
from sklearn.metrics import mean_absolute_error #đo sai số tuyệt đối trung bình
from sklearn.metrics import mean_absolute_percentage_error #đo % sai số tuyệt đối trung bình

def Stock_prediction_mode(date, macp):
    start_date = "2013-01-01"
    #df = stock_historical_data(symbol=macp, start_date="2021-01-01", end_date = str(date), resolution="1D", type="stock", beautify=True, decor=False, source='DNSE')
    df = stock_historical_data(symbol=macp, start_date=start_date, end_date=str(date), resolution="1D", type="stock", beautify=True, decor=False)
    df.drop(columns=['ticker'], inplace=True)
    #định dạng cấu trúc thời gian
    df["time"] = pd.to_datetime(df.time, format="%Y-%m-%d")
    ta.supertrend(high=df['high'],low=df['low'],close=df['close'],period = 7,multiplier= 3)  #applying supertrend algo

    df['sup'] = ta.supertrend(high=df['high'],low=df['low'],close=df['close'],period = 7,multiplier= 3)['SUPERT_7_3.0']
    df1 = pd.DataFrame(df,columns=['time','close', 'open', 'high', 'low'])
    df1.index = df1.time
    df1.drop('time',axis=1,inplace=True)
    data = df1.values
    train_data = data[:1500]
    test_data = data[1500:]
    sc = MinMaxScaler(feature_range=(0, 1))
    sc_train = sc.fit_transform(data)  # Chỉ sử dụng train_data để fit scaler
    x_train, y_train = [], []
    for i in range(50, len(train_data)):
        x_train.append(sc_train[i-50:i, 0:4])  # Lấy 50 giá trị liên tiếp, bao gồm cả close và open, high, low
        y_train.append(sc_train[i, 0:4])  # Lấy ra giá close và open, high, low ngày hôm sau
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    #xếp lại dữ liệu thành mảng 1 chiều
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],4))
    y_train = np.reshape(y_train,(y_train.shape[0],4))

    model = Sequential()  # Tạo lớp mạng cho dữ liệu đầu vào

    # Thêm 2 lớp LSTM mới, tổng cộng là 3 lớp LSTM với tham số return_sequences=True
    model.add(LSTM(units=128, input_shape=(x_train.shape[1], 4), return_sequences=True))
    model.add(Dropout(0.2))  # Thêm Dropout để giảm overfitting sau mỗi lớp LSTM

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))  # Thêm Dropout

    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))  # Thêm Dropout

    # Lớp LSTM cuối cùng không cần tham số return_sequences vì đây là lớp cuối trước khi kết nối với Dense layer
    model.add(LSTM(units=64))
    model.add(Dropout(0.5))  # Giữ nguyên Dropout như lớp LSTM cuối cũ trong mô hình ban đầu

    model.add(Dense(4))  # Output đầu ra 1 chiều

    # Đo sai số tuyệt đối trung bình có sử dụng trình tối ưu hóa adam
    model.compile(loss='mean_absolute_error', optimizer='adam')

    #huấn luyện mô hình
    save_model = "../model/models3.keras"
    best_model = ModelCheckpoint(save_model,monitor='loss',verbose=2,save_best_only=True,mode='auto')
    model.fit(x_train,y_train,epochs=100,batch_size=50,verbose=2,callbacks=[best_model])

    #dữ liệu train
    y_train = sc.inverse_transform(y_train) #giá thực
    final_model = load_model("../model/models3.keras")
    y_train_predict = final_model.predict(x_train) #dự đoán giá đóng cửa trên tập đã train
    y_train_predict = sc.inverse_transform(y_train_predict) #giá dự đoán

    test = df1[len(train_data)-50:].values  # Lấy dữ liệu test với cả hai cột

    # Sử dụng MinMaxScaler đã được fit với cả hai cột để biến đổi
    sc_test = sc.transform(test)

    x_test, y_test = [], []
    for i in range(50, test.shape[0]):
        x_test.append(sc_test[i-50:i])
    x_test = np.array(x_test)

    # Đối với mô hình LSTM, dữ liệu đầu vào cần được reshape thành [samples, time steps, features]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 4))

    # Sau đó, bạn có thể sử dụng mô hình để dự đoán
    y_test_predict = final_model.predict(x_test)
    y_test_predict = sc.inverse_transform(y_test_predict)

    # Lấy ngày kế tiếp sau ngày cuối cùng trong tập dữ liệu để dự đoán
    next_date = df['time'].iloc[-1] + pd.Timedelta(days=1)

    # Sử dụng mô hình để dự đoán

    # Điều chỉnh đoạn code để lấy 50 điểm dữ liệu gần nhất từ cả 'close' và 'open' cho dự đoán
    x_next = np.array([sc_train[-50:, :]])  # Lấy 50 bản ghi gần nhất cho cả 'close' và 'open'
    x_next = np.reshape(x_next, (x_next.shape[0], x_next.shape[1], 4))  # Reshape để phù hợp với mô hình

    y_next_predict = final_model.predict(x_next)
    y_next_predict = sc.inverse_transform(y_next_predict)  # Chuyển về giá trị ban đầu
    # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
    df_next = pd.DataFrame({'time': [next_date],
                        'close': [y_next_predict[0][0]],
                        'open': [y_next_predict[0][1]],
                        'high': [y_next_predict[0][2]],
                        'low': [y_next_predict[0][3]]})
    df1 = pd.concat([df1, df_next.set_index('time')])
    comparison_df = pd.DataFrame({
        'time': [next_date],
        'Giá đóng cửa dự đoán': [y_next_predict[0][0]],
        'Giá mở cửa dự đoán': [y_next_predict[0][1]],
        'Giá cao nhất dự đoán': [y_next_predict[0][2]],  # Giả sử cột thứ ba là dự đoán giá cao nhất
        'Giá thấp nhất dự đoán': [y_next_predict[0][3]]  # Giả sử cột thứ tư là dự đoán giá thấp nhất
    })
    return comparison_df

