
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Đường dẫn đến mô hình đã được huấn luyện
model_path = "/content/drive/MyDrive/chess/vgg16_model.h5"

# Load mô hình từ file .h5
model = load_model(model_path)

# Kích thước ảnh đầu vào của mô hình
input_shape = (224, 224)

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = "/content/drive/MyDrive/chess/dataset_model/train"

# Danh sách các lớp dựa trên tên thư mục trong thư mục chứa dữ liệu
CLASSES = sorted(os.listdir(data_dir))

# Chuẩn bị hình ảnh đầu vào và dự đoán lớp
def predict_class(img_path):
    # Đọc và tiền xử lý hình ảnh
    img = image.load_img(img_path, target_size=input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Mở rộng chiều để tạo batch

    # Chuẩn hóa dữ liệu
    img_array = img_array / 255.0

    # Dự đoán lớp của hình ảnh
    predictions = model.predict(img_array)

    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_class_index = np.argmax(predictions)

    # Trả về tên lớp tương ứng với chỉ số dự đoán
    return CLASSES[predicted_class_index]

# Tiêu đề cho ứng dụng Streamlit
st.title('Chess Piece Recognition')

# Widget để upload hình ảnh từ máy tính
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

# Nếu người dùng đã upload ảnh, thực hiện dự đoán và hiển thị kết quả
if uploaded_file is not None:
    # Hiển thị hình ảnh đã upload
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Thực hiện dự đoán lớp của hình ảnh
    predicted_class = predict_class(uploaded_file)

    # Hiển thị kết quả dự đoán
    st.write("Predicted class:", predicted_class)

