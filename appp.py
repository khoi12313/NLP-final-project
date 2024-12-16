
import streamlit as st
import joblib
from transformers import AutoModel, AutoTokenizer


# Hàm chuyển câu thành vector bằng PhoBERT
def sentence_to_vector_phobert(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=256)
    outputs = phobert(**inputs)
    # Lấy vector embedding từ tầng đầu ra (CLS token)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()
    return cls_embedding



# Cấu hình giao diện trang
st.set_page_config(page_title="NLP App", layout="centered", initial_sidebar_state="expanded")

# Thêm hình nền cho trang
st.markdown(
    """
    <h1 style='
        text-align: center;
        color: #d36f31;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    '> 
        Phân tích quan điểm bình luận dựa trên các đánh giá về chủ đề phim
    </h1>
    <style>
    .stApp {
        background-image: url('https://www.ueh.edu.vn/images/upload/editer/dien%20mao%20UEH%20don%20xuan%202022_1_N.jpg');
        background-size: cover;
        background-position: center;
    }
    .result-box {
        border-radius: 10px;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.9);
        color: #000;
        font-size: 18px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: chọn mô hình
st.sidebar.title("Tuỳ chọn")
model_option = st.sidebar.selectbox(
    "Mô hình:",
    ["Naive Bayes", "Logistic Regression", "SVM", "LSTM"]
)

# Hộp nhập văn bản với placeholder
user_input = st.text_area(
    "", 
    placeholder="Hãy nhập comment nào đó từ trên cái web phim...",
    height=150,
    label_visibility="collapsed"  # Ẩn label mặc định của text_area
)


if __name__ == '__main__':
    nb_model = joblib.load('/Users/quachtuankhoi/NLP/model/nb_model.joblib')
    svm_model = joblib.load('/Users/quachtuankhoi/NLP/model/svm_model.joblib')
    logistic_regression = joblib.load('/Users/quachtuankhoi/NLP/model/logistic_regression.joblib')
    model_lstm = joblib.load('/Users/quachtuankhoi/NLP/model/model.joblib')

    # Tải tokenizer và mô hình PhoBERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)
    phobert = AutoModel.from_pretrained("vinai/phobert-base")

    # Nút phân tích
    if st.button("Phân tích quan điểm"):
        if user_input.strip() == "":
            st.warning("Vui lòng nhập comment để phân tích.")
        else:
            polarity = 1
            if model_option == 'Naive Bayes':
                # Chuyển câu thành vector
                vector = sentence_to_vector_phobert(user_input).reshape(1, -1)  # Đảm bảo vector có đúng dạng
                # Dự đoán với mô hình Naive Bayes
                polarity = nb_model.predict(vector)
            elif model_option == "Logistic Regression":
                vector = sentence_to_vector_phobert(user_input).reshape(1, -1)  # Đảm bảo vector có đúng dạng
                # Dự đoán với mô hình Logistic Regression
                polarity = logistic_regression.predict(vector)
            elif model_option == 'SVM':
                vector = sentence_to_vector_phobert(user_input).reshape(1, -1)  # Đảm bảo vector có đúng dạng
                # Dự đoán với mô hình SVM
                polarity = svm_model.predict(vector)
            elif model_option == 'LSTM':
                vector = sentence_to_vector_phobert(user_input)
                # Đảm bảo vector có đúng định dạng (batch_size, timesteps, features)
                vector_reshaped = vector.reshape(1, 1, vector.shape[0])  # (1, 1, num_features)
                # Dự đoán với mô hình LSTM trong Keras
                polarity = model_lstm.predict(vector_reshaped)
                polarity[0] = round(float(polarity[0]))


            # Hiển thị kết quả với khung
            if polarity[0] == 1:
                result = "Đây là một comment tích cực!! 😊"
            elif polarity[0] == 0:
                result = "Đây là một comment tiêu cực. 😔"


            st.markdown(f"<div class='result-box'>{result}</div>", unsafe_allow_html=True)


# source env/bin/activate
# streamlit run appp.py