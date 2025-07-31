import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Tanaman Tomat",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Daftar kelas penyakit
CLASS_NAMES = [
    'Hawar Daun',      # Hawar daun
    'Kudis Daun',      # Kudis daun
    'Infeksi Bakteri', # Infeksi bakteri
    'Sehat'            # Sehat
]

# Penjelasan penyakit dalam Bahasa Indonesia
DISEASE_INFO = {
    'Hawar Daun': {
        'name': 'Hawar Daun',
        'description': 'Penyakit ini ditandai dengan bercak coklat pada daun yang dapat menyebar dengan cepat.',
        'treatment': [
            'Gunakan fungisida berbahan aktif klorotalonil.',
            'Hindari kelembapan berlebih pada tanaman.',
            'Buang daun yang terinfeksi.'
        ]
    },
    'Kudis Daun': {
        'name': 'Kudis Daun',
        'description': 'Penyakit ini menyebabkan bercak-bercak putih atau kuning pada daun.',
        'treatment': [
            'Gunakan fungisida berbahan aktif tembaga.',
            'Pastikan sirkulasi udara yang baik di sekitar tanaman.',
            'Hindari penyiraman dari atas.'
        ]
    },
    'Infeksi Bakteri': {
        'name': 'Infeksi Bakteri',
        'description': 'Penyakit ini menyebabkan bercak air pada daun dan dapat menyebabkan kerusakan serius.',
        'treatment': [
            'Gunakan antibiotik tanaman jika tersedia.',
            'Hindari penyiraman yang membasahi daun.',
            'Buang tanaman yang terinfeksi untuk mencegah penyebaran.'
        ]
    },
    'Sehat': {
        'name': 'Sehat',
        'description': 'Tanaman tomat dalam kondisi sehat tanpa tanda-tanda penyakit.',
        'treatment': [
            'Lanjutkan praktik budidaya yang baik.',
            'Pantau tanaman secara rutin.',
            'Jaga kebersihan kebun dari gulma dan sisa tanaman.'
        ]
    }
}

# Fungsi untuk memuat model
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# Fungsi untuk memproses gambar
def preprocess_image(img):
    img = img.resize((224, 224))  # Ubah ukuran gambar sesuai dengan input model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Judul aplikasi
st.title("Deteksi Penyakit Tanaman Tomat")

# Unggah gambar
uploaded_file = st.file_uploader("Unggah gambar daun tomat", type=["jpg", "jpeg", "png"])

# Memuat model
model_path = "/Users/humairahafizah/Documents/streamlit_project/model_resnet50_tomat22.h5"  # Ganti dengan path model Anda
model = load_model(model_path)

if uploaded_file is not None and model is not None:
    # Proses gambar
    img = Image.open(uploaded_file)
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Prediksi
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"Penyakit yang terdeteksi: **{predicted_class}**")

    # Tampilkan informasi penyakit
    disease_info = DISEASE_INFO[predicted_class]
    st.write(f"**Deskripsi:** {disease_info['description']}")
    st.write("**Pengobatan:**")
    for treatment in disease_info['treatment']:
        st.write(f"- {treatment}")

