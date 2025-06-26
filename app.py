import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- Configurações da Página e Título ---
st.set_page_config(layout="centered", page_title="Assistente de Reciclagem")
st.title("♻️ Assistente de Reciclagem IA")
st.write(
    "Não tem certeza de como descartar um item? "
    "Tire uma foto e eu direi a categoria correta para você!"
)

# --- Carregamento do Modelo (com cache para performance) ---
# Esta função carrega o modelo e os nomes das classes apenas uma vez.
@st.cache_resource
def load_model_and_classes():
    """Carrega o modelo treinado e os nomes das classes a partir dos arquivos."""
    try:
        model = tf.keras.models.load_model('best_model.h5')
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou arquivo de classes: {e}")
        return None, None

model, CLASS_NAMES = load_model_and_classes()

def preprocess_image(image):
    """Prepara a imagem para ser enviada ao modelo."""
    # Redimensiona para o tamanho que o modelo espera
    img_resized = image.resize((224, 224))
    # Converte para um array numpy e normaliza
    img_array = np.array(img_resized) / 255.0
    # Adiciona uma dimensão extra para o batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Lógica da Interface ---
if model is None or CLASS_NAMES is None:
    st.warning("O modelo não está carregado. Execute o script 'train_model.py' primeiro.")
else:
    # Componente de upload de arquivo do Streamlit
    uploaded_file = st.file_uploader(
        "Tire uma foto ou escolha uma imagem da sua galeria",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Exibe a imagem que o usuário enviou
        image = Image.open(uploaded_file)
        
        # Cria duas colunas para mostrar a imagem e o resultado lado a lado
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Sua imagem', use_column_width=True)

        # Prepara a imagem e faz a predição
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        
        # Pega a classe com maior probabilidade
        score = tf.nn.softmax(predictions[0])
        confidence = 100 * np.max(score)
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index].capitalize()

        # Exibe o resultado na segunda coluna
        with col2:
            st.subheader("Resultado da Análise:")
            st.success(f"**Categoria:** {predicted_class_name}")
            st.info(f"**Confiança:** {confidence:.2f}%")
            
            # Adiciona uma dica sobre o descarte
            dicas_descarte = {
                'cardboard': 'Descarte no lixo de PAPEL/PAPELÃO (Azul).',
                'glass': 'Descarte no lixo de VIDRO (Verde). Cuidado ao manusear.',
                'metal': 'Descarte no lixo de METAL (Amarelo).',
                'paper': 'Descarte no lixo de PAPEL/PAPELÃO (Azul).',
                'plastic': 'Descarte no lixo de PLÁSTICO (Vermelho).',
                'trash': 'Este item não é reciclável. Descarte no LIXO COMUM/REJEITO.'
            }
            st.markdown(f"**Como descartar:** {dicas_descarte.get(CLASS_NAMES[predicted_class_index], 'Consulte as regras locais.')}")