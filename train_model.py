import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

print("Versão do TensorFlow:", tf.__version__)

# --- Constantes e Configurações ---
DATASET_PATH = 'data/garbage-classification'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Definimos épocas para cada fase do treino
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# --- Verificação do Dataset ---
if not os.path.exists(DATASET_PATH):
    print(f"Erro: O diretório do dataset não foi encontrado em '{DATASET_PATH}'")
    print("Por favor, baixe e descompacte o dataset do Kaggle neste local.")
    exit()

# --- Pré-processamento e Geração de Dados (Data Augmentation Agressiva) ---
# O gerador de treino aplica muitas transformações para criar variedade
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2 # Separa 20% dos dados para validação
)

# O gerador de validação NÃO deve ter augmentation, apenas o rescale,
# pois queremos validar o modelo em imagens "limpas", como as do mundo real.
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --- Construção do Modelo ---
print("Construindo o modelo...")
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False # Congela o modelo base inicialmente

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Dropout para regularização
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilação para a primeira fase
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Callbacks Inteligentes ---
# Salva apenas o melhor modelo encontrado durante o treino (baseado na menor perda de validação)
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)
# Para o treinamento se a performance não melhorar por 3 épocas consecutivas
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    verbose=1, 
    restore_best_weights=True # Restaura os pesos da melhor época ao final
)

# --- FASE 1: TREINAMENTO INICIAL (TRANSFER LEARNING) ---
print("\n--- Iniciando FASE 1: Treinamento da 'cabeça' do modelo ---")
history = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping]
)

# --- FASE 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- Iniciando FASE 2: Ajuste Fino (Fine-Tuning) ---")
base_model.trainable = True # Descongela o modelo base

# Vamos descongelar a partir da camada 100 (de 154)
# Isso permite que as camadas de mais alto nível se adaptem aos nossos dados
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompilamos o modelo com uma taxa de aprendizado (learning_rate) muito baixa
# Isso é CRUCIAL para o fine-tuning, para não destruir os pesos já aprendidos
model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.summary()

# Continua o treinamento de onde parou
history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=len(history.history['loss']), # Começa a contar as épocas do ponto onde parou
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping] # Reutiliza os mesmos callbacks
)

# --- FINALIZAÇÃO ---
print("\nTreinamento concluído!")

# Carrega o melhor modelo que foi salvo pelo ModelCheckpoint
print("Carregando o melhor modelo salvo...")
best_model = tf.keras.models.load_model('best_model.h5')

# Salva os nomes das classes para uso no app
print("Salvando os nomes das classes em 'class_names.txt'...")
class_names = list(train_generator.class_indices.keys())
with open('class_names.txt', 'w') as f:
    for name in class_names:
        f.write(f"{name}\n")

print("\nProcesso finalizado com sucesso!")
print("O melhor modelo foi salvo como 'best_model.h5'")