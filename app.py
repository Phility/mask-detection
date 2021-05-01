import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image

#Desabilitando Warning
st.set_option('deprecation.showfileUploaderEncoding', False)

#Configuração início do título da página
st.beta_set_page_config(page_title='Aglomerou.SmartBR', page_icon='👥', layout='centered', initial_sidebar_state='expanded')


def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def mask_image():
    global RGB_img
    #Carrega o modelo de detecção de face
    print("Carregado modelo de Face Detector")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    #Faz a leitura da rede de deep learning
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    #Carrega o modelo de dectção de mascára
    print("Carregando modelo de Mask Detector")
    model = load_model("mask_detector.model")

    #Carrega a imagem de entrada e pega suas dimensões
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    #Constroi o Blob Image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    #Detecção de Face a partir do blob
    print("Carregando Face Detection")
    net.setInput(blob)
    detections = net.forward()

    #Criaçãode variáveis para auxiliar na contagem de faces
    count = 0
    countPeople = []

    #Processamento de imagem enquanto ouver detecções
    for i in range(0, detections.shape[2]):
        #Trás a probabilidade associada a detecção
        confidence = detections[0, 0, i, 2]

        #Determina um nível mínimo de confiança na hora da detecção  
        if confidence > 0.5:
            # Calcula o x e y dos bounding box de cada detecção
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #Certifique-se de que as caixas delimitadoras estejam dentro das dimensões da moldura
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #Extrai o ROI da Face, convert eBGR para RGB, redimensiona para 224x224 para pre processar
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            #Faz a contagem de iterações para saber o número de pessoas detectadas
            count = count + 1
            countPeople.append(count)
            print("Face")
            print(countPeople)

            #Prediz se o rosto tem um máscara ou não através do modelo
            (mask, withoutMask) = model.predict(face)[0]

            #Determina se existe máscara ou não e qual a cor associada ao Label delas
            label = "Mascara" if mask > withoutMask else "Sem Mascara"
            color = (0, 255, 0) if label == "Mascara" else (0, 0, 255)

            #Salva numa variável auxiliar se existe máscra ou não
            compareLabel = label

            #Contatena com o Label a probabilidade de ter ou não máscara
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            #Motras os Labels e as informações referentes ao processamento da imagem
            cv2.putText(image, "Numero Pessoas: " + str(countPeople),  (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,0,0), 2)
            #Verifica a quantidade de pessoas e se estão sem máscara para exibir um aviso
            if (count >= 2 and compareLabel == "Sem Mascara"):
                cv2.putText(image, "AGLOMERACAO",  (500,15), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,255), 2)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_image()

def mask_detection():
    #Construção da página web para carregamento da foto
    local_css("css/styles.css")
    #st.markdown('<h1 align="center">Aglomerou.SmartBR</h1>', unsafe_allow_html=True)
    #st.markdown("<img src='https://uploaddeimagens.com.br/images/003/221/952/full/logo.jpg?1619849282' alt='logo' align='middle'>", unsafe_allow_html=True)

    choice = 'Image'
    if choice == 'Image':
        st.markdown('<div id="home">', unsafe_allow_html=True)
        st.markdown('<h2 align="center"><b>Detecção de aglomerações e pessoas sem máscara</b></h2>', unsafe_allow_html=True)
        image_file = st.file_uploader("", type=['jpg']) 
        if image_file is not None:
            our_image = Image.open(image_file)
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Imagem carregada!</h3>', unsafe_allow_html=True)
            if st.button('Detectar'):
                st.image(RGB_img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
mask_detection()
