# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os



def detect_and_predict_mask(frame, faceNet, maskNet):
	#Pega as dimensões do frame e faz o BlobImage
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	#Detecção de Face a partir do blob
	faceNet.setInput(blob)
	detections = faceNet.forward()

	#Inicia as listas de Face, Loais (coordenadas) e Predicções
	faces = []
	locs = []
	preds = []

	#Processamento de imagem enquanto ouver detecções
	for i in range(0, detections.shape[2]):
		#Trás a probabilidade associada a detecção
		confidence = detections[0, 0, i, 2]

		#Determina um nível mínimo de confiança na hora da detecção  
		if confidence > args["confidence"]:
			#Calcula o x e y dos bounding box de cada detecção
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			#Certifique-se de que as caixas delimitadoras estejam dentro das dimensões da moldura
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#Extrai o ROI da Face, convert eBGR para RGB, redimensiona para 224x224 para pre processar
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			#Adiciona as listas as faces e os locias (coordenadas)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
			
			#Salva em uma variável axuliar a quantidade de faces detectadas
			global countPeople 
			countPeople = len(faces)

	#Determina as predições se alguma Face por detectada
	if len(faces) > 0:
		#Faz predicções em todas as Faces de uma vez
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	#REtorno das Faces, Locais (coordenadas) e quantidade de faces
	return (locs, preds, countPeople)

#Analisa os argumentos chamados no terminal
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Carrega o modelo de detecção de face
print("Carregado modelo de Face Detector")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#Carrega o modelo de dectção de mascára
print("Carregando modelo de Mask Detector")
maskNet = load_model(args["model"])

#Inicializa a câmera
print("Iniciando Câmera")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	#Redemensionamento do vídeo
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	#Faz a detecção das Faces e se estão usando máscaraa
	(locs, preds, countPeople) = detect_and_predict_mask(frame, faceNet, maskNet)
	

	#Pega as Faces e os locais encotnrados
	for (box, pred) in zip(locs, preds):
		#Trás os bounding box e predicções 
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		#Determina se existe máscara ou não e qual a cor associada ao Label delas
		label = "Mascara" if mask > withoutMask else "Sem Mascara"
		color = (0, 255, 0) if label == "Mascara" else (0, 0, 255)

		#Salva numa variável auxiliar se existe máscra ou não
		compareLabel = label

		#Contatena com o Label a probabilidade de ter ou não máscara
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
	
		#Motras os Labels e as informações referentes ao processamento da imagem
		cv2.putText(frame, "Numero Pessoas: " + str(countPeople),  (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 2)
		#Verifica a quantidade de pessoas e se estão sem máscara para exibir um aviso
		if (countPeople >= 2 and compareLabel == "Sem Mascara"):
			cv2.putText(frame, "AGLOMERACAO",  (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,255), 2)
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
		print("Face")
		print(countPeople)
	
	#Retorna a imagem processada
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#Determina a condição de paaraada
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
