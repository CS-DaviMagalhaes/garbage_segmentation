from keras.models import load_model  # Libreria Tensorflow que incluye keras
import cv2  # Libreria opencv-python para cámara
import numpy as np
import time 

# desactivar notación cientifica (detallito nomás)
np.set_printoptions(suppress=True)

# Cargar modelo y labels (clases que queremos predecir)
model = load_model("./models/keras_v2_100_ep.h5", compile=False)
class_names = open("./models/labels.txt", "r").readlines()

# Instanciar camara
camera = cv2.VideoCapture(0) #Puede ser 0 o 1, dependiendo de la camara

#Variables para confirmar predicción segun cierto valor de precisión y tiempo
confidence_threshold = 0.80  # 80% certeza
time_threshold = 2.0  # 2 segundos
start_time = None
current_class = None

# Loop infinito para predecir y mostrar imagen
while True:
    ret, image = camera.read() #Leer la imagen de la cámara

    #Redimensionar imagenes de la webcam a (224px,224px), igual que las imagenes del modelo
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) 
    
    cv2.imshow("Webcam Image", image) #Mostrar imagen de la webcam

    # Redimensionar imagen y convertir a array numerico para que el modelo lo procese
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1 #Normalizar array

    # Predecir
    prediction = model.predict(image, verbose=0) #Verbose=0 para que keras deje de imprimir en consola
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Doy el mismo output para metal y cartón ya que no queremos hacer nada con esas 2 clases
    classes = {"0 Plastic":0, "1 Glass":1, "2 Paper":2, "3 Metal":3, "4 Cardboard":3} 

    if confidence_score >= confidence_threshold: #Si tengo más de threshold% de certeza
        if current_class == class_name: # si coincide la clase que tengo
            
            # empiezo a contar el tiempo
            elapsed_time = time.time() - start_time
            if elapsed_time >= time_threshold: # si tengo más de x segundos en la misma clase, confirmo la prediccion
                print(f"Confirmed Class: {class_name} with {confidence_score * 100:.2f}% confidence") #mensaje en consola
                
                # TODO: mandar class_name a código arduino aqui, para que mueva los motores <-------------
                
                start_time = time.time() #reseteo timer             
                
        else:
            # Si tengo una nueva clase empiezo el timer denuevo
            current_class = class_name
            start_time = time.time()
    else:
        # Reseteo si la certeza es menor que el threshold
        current_class = None
        start_time = None

    # Detener programa y camara si se presiona la tecla ESC
    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

#destructores
camera.release() 
cv2.destroyAllWindows()