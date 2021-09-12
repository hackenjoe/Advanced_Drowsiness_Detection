from drowsiness_imports import *

def eye_aspect_ratio(eye):
    """Calculates EAR -> ratio length and width of eyes"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculates MAR -> ratio length and width of mouth"""
    A = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    """Calculates PUC -> low perimeter leads to lower pupil"""
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area / (p**2)

def mouth_over_eye(eye):
    """Calculates the MOE -> ratio of MAR to EAR"""
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def sound_alarm(path):
    """play an alarm sound"""
    playsound.playsound(path)
	
def copy_imgs(participants, train_set = True):
    """Helper function for copying and centralizing images"""
    x = [int(x) for x in participants]
    train_set = False
    fileList_a, fileList_d = [], []
    for i in x:
        alert_states = len(glob.glob("img\\"+str(i)+"\\p"+str(i)+"_s0*"))
        drowsy_states = len(glob.glob("img\\"+str(i)+"\\p"+str(i)+"_s10*"))
        MIN_T = 0 if train_set else int((alert_states*0.8))+1
        MAX_T = int((alert_states*0.8)) if train_set else alert_states
        for j in np.arange(MIN_T, MAX_T):
            fileList_a.append(str(i)+"\\p"+str(i)+"_s0_"+str(j)+"sec.jpg")
        MIN_T = 0 if train_set else int((drowsy_states*0.8))+1
        MAX_T = int((drowsy_states*0.8)) if train_set else drowsy_states
        for j in np.arange(MIN_T, MAX_T):
            fileList_d.append(str(i)+"\\p"+str(i)+"_s10_"+str(j)+"sec.jpg")
    
    for item in fileList_a:
        if not (os.path.isfile(r'img\\'+str(item))):
            print("Not found", item)
            continue
        if train_set:
            shutil.copy(r'img\\'+str(item), r'cnn_train\0\\')
        else:
            shutil.copy(r'img\\'+str(item), r'cnn_test\0\\')
    
    for item in fileList_d:
        if not (os.path.isfile(r'img\\'+str(item))):
            print("Not found", item)
            continue
        if train_set:
            shutil.copy(r'img\\'+str(item), r'cnn_train\1\\')
        else:
            shutil.copy(r'img\\'+str(item), r'cnn_test\1\\')
			
			
            
def calibration(detector, predictor, cap = cv2.VideoCapture(0)):
    """Helper function for determing mean and std"""
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,400)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    data = []
    cap = cap

    while True:
        # Getting out image by webcam 
        _, image = cap.read()
        # Converting the image to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(image, 0)

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            data.append(shape)
            cv2.putText(image,"Calibrating...", bottomLeftCornerOfText, font, fontScale, fontColor,lineType)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Show the image
        cv2.imshow("Output", image)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    
    features_test = []
    for d in data:
        eye = d[36:68]
        ear = eye_aspect_ratio(eye)
        mar = mouth_aspect_ratio(eye)
        cir = circularity(eye)
        mouth_eye = mouth_over_eye(eye)
        features_test.append([ear, mar, cir, mouth_eye])
    
    features_test = np.array(features_test)
    x = features_test
    y = pd.DataFrame(x, columns=["EAR","MAR","Circularity","MOE"])
    df_means = y.mean(axis=0)
    df_std = y.std(axis=0)
    
    return df_means, df_std