from models.TransFaceGCNEmotion import TransFaceGCNEmotion
from torchvision import transforms
from deepface import DeepFace
import numpy as np
from PIL import Image
import cv2
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# The face extraction tool facenet_pytorch uses backends
backends = [
    'retinaface', 'mtcnn', 'opencv',
    'ssd', 'dlib', 'fastmtcnn',
    'mediapipe', 'yolov11s', 'yolov8',
    'yolov11n', 'yolov11m', 'yunet',
    'centerface',
]

def face_detection_to_frame(frame, video_path, frame_index, detector_backend):
    data = DeepFace.extract_faces(img_path=frame, detector_backend=detector_backend, align=True)
    if len(data) == 0:
        raise ValueError(f"【video {video_path} => the {(frame_index + 1)}frame is no human face.")
    face_frame = data[0]['face']
    face_frame = np.array(Image.fromarray((face_frame * 255).astype(np.uint8)))
    return face_frame



if __name__ == '__main__':
    crop_face_frame_size = 112
    frame_transform = transforms.Compose([
        transforms.Resize((crop_face_frame_size, crop_face_frame_size)),
        transforms.ToTensor(),

    ])
    video_path  = './video/AN_WILTY_EP15_lie12.mp4'
    face_frame_num = 32
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"【The video file cannot be opened => Video address => {video_path}】")

    # Calculate the total number of frames in the video
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # TODO Note that we have observed a phenomenon in the dataset
    '''
        Often, in the captured videos, there are situations where other faces appear at the beginning and the end. This can lead to the collection of some non-test subject faces during the process.
        Therefore, two strategies should be considered when collecting frames.
        1. Based on the center frame of the video, collect the face_frame_num/2 frame on the left and the face_frame_num/2 frame on the right.
        2. Exclude the first 10% and the last 10% frames.
        However, during the collection process, there may be specific frames where the test subject's face does not appear, which requires optimization
        Frame index for pre-collected faces
    '''
    start = (total_frame // 10)
    end = total_frame - (total_frame // 10)
    frame_index_list = np.linspace(start, end, face_frame_num, dtype=int)
    emotion_list = []
    face_frame_list = []
    for index, frame_index in enumerate(frame_index_list):
        is_face = False
        # 1.Polling face detection device
        tools_index = 0
        # 2.Index frame +1 reset detector index whether index frame switching is initiated for the first time
        pre = 1
        cur_frame_index = frame_index
        while not is_face:
            video.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_index)
            ret, frame = video.read()
            if ret:
                # ======================== face frame
                try:
                    # Extract face frames
                    face_frame = face_detection_to_frame(frame, video_path, cur_frame_index,
                                                         detector_backend=backends[tools_index])
                    # Is there a human face?
                    is_face = True
                    face_frame_list.append(frame_transform(Image.fromarray(face_frame)))
                except Exception as e:
                    print(
                        f"【video {video_path} => calculate the expression state  => the{(frame_index + 1)} frame cannot detect the clear expression state => {e}】")
                # ========================Expression status=========
                if is_face:
                    try:
                        # calculate the expression status labels as respectively ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] ,enforce_detection=False
                        emotion = \
                        DeepFace.analyze(img_path=face_frame, actions=['emotion'], detector_backend=backends[tools_index])[
                                0]['emotion']
                        emotion_index = np.argmax(list(emotion.values()))
                        emotion_list.append(emotion_index)
                    except Exception as e:
                        print(
                                f"【video {video_path} => calculate the expression state  => the{(frame_index + 1)} frame cannot detect the clear expression state => {e}】")
                        # If the expression status cannot be detected clearly, it is uniformly set to None
                        emotion_list.append(None)
            else:
                raise ValueError(f"【video {video_path} => the {(frame_index + 1)} frame not exist 】")
            # 2.Index frame +1 reset detector index
            if tools_index >= len(backends):
                # If it happens to be the first frame, move forward to obtain a new frame
                if index == 0:
                    # If the face cannot be detected in the first frame, directly reset the starting frame
                    cur_frame_index = cur_frame_index + 1
                else:
                    if pre <= index:
                        # If it is an intermediate frame, obtain the pre-pre frame
                        cur_frame_index = frame_index_list[index - pre]
                        pre = pre + 1
                    else:
                        # find the next frame
                        cur_frame_index = cur_frame_index + 1
                print(
                        f"【video {video_path} => the {(frame_index + 1)} frame replace => make a {cur_frame_index + 1} frame compensation】")
                tools_index = 0
    video.release()


    face_frame_list = torch.stack(face_frame_list).unsqueeze(0).to(device)

    model = TransFaceGCNEmotion(image_size=crop_face_frame_size, frame=face_frame_num).to(device)
    weight = './weight/end-deception-Dolos.pt'
    model.load_state_dict(torch.load(weight))
    model.eval()

    result= model(face_frame_list)
    # Obtain the prediction results and probabilities
    batch_preds = torch.argmax(result, 1).cpu().numpy()
    if batch_preds == 1:
        print(f'The person in the current video is engaged in deceptive behavior.')
    else:
        print(f'The person in the current video is behaving normally.')