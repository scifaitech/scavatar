import os, sys
from sys import platform as _platform
import glob
import yaml
#import time
import requests
import streamlit as st
import numpy as np
import cv2

from videocaptureasync import VideoCaptureAsync
from arguments import opt
from utils import info, crop, pad_img, resize, TicToc
import camera_selector as cam_selector



# Where to split an array from face_alignment to separate each landmark
LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])

if _platform == 'darwin':
    if not opt.is_client:
        info('\nOnly remote GPU mode is supported for Mac (use --is-client and --connect options to connect to the server)')
        info('Standalone version will be available lately!\n')
        exit()

@st.cache
def is_new_frame_better(source, driving, predictor):
    global avatar_kp
    global display_string
    
    if avatar_kp is None:
        display_string = "No face detected in avatar."
        return False
    
    if predictor.get_start_frame() is None:
        display_string = "No frame to compare to."
        return True
    
    driving_smaller = resize(driving, (128, 128))[..., :3]
    new_kp = predictor.get_frame_kp(driving)
    
    if new_kp is not None:
        new_norm = (np.abs(avatar_kp - new_kp) ** 2).sum()
        old_norm = (np.abs(avatar_kp - predictor.get_start_frame_kp()) ** 2).sum()
        
        out_string = "{0} : {1}".format(int(new_norm * 100), int(old_norm * 100))
        display_string = out_string
        
        return new_norm < old_norm
    else:
        display_string = "No face found!"
        return False

@st.cache
def load_stylegan_avatar():
    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={'User-Agent': "My User Agent 1.0"}).content

    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize(image, (IMG_SIZE, IMG_SIZE))

    return image
@st.cache
def load_images(IMG_SIZE = 256):
    avatars = []
    filenames = []
    images_list = sorted(glob.glob(f'{opt.avatars}/*'))
    for i, f in enumerate(images_list):
        if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'):
            img = cv2.imread(f)
            if img is None:
                continue

            if img.ndim == 2:
                img = np.tile(img[..., None], [1, 1, 3])
            img = img[..., :3][..., ::-1]
            img = resize(img, (IMG_SIZE, IMG_SIZE))
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames

def change_avatar(predictor, new_avatar):
    global avatar, avatar_kp, kp_source
    avatar_kp = predictor.get_frame_kp(new_avatar)
    kp_source = None
    avatar = new_avatar
    predictor.set_source_image(avatar)


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    l = w * (1 - rw) // 2
    r = w - l
    u = h * (1 - rh) // 2
    d = h - u
    img = cv2.rectangle(img, (int(l), int(u)), (int(r), int(d)), color, thickness)
@st.cache
def kp_to_pixels(arr):
    '''Convert normalized landmark locations to screen pixels'''
    return ((arr + 1) * 127).astype(np.int32)
@st.cache
def draw_face_landmarks(img, face_kp, color=(20, 80, 255)):
    if face_kp is not None:
        img = cv2.polylines(img, np.split(kp_to_pixels(face_kp), LANDMARK_SLICE_ARRAY), False, color)

def print_help():
    info('\n\n=== Control keys ===')
    info('1-9: Change avatar')
    for i, fname in enumerate(avatar_names):
        key = i + 1
        name = fname.split('/')[-1]
        info(f'{key}: {name}')
    info('W: Zoom camera in')
    info('S: Zoom camera out')
    info('A: Previous avatar in folder')
    info('D: Next avatar in folder')
    info('Q: Get random avatar')
    info('X: Calibrate face pose')
    info('I: Show FPS')
    info('ESC: Quit')
    info('\nFull key list: https://github.com/alievk/avatarify#controls')
    info('\n\n')


def draw_fps(frame, fps, timing, x0=10, y0=20, ystep=30, fontsz=0.5, color=(255, 255, 255)):
    frame = frame.copy()
    cv2.putText(frame, f"FPS: {fps:.1f}", (x0, y0 + ystep * 0), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Model time (ms): {timing['predict']:.1f}", (x0, y0 + ystep * 1), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Preproc time (ms): {timing['preproc']:.1f}", (x0, y0 + ystep * 2), 0, fontsz * IMG_SIZE / 256, color, 1)
    cv2.putText(frame, f"Postproc time (ms): {timing['postproc']:.1f}", (x0, y0 + ystep * 3), 0, fontsz * IMG_SIZE / 256, color, 1)
    return frame


def draw_landmark_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "ALIGN FACES", (60, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame


def draw_calib_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255)):
    frame = frame.copy()
    cv2.putText(frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk)
    return frame

@st.cache
def select_camera(config):
    cam_config = config['cam_config']
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, 'r') as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config['cam_id']
    else:
        cam_frames = cam_selector.query_cameras(config['query_n_cams'])

        if cam_frames:
            if len(cam_frames) == 1:
                cam_id = list(cam_frames)[0]
            else:
                cam_id = cam_selector.select_camera(cam_frames, window="CLICK ON YOUR CAMERA")

            with open(cam_config, 'w') as f:
                yaml.dump({'cam_id': cam_id}, f)


    return cam_id


if __name__ == "__main__":
    st.title('_LIVE streaming with Scavatar_')
    predictor_args = {
        'config_path': 'vox-adv-256.yaml',
        'checkpoint_path': "vox-adv-cpk.pth.tar",
        'relative': opt.relative,
        'adapt_movement_scale': opt.adapt_scale,
        'enc_downscale': opt.enc_downscale
    }
    if opt.is_worker:
        from afy import predictor_worker
        predictor_worker.run_worker(opt.in_port, opt.out_port)
        sys.exit(0)
    elif opt.is_client:
        from afy import predictor_remote
        try:
            predictor = predictor_remote.PredictorRemote(
                in_addr=opt.in_addr, out_addr=opt.out_addr,
                **predictor_args
            )
        except ConnectionError as err:
            sys.exit(1)
        predictor.start()
    else:
        from afy import predictor_local
        predictor = predictor_local.PredictorLocal(
            **predictor_args
        )

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    st.markdown('Please select options below to proceed:')
    global display_string
    display_string = ""
    IMG_SIZE = 256

    cam_id = select_camera(config)

    if cam_id is None:
        exit(1)

    cap = VideoCaptureAsync(cam_id)
    cap.start()

    avatars, avatar_names = load_images()
    enable_vcam = not opt.no_stream
    cur_ava = 4    
    avatar = None
    change_avatar(predictor, avatars[cur_ava])
    passthrough = False
    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    overlay_alpha = 0.0
    preview_flip = False
    output_flip = False
    find_keyframe = False
    is_calibrated = False

    show_landmarks = False

    fps_hist = []
    fps = 0
    show_fps = False

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            FRAME_WINDOW = st.image([], width=None)
            #camera = cv2.VideoCapture(0)
        with col2:
            FRAME_WINDOW1 = st.image([], width=None)
        with col3:
            st.subheader('_Controls_:')
            if st.button('Reset Image'):
                predictor.reset_frames()
            if st.button('Next Image'):
                cur_ava += 1
                if cur_ava >= len(avatars):
                    cur_ava = 0
                passthrough = False
                change_avatar(predictor, avatars[cur_ava])
            if st.button('Previous Image'):
                cur_ava -= 1
                if cur_ava < 0:
                    cur_ava = len(avatars) - 1
                passthrough = False
                change_avatar(predictor, avatars[cur_ava])
            if st.button('Zoon In (Webcam)'):
                frame_proportion -= 0.05
                frame_proportion = max(frame_proportion, 0.1)                
            if st.button('Zoom Out (Webcam'):
                frame_proportion += 0.05
                frame_proportion = min(frame_proportion, 1.0)

        run = st.checkbox('Game-Mode', key='isOn')

    ret, frame = cap.read()
    stream_img_size = frame.shape[1], frame.shape[0]

    if enable_vcam:
        if _platform in ['linux', 'linux2']:
            try:
                import pyfakewebcam
            except ImportError:
                exit(1)

            stream = pyfakewebcam.FakeWebcam(f'/dev/video{opt.virt_cam}', *stream_img_size)
        else:
            enable_vcam = False
            # log("Virtual camera is supported only on Linux.")
        
        # if not enable_vcam:
            # log("Virtual camera streaming will be disabled.")

    #cv2.namedWindow('cam', cv2.WINDOW_GUI_NORMAL)
    #cv2.moveWindow('cam', 500, 250)

    print_help()
    
    vid = cv2.VideoCapture("avatars/result.mp4")
    #try:
    frame_counter = 0
    while run:

            #run = st.checkbox('Loading...', key='isOn')
        #vid = cv2.VideoCapture("avatars/result.mp4")
        successful_frame_read,vidframe = vid.read()
        frame_counter += 1

        if frame_counter == vid.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        #_, frame = camera.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[..., ::-1]
        frame_orig = frame.copy()
        frame, (frame_offset_x, frame_offset_y) = crop(frame, p=frame_proportion, offset_x=frame_offset_x, offset_y=frame_offset_y)
        frame = resize(frame, (IMG_SIZE, IMG_SIZE))[..., :3]
        tt = TicToc()

        timing = {
            'preproc': 0,
            'predict': 0,
            'postproc': 0
            }

        green_overlay = False

        tt.tic()

            

        if find_keyframe:
            if is_new_frame_better(avatar, frame, predictor):
                #log("Taking new frame!")
                green_overlay = True
                predictor.reset_frames()

       
        timing['preproc'] = tt.toc()
        FRAME_WINDOW.image(vidframe)
        #cv2.imshow("ScifAI Video Face Detector" , vidframe)
        #key = cv2.waitKey(10)

        
        out = predictor.predict(vidframe)

        #FRAME_WINDOW.image(frame)
        FRAME_WINDOW1.image(out)
            #run = ('Game-Mode started')


            #fps_hist.append(tt.toc(total=True))
            #if len(fps_hist) == 10:
            #    fps = 10 / (sum(fps_hist) / 1000)
             #   fps_hist = []
    
    #except KeyboardInterrupt:
       # print("stopped")
        #log("main: user interrupt")

    #log("stopping camera")
    #cap.stop()

   # cv2.destroyAllWindows()

    #if opt.is_client:
    # log("stopping remote predictor")
        #predictor.stop()

    #log("main: exit")
