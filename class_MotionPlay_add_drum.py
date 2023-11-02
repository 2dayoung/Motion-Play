import threading
import pygame
from threading import Thread, Lock
import mido
import cv2
import simpleaudio as sa
import mediapipe as mp
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from tkinter import *
from tkinter import messagebox
# from multiprocessing import Process

#======================================================================
#함수
class Instrument() :
    def __init__(self):
        self.mouse_click_count  = 0
        self.roi_points = []
        self.mouse_clicked = False

        

    def is_object_in_area(self,object_location, area):
        x, y = object_location
        start_x, start_y, width, height = area
        return start_x <= x < start_x + width and start_y <= y < start_y + height

    # def mouse_callback(self,event, x, y): #, flags, param):  
    #         if event == cv2.EVENT_LBUTTONDOWN:
    #             if self.mouse_click_count < 10:
    #                 # 좌표 출력
    #                 print(f"Clicked: ({x}, {y})")
    #                 self.roi_points.append((x, y))
    #                 self.mouse_click_count += 1
                    
    #             if self.mouse_click_count == 2:
    #                 self.mouse_click_count=0           
    #                 self.divide_area()
    #                 self.mouse_clicked=True
    #                 print("true")
    #                 self.roi_points=[]

    def start(self):
        print("Instrument started")
#======================================================================
class Piano(Instrument):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                        max_num_hands=2, 
                        min_detection_confidence=0.7, 
                        min_tracking_confidence=0.7)
        self.mouse_click_count=0
        self.areas = []
        self.black_key=[]


        self.exit_event = threading.Event()
        self.mouse_click_count  = 0
        self.roi_points = []
        self.mouse_clicked = False
        self.outport = mido.open_output()
        
        # midi filename 생성
        self.midi_filename= []
        notes_make_file = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76,77,79,81,83]
        for i, note in enumerate(notes_make_file):    
            self.midi_filename.append(str(note) + '.mid')
        print(self.midi_filename)


        self.pressed_key = []
        self.one_hand_event = []
        self.two_hand_event = []      
        self.prev_one_hand = None 
        self.prev_two_hand = None

        self.num_key = 0

            
    def get_black_key(self, num):
        black=[0, 1, 3, 4, 5, 7, 8, 10, 11, 12]
        return black
    
    def get_C4_key(self, num):
        return 7
    
    def divide_area(self):        
        x1 = self.roi_points[0][0]
        x2 = self.roi_points[1][0]
        y1=  self.roi_points[0][1]
        n=14
        width = x2 - x1
        area_width = width // n
        
        for i in range(n):
            area_x = x1 + (i * area_width)
            area_y = y1
            area_w = area_width
            area_h = 50
            if i == n - 1:
                area_w = width - (i * area_width)
            self.areas.append([area_x, area_y, area_w, area_h])

    def mouse_callback(self,event, x, y, flags, param):  
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mouse_click_count < 10:
                # 좌표 출력
                print(f"Clicked: ({x}, {y})")
                self.roi_points.append((x, y))
                self.mouse_click_count += 1
                
            if self.mouse_click_count == 2:
                self.mouse_click_count=0           
                self.divide_area()
                self.mouse_clicked=True
                print("true")
                self.roi_points=[]

    # play_key 함수
    def play_key(self, midi_filename):
        mid = mido.MidiFile(midi_filename)
        for message in mid.play():
            self.outport.send(message)
        print(midi_filename)


    def cam(self):
    
        cap = cv2.VideoCapture(0)

        cv2.namedWindow("piano")
        cv2.setMouseCallback("piano", self.mouse_callback)

        while not self.exit_event.is_set() :
            
            # 웹캠에서 프레임 읽기& 좌우반전
            ret, frame = cap.read()
            result = cv2.flip(frame, 1)

            # 창 크기 조정
            cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 현재 캠의 가로 해상도
            cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 현재 캠의 세로 해상도           
            width = cap_width * 1.5
            height = cap_height * 1.5           
            result = cv2.resize(result, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
            
            cv2.imshow("piano", result)
            if self.mouse_clicked:
                # 프레임을 RGB로 변환
                image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                
                # Mediapipe Hand Landmark 모델을 사용하여 이미지 처리
                results = self.hands.process(image)   
                     
                # 손이 한손만 있을 때
                if results.multi_hand_landmarks:
                    if len(results.multi_hand_landmarks) == 1:  
                        handLms = results.multi_hand_landmarks[0]
                        
                    # 각 손가락 끝의 랜드마크 좌표 추출
                        fingertips = []
                        for finger_tip_id in [4, 8, 12, 16, 20]:
                            lm = handLms.landmark[finger_tip_id]
                            h, w, c = result.shape   #좌표가 0~1값임.화면상의 픽셀 좌표로 변환하기 위해 이미지의 크기필요 C는 채널
                            cx, cy = int(lm.x *w), int(lm.y*h)
                            fingertips.append((cx,cy))

                                    
                        for location in  fingertips:
                            # 손가락 끝에 원 그리기              
                            cv2.circle(result, location, 5, (255, 0, 0), -1)  
                            #영역에 들어왔는지 검사            
                            for i in range(len(self.areas)):                   
                                if self.is_object_in_area(location, self.areas[i]):
                                    event = i
                                    self.one_hand_event.append(event)
                                    
                        #중복요소 제거
                        self.one_hand_event=set(self.one_hand_event)
                        self.one_hand_event=list(self.one_hand_event)

                        #전 array와 다르면 global변수로 전달 
                        if self.prev_one_hand != self.one_hand_event:
                            self.pressed_key = self.one_hand_event                                       
                            #print (self.one_hand_event)
                            pass

                        self.prev_one_hand = self.one_hand_event 
                        self.one_hand_event = [] 

                    # 양손다 들어왔을때    
                    else :      
                        pass
                        handLms1 = results.multi_hand_landmarks[0]  # 왼손
                        handLms2 = results.multi_hand_landmarks[1]  # 오른손

                    # 각 손가락 끝의 랜드마크 좌표 추출 (왼손)
                        fingertips1 = []
                        for finger_tip_id in [4, 8, 12, 16, 20]:
                            lm = handLms1.landmark[finger_tip_id]
                            h, w, c = result.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            fingertips1.append((cx, cy))
                        
                    # 각 손가락 끝의 랜드마크 좌표 추출 (오른손)
                        fingertips2 = []
                        for finger_tip_id in [4, 8, 12, 16, 20]:
                            lm = handLms2.landmark[finger_tip_id]
                            h, w, c = result.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            fingertips2.append((cx, cy))
                        #왼손
                        for location in  fingertips1:       
                            cv2.circle(result, location, 5, (255, 0, 0), -1)  
                            #영역안에 들어왔을 경우 array에 추가 
                            for i in range(len(self.areas)):                   
                                if self.is_object_in_area(location, self.areas[i]):
                                    event = i
                                    self.two_hand_event.append(event)
                        #오른손
                        for location in  fingertips2:              
                            cv2.circle(result, location, 5, (255, 0, 0), -1)  
                            #영역안에 들어왔을 경우 array에 추가
                            for i in range(len(self.areas)):                   
                                if self.is_object_in_area(location, self.areas[i]):
                                    event = i
                                    self.two_hand_event.append(event)            
                                    

                        #중복요소 제거
                        self.two_hand_event=set(self.two_hand_event)
                        #리스트로 변경
                        self.two_hand_event=list(self.two_hand_event)

                        if self.prev_two_hand != self.two_hand_event:                        
                            self.pressed_key = self.two_hand_event                                       
                            #print ("양손",self.two_hand_event)
                            pass

                        self.prev_two_hand = self.two_hand_event 
                        self.two_hand_event = [] 
                
                #영역에 사각형으로 표시 
                for i, area in enumerate(self.areas):
                    cv2.rectangle(result, (area[0], area[1]),(area[0]+area[2],area[1]+area[3]),(0,0,0), 2)
                    if i in self.get_black_key(self.num_key) :
                        cv2.circle(result, (area[0]+area[2], area[1]+7), int(area[2]*0.3), (0,0,0), -1)
                    if i == self.get_C4_key(self.num_key) :
                        cv2.putText(result,"C4",(area[0]+5,area[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                for idx in self.pressed_key:
                    area = self.areas[idx]
                    cv2.rectangle(result, (area[0], area[1]), (area[0] + area[2], area[1] + area[3]), (200, 200, 25), thickness=cv2.FILLED)
            cv2.imshow("piano", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_event.set()  # 종료 신호 설정
                break

        cap.release()
        cv2.destroyAllWindows()

    def start(self):
        print("Piano started")

 
    def make_threads(self):
            # Thread list 생성
        threads_list = []

        # 카메라 스레드 추가
        cam_thread  =  threading.Thread(target=self.cam)
        cam_thread.start()
        prev_pressed_key = []

        while not self.exit_event.is_set() :
            while self.pressed_key != [] and self.pressed_key != prev_pressed_key :
                #print(self.pressed_key)
                
                threads_list = []
                #different_notes = [num for num in self.pressed_key if num not in prev_pressed_key]
                #print("diff",different_notes)
                # MIDI 파일 재생 스레드 추가
                for i in self.pressed_key:
                #  /   midi_thread = threading.Thread(target=self.play_key, args=(self.midi_filename[i],))
                    midi_thread = threading.Thread(target=self.play_key, args=(self.midi_filename[i],))
                    threads_list.append(midi_thread)

                for tread in threads_list:
                    tread.start()

                for tread in threads_list:
                    tread.join()
                prev_pressed_key = self.pressed_key 

        cam_thread.join()
        print('End')

class Drum(Instrument) :
    def __init__(self):
        self.selected_drum_sounds = []
        self.root = None
        self.cap = None
        self.current_rectangles = []
        self.prev_event = [0, 0]
        self.drum_sound_window = None

        self.mouse_drag_started = False
        self.roi_points = []
        self.mouse_click_count = 0
        self.rectangles = []  
        
        self.drum_sounds = ["ride", "crash", "hihat", "low_tom", "mid_tom", "high_tom", "kick", "snare", "sticks"]
        self.drum_sound_objects = [sa.WaveObject.from_wave_file(f'drum/{sound}.wav') for sound in self.drum_sounds]
        self.selected_drum_sound_indices = []  

    def choose_drum_sounds(self):
        if self.drum_sound_window is not None:
            self.drum_sound_window.destroy()  
        self.drum_sound_window = Toplevel(self.root)
        self.drum_sound_window.title("드럼 소리 선택")

        drum_sound_listbox = Listbox(self.drum_sound_window, selectmode=MULTIPLE)
        for sound in self.drum_sounds:
            drum_sound_listbox.insert(END, sound)
        drum_sound_listbox.pack(padx=20, pady=20)

        confirm_button = Button(self.drum_sound_window, text="확인", command=lambda: self.set_selected_drum_sounds(drum_sound_listbox.curselection()))
        confirm_button.pack()

    def set_selected_drum_sounds(self, selected_indices):
        if self.drum_sound_window is not None:
            self.drum_sound_window.destroy()  

        self.selected_drum_sound_indices = selected_indices
        self.selected_drum_sounds = [self.drum_sounds[i] for i in selected_indices]
        if not self.selected_drum_sounds:
            messagebox.showwarning("경고", "드럼 소리를 선택하세요.")
        else:
            self.start_webcam()

    def clear_rectangles(self):
        self.current_rectangles = []

    def get_color_for_sound(self, sound):
        sound_colors = {
            "ride": (0, 0, 255),    
            "crash": (0, 255, 0),   
            "hihat": (255, 0, 0),   
            "low_tom": (0, 255, 255),  
            "mid_tom": (255, 255, 0),  
            "high_tom": (255, 0, 255),  
            "kick": (128, 0, 128),  
            "snare": (0, 128, 128),  
            "sticks": (128, 128, 0)  
        }
        return sound_colors.get(sound, (255, 255, 255))  

    def play_drum_sound(self, sound_idx):
        drum_sound = self.drum_sound_objects[sound_idx]
        play_obj = drum_sound.play()
        play_obj.wait_done()

    event = None
    prev_event = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mouse_click_count < 2:
                print(f"Clicked: ({x}, {y})")
                self.roi_points.append((x, y))
                self.mouse_click_count += 1

            if self.mouse_click_count == 2:
                self.mouse_click_count = 0
                if len(self.roi_points) == 2:  
                    x1, y1 = self.roi_points[0]
                    x2, y2 = self.roi_points[1]
                    self.roi_points = []

                if self.selected_drum_sounds and len(self.current_rectangles) < len(self.selected_drum_sounds):
                    self.current_rectangles.append(((x1, y1, x2, y2), self.selected_drum_sounds[len(self.current_rectangles)]))
                    self.mouse_drag_started = True

    def start(self):
        self.clear_rectangles()
        self.choose_drum_sounds()
        print("Drum started")

    def start_webcam(self):
        self.clear_rectangles()
        self.cap = cv2.VideoCapture(1)
        cv2.namedWindow('img')
        cv2.setMouseCallback('img', self.mouse_callback)

        # 드럼 소리를 재생할 오디오 파일 경로
        drum_sounds = [self.drum_sound_objects[i] for i in self.selected_drum_sound_indices]

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (800, 600))

            # 이미지를 BGR에서 HSV로 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            img = frame.copy()

            for rectangle, sound in self.current_rectangles:
                x1, y1, x2, y2 = rectangle
                color = self.get_color_for_sound(sound)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.imshow('img', img)

            lower_blue = (90, 100, 100)
            upper_blue = (120, 255, 255)
            # HSV 이미지에서 색상 범위에 해당하는 영역을 이진화합니다.
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 잡음 제거를 위한 모폴로지 연산
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("closing ",closing )

            # 객체 검출
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 두 개의 가장 큰 객체만 추출
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            if contours and len(self.current_rectangles) > 0:
                # 객체 위치 추출
                event = [-1, -1]  # 두 개의 객체에 대한 event 초기화
                for i, c in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(c)
                    cx = x + w // 2
                    cy = y + h // 2

                    for rectangle, sound in self.current_rectangles:
                        x1, y1, x2, y2 = rectangle
                        if x1 < cx < x2 and y1 < cy < y2:
                            event[i] = self.selected_drum_sounds.index(sound) + 1  

                            # 그리고 해당 드럼 사각형을 초록색으로 표시
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 원 중심 좌표를 사용하여 소리 재생
                    if self.rectangles:
                        for idx, rectangle in enumerate(self.rectangles):
                            x1, y1, x2, y2 = rectangle
                            if x1 < cx < x2 and y1 < cy < y2:
                                event[i] = idx + 1  
                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 두 개의 event 값을 독립적으로 처리
                for event_idx, event_value in enumerate(event):
                    if event_value != -1:
                        if event_value == self.prev_event[event_idx]:
                            pass
                        else:
                            drum_sound = drum_sounds[event_value - 1]  
                            play_obj = drum_sound.play()
                            play_obj.wait_done()
                            self.prev_event[event_idx] = event_value

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                if self.current_rectangles:
                    self.current_rectangles.pop()
            elif key == ord('2'):
                self.current_rectangles = []  
            elif key == ord('3'):
                self.current_rectangles = []  
                self.cap.release()
                self.choose_drum_sounds() 

        self.cap.release()
        cv2.destroyAllWindows()
        

def start():
    piano = Piano()
    drum= Drum()

    root = Tk()
    root.title("Motion Play")

    # 윈도우 크기 설정
    window_width = 720
    window_height = 480
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

    # 라벨 추가
    label = Label(root, text="연주할 악기를 선택하세요", font=("함초롬돋음", 18))
    label.configure(font=("휴먼엑스포", 20))
    label.place(relx=0.5, rely=0.55, anchor="center")
    title = Label(root, text="Motion Play")
    title.configure(font=("Perpetua Titling MT", 50))
    title.place(relx=0.5, rely=0.3, anchor="center")

    # 버튼 추가
    piano_button = Button(root, text="피아노",command= piano.make_threads)
    piano_button.configure(font=("휴먼엑스포", 20))
    piano_button.place(relx=0.3, rely=0.7, anchor="center")
    piano_button.config(width=8, height=2)
   

    drum_button = Button(root, text="드럼",command= drum.start)
    drum_button.configure(font=("휴먼엑스포", 20))
    drum_button.place(relx=0.7, rely=0.7, anchor="center")
    drum_button.config(width=8, height=2)
    drum_button.config()

    root.mainloop()

if __name__ == '__main__':
    start()
