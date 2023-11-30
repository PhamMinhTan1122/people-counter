from os import environ
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from utils.thread import ThreadingClass
from imutils.video import VideoStream
import numpy as np
import threading
import argparse
import schedule
import imutils
import random
import pygame
import dlib
import time
import json
import cv2

class PeopleCounter:
    def __init__(self, args):
        self.args = args
        self.net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        self.vs = None
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackableObjects = {}
        self.totalFrames = 0
        self.totalUp = 0
        self.totalDown = 0
        self.cache = {}
        self.config = None
        self.song = None
        self.W = None
        self.H = None
        self.path_cache = "./cache.json"
        self.path_config = "./utils/config.json"
        self.path_songs_json = "./songs.json"
        self.start_time = time.time()
        self.music_flag = False
    def parse_arguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt", required=False,
            help="path to Caffe 'deploy' prototxt file")
        ap.add_argument("-m", "--model", required=True,
            help="path to Caffe pre-trained model")
        ap.add_argument("-i", "--input", type=str,
            help="path to optional input video file")
        # confidence default 0.4
        ap.add_argument("-c", "--confidence", type=float, default=0.4,
            help="minimum probability to filter weak detections")
        ap.add_argument("-s", "--skip-frames", type=int, default=30,
            help="# of skip frames between detections")
        args = vars(ap.parse_args())
        return args
    def reset_cache(self):
        with open(self.path_cache, "w") as file:
            data = {"totalUp": 0, "totalDown": 0, "total": 0, "soundFlag": False}
            json.dump(data, file)
            file.close()

    def read_file_json(self, key: str, value):
        with open(self.path_cache, "r") as file:
            data = json.load(file)
            data[key] = value
            return data

    def write_file_json(self, data):
        with open(self.path_cache, "w") as file:
            json.dump(data, file)
            file.close()

    def load_file_config(self):
        with open(self.path_config, "r") as file:
            self.config = json.load(file)

    def check_input(self):
        if not self.args.get("input", False):
            self.vs = VideoStream(int(self.config["url"])).start()
            time.sleep(2.0)
        else:
            self.vs = cv2.VideoCapture(self.args["input"])

    def check_cofig_thread(self):
        if self.config["Thread"]:
            self.vs = ThreadingClass(int(self.config["url"]))

    def text_display(self, frame, text: str, key_data: str, pos: int, index: int, index1: int, color: tuple):
        with open(self.path_cache, "r") as file:
            data = json.load(file)
            cv2.putText(frame, f"{text}: {data[key_data]}", (pos, self.H - ((index * 20) + index1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
    def play_song(self):
        with open(self.path_songs_json, "r") as file:
            song_data = json.load(file)
        last_id = len([d["id"] for d in song_data])
        while True:
            time.sleep(0.2)
            with open(self.path_cache, "r") as file:
                data = json.load(file)
            if data["soundFlag"]:
                # print("PS: ", self.music_flag)
                rand_id = random.randint(1, last_id)
                for d in song_data:
                    if d["id"] == rand_id:
                        print(d["path"])
                        self.song = d["path"]
                        break  # exit the loop after finding the random song
                print("playing")
                pygame.mixer.music.load(self.song)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.3)
                    with open(self.path_cache, "r") as file:
                        data = json.load(file)
                    # If data["soundFlag"] is False, stop the music and break out of the loop
                    if data["soundFlag"] is False:
                        print("stopped")
                        pygame.mixer.music.fadeout(5000)
                        # break
                        # break
                data = self.read_file_json("soundFlag", False)
                self.write_file_json(data)
            else:
                pygame.mixer.music.fadeout(5000)
                pygame.mixer.music.stop()
            

    def peoplCounter(self):
        
        self.args = self.parse_arguments()
        self.reset_cache()
        self.net = cv2.dnn.readNetFromCaffe(self.args["prototxt"], self.args["model"])
        self.load_file_config() 
        self.check_input() 
        self.check_cofig_thread()
        while True:
            frame = self.vs.read()
            frame = frame[1] if self.args.get("input", False) else frame

            if self.args["input"] is not None and frame is None:
                break
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]
            
            rects = []
            if self.totalFrames % self.args["skip_frames"] == 0:
                trackers = []
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.args["confidence"]:
                        box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                        (startX, startY, endX, endY) = box.astype("int")
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
            else:
                for tracker in trackers:
                    tracker.update(rgb)
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    rects.append((startX, startY, endX, endY))
            cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 0, 0), 3)
            objects = self.ct.update(rects)
            for (objectID, centroid) in objects.items():
                to = self.trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if direction < 0 and centroid[1] < self.H // 2:
                            self.totalUp += 1
                            data = self.read_file_json("totalUp", self.totalUp)
                            self.write_file_json(data=data)
                            to.counted = True
                        elif direction > 0 and centroid[1] > self.H // 2:
                            self.totalDown += 1
                            data = self.read_file_json("totalDown", self.totalDown)
                            self.write_file_json(data=data)
                            data = self.read_file_json("total", self.totalDown - self.totalUp)
                            self.write_file_json(data=data)
                            # data = self.read_file_json("soundFlag", self.totalDown - self.totalUp >= 1) # calculate the music flag
                            flag = self.totalDown - self.totalUp >= 1
                            time.sleep(0.1)
                            with open(self.path_cache, "r") as file:
                                data = json.load(file)
                                file.close()
                            if flag != data["soundFlag"]:
                                data["soundFlag"] = flag
                                self.write_file_json(data)
                                print("saved")
                        
                            
                            to.counted = True
                self.trackableObjects[objectID] = to
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
            self.text_display(frame, "Exit", "totalUp", 10, 1, 20, (0,0,0))
            self.text_display(frame, "Enter", "totalDown", 10, 2, 20, (0,0,0))
            self.text_display(frame, "Total", "total", 265, 1, 60, (255,255,255))
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                pygame.QUIT()
                break
            self.totalFrames += 1
            if self.config["Timer"]:
                end_time = time.time()
                num_seconds = (end_time - self.start_time)
                if num_seconds > 216000:
                    break
        if self.config["Thread"]:
            self.vs.release()
        cv2.destroyAllWindows()
    
        if self.config["Scheduler"]:
            schedule.every().day.at("06:00").do(self.peoplCounter)
            while True:
                schedule.run_pending()
        else:
            PeopleCounter()
if __name__ == "__main__":
    pygame.mixer.init()

    args = PeopleCounter.parse_arguments(None)
    Count_thread = threading.Thread(target=PeopleCounter(args).peoplCounter)
    Song_thread = threading.Thread(target=PeopleCounter(args).play_song)
    time.sleep(3)
    Count_thread.start()
    Song_thread.start()

    Count_thread.join()
    Song_thread.join()
    pygame.mixer.quit()
    pygame.mixer.music.stop()
    pygame.quit()
    pygame.QUIT
