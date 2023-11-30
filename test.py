import json
import pygame
import random
import threading
import time

class TestPlaySong():
    def __init__(self) -> None:
        pass
    def reset_cache(self):
        with open("./cache.json", "w") as file:
            data = {"totalUp": 0, "totalDown": 0, "total": 0, "soundFlag": False}
            json.dump(data, file)
            file.close()
    def read_file_json(self, key: str, value):
        with open("./cache.json", "r") as file:
            data = json.load(file)
            data[key] = value
            return data
    def write_file_json(self, data):
        with open("./cache.json", "w") as file:
            json.dump(data, file)
            file.close()
    def play_song(self):
        with open("./songs.json", "r") as file:
            song_data = json.load(file)
        last_id = len([d["id"] for d in song_data])

        while True:
            with open("./cache.json", "r") as file:
                data = json.load(file)
                file.close
            # print(f'AFTER WHILE: {data["soundFlag"]}')
            # print(data["soundFlag"])
            if data["soundFlag"]:
                rand_id = random.randint(1, last_id)
                print(rand_id)
                for d in song_data:
                    if d["id"] == rand_id:
                        print(d["path"])
                        self.song = d["path"]
                        break  # exit the loop after finding the random song
                pygame.mixer.music.load(self.song)
                pygame.mixer.music.play()
                print("playing")
                while pygame.mixer.get_busy():
                    pass
                print(f'AFTER PS: {data["soundFlag"]}')
                data = self.read_file_json("soundFlag", False)
                self.write_file_json(data)
    def detect(self):
        count = 0
        while True:
            if count > 20:
                data = self.read_file_json("soundFlag", True)
                self.write_file_json(data)
                break
            count+=1
        time.sleep(5)
        print("CONTINUE")

if __name__ == "__main__":
    pygame.init()
    TestPlaySong().reset_cache()
    count_thread = threading.Thread(target=TestPlaySong().detect())
    play_thread = threading.Thread(target=TestPlaySong().play_song())

    count_thread.start()
    play_thread.start()

    count_thread.join()
    play_thread.join()

    pygame.quit()
