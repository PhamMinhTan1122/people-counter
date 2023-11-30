import json
import os
import argparse


class LoadSong2Json:
    def __init__(self, args) -> None:
        self.args = args
        self.PATH_SONG = args["path_song"]
        self.JSON_FILE = args["path_json"]

    def parse_arguments(self):
        ap = argparse.ArgumentParser()
        ap.add_argument(
            "-s",
            "--path-song",
            default="./songs",
            required=False,
            help="path folder contain songs",
        )
        ap.add_argument(
            "-j",
            "--path-json",
            default="./songs.json",
            required=False,
            help="path json file to write id and id",
        )
        ap.add_argument(
            "-r",
            "--rename-song",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="--rename-song to rename file in sub folder PATH_SONG",
        )
        args = vars(ap.parse_args())
        return args

    def rename_song(self):
        i = 0
        for entry in os.scandir(self.PATH_SONG):
            if entry.name.endswith((".mp3", ".wav")):
                file_ext = os.path.splitext(entry.name)[1]
                new_name = f"song_{i}{file_ext}"
                os.rename(
                    os.path.join(self.check_slash(self.PATH_SONG), entry.name),
                    os.path.join(self.check_slash(self.PATH_SONG), new_name),
                )
                i += 1

    def check_slash(self, path):
        if path[-1] != "/":
            path += "/"
        path = path.replace("\\", "/")
        return path

    def load_song(self):
        json_list = []
        for i, entry in enumerate(os.scandir(self.PATH_SONG)):
            data = {
                "id": i + 1,
                "path": f"{self.check_slash(self.PATH_SONG) + entry.name}",
            }
            json_list.append(data)
        return json_list

    def write(self):
        if self.args["rename_song"]:
            self.rename_song()
        with open(self.JSON_FILE, "w") as file:
            json_list = self.load_song()
            json.dump(json_list, file)
            file.close()


if __name__ == "__main__":
    args = LoadSong2Json.parse_arguments(None)
    LoadSong2Json(args).write()
