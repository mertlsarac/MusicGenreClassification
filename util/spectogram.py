import os

import librosa
import librosa.display
import matplotlib.pyplot as plt

genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
FRAME_SIZE = 2048
HOP_SIZE = 512

SAVE_PATH = "database/spectograms/"


class Spectogram:
    def __init__(self):
        pass

    def create_database(self, enable_5s=False):
        index_list = []

        # get the index list
        for i in range(100):
            if i < 10:
                index = '0' + str(i)
            else:
                index = str(i)
            index_list.append(index)
        print('Index list created.')

        print("Enable5s: ", enable_5s)

        if enable_5s:
            path = SAVE_PATH + 'enable_5s/'

            if not os.path.exists(path):
                os.makedirs(path)

            for gen_type in genre_types:
                print('Gen type: ', gen_type, ' started.')

                if not os.path.exists(path + gen_type):
                    os.makedirs(path + gen_type)

                for index in index_list:
                    url = "database/Data/genres_original/" + gen_type + "/" + gen_type + '.000' + index + '.wav'
                    for i in range(0, 30, 5):
                        if not os.path.exists(
                                path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png'):

                            try:
                                scale, sr = self.load(url, i, 5)
                                S_scale, Y_scale = self.create(scale)
                                self.save(Y_scale, sr,
                                          path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png')
                            except:
                                pass

                    print("Index: ", index, " created.")


        else:
            path = SAVE_PATH + 'default/'
            if not os.path.exists(path):
                os.makedirs(path)
            for gen_type in genre_types:
                print('Gen type: ', gen_type, ' started.')
                if not os.path.exists(path + gen_type):
                    os.makedirs(path + gen_type)

                for index in index_list:
                    if not os.path.exists(path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png'):
                        url = "database/Data/genres_original/" + gen_type + "/" + gen_type + '.000' + index + '.wav'

                        try:
                            scale, sr = self.load(url)
                            S_scale, Y_scale = self.create(scale)
                            self.save(Y_scale, sr, path + gen_type + '/' + gen_type + '.000' + index + '.png')

                        except:
                            pass

                    print("Index: ", index, " created.")

    def create(self, scale):
        S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_scale = librosa.amplitude_to_db(abs(S_scale))
        return S_scale, Y_scale

    def save(self, Y_scale, sr, save_path):
        plt.figure(figsize=(25, 10))

        librosa.display.specshow(
            Y_scale,
            sr=sr,
            hop_length=HOP_SIZE,
            x_axis="time",
            y_axis="linear"
        )

        plt.savefig(save_path, bbox_inches=None, pad_inches=0)
        plt.close()

    def load(self, url, offset=0, duration=6):
        return librosa.load(url, offset=offset, duration=duration)
