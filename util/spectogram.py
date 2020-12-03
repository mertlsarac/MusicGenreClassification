import librosa, librosa.display
import os
import matplotlib.pyplot as plt

genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
FRAME_SIZE = 2048
HOP_SIZE = 512

SAVE_PATH = "database/spectograms/"
class Spectogram:
    def __init__(self):
        pass

    def create_database(self, enable_6s=False):
        index_list = []

        # get the index list
        for i in range(100):
            if i < 10:
                index = '0' + str(i)
            else:
                index = str(i)
            index_list.append(index)
        print('Index list created.')

        if enable_6s:
            for gen_type in genre_types:
                print('Gen type: ', gen_type, ' started.')
                os.makedirs(SAVE_PATH + gen_type)

                for index in index_list:
                    url = "database/Data/genres_original/" + gen_type + "/" + gen_type + '.000' + index + '.wav'
                    for i in range(0, 30, 6):
                        scale, sr = self.load(url, i, 6)
                        S_scale, Y_scale = self.create(scale)
                        self.save(Y_scale, sr, SAVE_PATH + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png')

                    print("Index: ", index, " created.")

        else:
            for gen_type in genre_types:
                print('Gen type: ', gen_type, ' started.')
                os.makedirs(SAVE_PATH + gen_type)

                for index in index_list:
                    url = "database/Data/genres_original/" + gen_type + "/" + gen_type + '.000' + index + '.wav'
                    scale, sr = self.load(url)
                    S_scale, Y_scale = self.create(scale)
                    self.save(Y_scale, sr, SAVE_PATH + gen_type + '/' + gen_type + '.000' + index + '.png')

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
