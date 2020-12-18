import os
import librosa
import librosa.display
import pylab

genre_types = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
FRAME_SIZE = 2048
HOP_SIZE = 512

SAVE_PATH = "database/spectrograms/"

OFFSET = 5

class Spectrogram:
    def __init__(self):
        pass

    def create_database(self, enableOffset=False):
        index_list = []

        # get the index list
        for i in range(100):
            if i < 10:
                index = '0' + str(i)
            else:
                index = str(i)
            index_list.append(index)
        print('Index list created.')

        print("EnableOffset: ", enableOffset, "\nOffset: ", OFFSET)

        if enableOffset:
            path = SAVE_PATH + 'enableOffset'

            # create folder
            if not os.path.exists(path):
                os.makedirs(path)

            # for each gen_type
            for gen_type in genre_types:
                print('Gen type: ', gen_type, ' started.')

                # create folder along with its gen_type
                if not os.path.exists(path + gen_type):
                    os.makedirs(path + gen_type)

                # for each song
                for index in index_list:
                    # create path for the song
                    url = "database/Data/genres_original/" + gen_type + "/" + gen_type + '.000' + index + '.wav'
                    for i in range(0, 30, OFFSET):
                        if not os.path.exists(
                                path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png'):

                            try:
                                scale, sr = self.load(url, i, OFFSET)
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
                    if not os.path.exists(path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(index) + '.png'):
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
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

        librosa.display.specshow(
            Y_scale
        )

        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()

    def load(self, url, offset=0, duration=OFFSET):
        return librosa.load(url, offset=offset, duration=duration)
