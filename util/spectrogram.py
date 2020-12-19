import os
import librosa
import librosa.display
import pylab
from tqdm.auto import tqdm

class Spectrogram:
    def __init__(self, frame_size, hop_size, genre_types, data_duration, split_data, number_of_data_for_each_gen):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.genre_types = genre_types
        self.data_duration = data_duration
        self.split_data = split_data
        self.number_of_data_for_each_gen = number_of_data_for_each_gen

    def create_database(self, data_path, save_path):
        index_list = []

        # get the index list
        for i in range(100):
            if i < 10:
                index = '0' + str(i)
            else:
                index = str(i)
            index_list.append(index)
        print("Split data: ", self.split_data, " Duration: ", self.data_duration)

        # If split_data enable
        if self.split_data:
            path = save_path + 'enableOffset/'

            # create folder
            if not os.path.exists(path):
                os.makedirs(path)

            # for each gen_type
            for gen_type in self.genre_types:
                pbar = tqdm(total=self.number_of_data_for_each_gen, position=0, leave=True)
                pbar.set_description('Gen type: ' + gen_type + ' started')
                # create folder along with its gen_type
                if not os.path.exists(path + gen_type):
                    os.makedirs(path + gen_type)

                # for each song
                for index in index_list:
                    # create path for the song
                    url = data_path + gen_type + "/" + gen_type + '.000' + index + '.wav'
                    for i in range(0, 30, self.data_duration):
                        if not os.path.exists(
                                path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png'):

                            try:
                                scale, sr = self.load(url, self.data_duration, i)
                                S_scale, Y_scale = self.create(scale)
                                self.save(Y_scale, sr,
                                          path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(i) + '.png')
                            except:
                                print('Error occured in creating spectrogram.')

                    #print("Index: ", index, " created.")
                    pbar.update()

        else:
            path = save_path + 'default/'
            if not os.path.exists(path):
                os.makedirs(path)
            for gen_type in self.genre_types:
                pbar = tqdm(total=self.number_of_data_for_each_gen, position=0, leave=True)
                pbar.set_description('Gen type: ' + gen_type + ' started')
                if not os.path.exists(path + gen_type):
                    os.makedirs(path + gen_type)

                for index in index_list:
                    if not os.path.exists(
                            path + gen_type + '/' + gen_type + '.000' + index + '_sec' + str(index) + '.png'):
                        url = data_path + gen_type + "/" + gen_type + '.000' + index + '.wav'

                        try:
                            scale, sr = self.load(url, self.data_duration)
                            S_scale, Y_scale = self.create(scale)
                            self.save(Y_scale, sr, path + gen_type + '/' + gen_type + '.000' + index + '.png')

                        except:
                            pass

                    #print("Index: ", index, " created.")
                    pbar.update()

    def create(self, scale):
        S_scale = librosa.stft(scale, n_fft=self.frame_size, hop_length=self.hop_size)
        Y_scale = librosa.amplitude_to_db(abs(S_scale))
        return S_scale, Y_scale

    def save(self, y_scale, sr, save_path):
        fig = pylab.gcf()
        fig.set_size_inches(2.4, 2.4)
        pylab.axis('off')  # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge

        librosa.display.specshow(
            y_scale
        )

        pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
        pylab.close()

    def load(self, url, data_duration, offset=0):
        return librosa.load(url, offset=offset, duration=data_duration)
