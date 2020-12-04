from util.spectogram import Spectogram

if __name__ == "__main__":
    spectogram = Spectogram()

    spectogram.create_database(enable_5s=True)
    spectogram.create_database(enable_5s=False)

