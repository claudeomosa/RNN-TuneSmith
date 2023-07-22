"""
    - Load dataset
    - Filter out songs with non-acceptable note duration (e.g. 1/8, 1/2, 1/4, 1/16)
    - Transpose songs to C major or A minor
    - encode song with music time series representation
        (e.g. ["60", "_", "_" "_"] to represent 1 beat, 60 is the pitch and "_" is how long the note is held)
    - save encoded songs to a single file
"""
import os
import json
from tensorflow import keras
import numpy as np
import music21 as m21

DATASET_PATH = "./usa"
SONGS_DIR = "./preprocessed_data"
UNIVERSAL_DATASET = "songs_dataset"
SEQUENCE_LENGTH = 64
MAPPINGS_PATH = "vocabulary.json"
ACCEPTABLE_DURATIONS = [
    0.25,  # 16th note
    0.5,  # 8th note
    0.75,  # dotted 8th note
    1.0,  # quarter note
    1.5,  # dotted quarter note
    2,  # half note
    3,  # dotted half note or 3 quarter notes
    4,  # whole note
]


def load_dataset(dataset_path):
    """
    loop through all files in the dataset folder and load them with `music21`, a music processing library
    :param dataset_path: path to the dataset
    :return: list of music21 scores
    """
    songs_as_music21_scores = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song_as_music21_score = m21.converter.parse(os.path.join(path, file))
                songs_as_music21_scores.append(song_as_music21_score)

    return songs_as_music21_scores


def has_acceptable_note_durations(song, acceptable_durations):
    """
    check if the song has only acceptable note durations
    :param song: music21 score object
    :param acceptable_durations: list of acceptable note durations
    :return: boolean
    """
    for note in song.flatten().notes:  # flat: return a list of notes and rests in the song
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """
    transpose song to C major or A minor
    :param song: music21 score object
    :return: transposed song: music21 score object
    """
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]  # element at index 4 of the first measure of the first part is the key

    # estimate key using music21 if key is not found
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g. Bmaj -> Cmaj or Dmin -> Amin
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    tranposed_song = song.transpose(interval)

    return tranposed_song


def encode_song(song, time_step=0.25):
    """
    encode a song with music time series representation
    :param song: music21 score object
    :param time_step: float, time resolution to use for encoding
    :return: encoded_song: str (music time series representation of the song)
    """
    encoded_song = []

    for event in song.flatten().notesAndRests:
        # for note
        if isinstance(event, m21.note.Note):
            encoding_symbol = event.pitch.midi  # midi number of the pitch
        # for rests
        elif isinstance(event, m21.note.Rest):
            encoding_symbol = "r"  # r for rest

        # convert the note/rest into time series notation
        step_count = int(event.duration.quarterLength / time_step)
        for step in range(step_count):
            if step == 0:
                encoded_song.append(encoding_symbol)
            else:
                encoded_song.append("_")
    # cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    """
    preprocess the dataset
    :param dataset_path: path to the dataset
    :return: None
    """
    # load the dataset
    songs_as_music21_scores = load_dataset(dataset_path)
    # filter out songs that have non-acceptable note durations
    for i, song in enumerate(songs_as_music21_scores):
        if not has_acceptable_note_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to C major or A minor
        transposed_song = transpose(song)
        # encode songs with music time series representation
        encoded_song = encode_song(transposed_song)
        # save encoded songs to a file
        save_path = os.path.join(SONGS_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def create_universal_dataset(songs_dir, dataset_path, sequence_length):
    """
    create a dataset that can be used for training a model
    :param songs_dir: path to the directory containing all songs
    :param dataset_path: path to the file to save the dataset
    :param sequence_length:
    :return:
    """
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    for path, _, files in os.walk(songs_dir):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]
    #  save the songs to a single dataset file
    with open(dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def load(file_path):
    """
    load encoded song from a file
    :param file_path: path to the file
    :return: file_content
    """
    with open(file_path, "r") as fp:
        file_content = fp.read()

    return file_content


def create_mapping(songs, mappings_path):
    mappings = dict()

    # identify the vocabulary
    vocabulary = list(set(songs.split()))
    # create the mapping
    mappings.update({symbol: index for index, symbol in enumerate(vocabulary)})
    # save the vocabulary to a json file
    with open(mappings_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    """
    converts the songs string to a list of integers, each mapping to a symbol in the vocabulary
    :param songs:
    :return: songs: list of int
    """
    # load vocabulary
    with open(MAPPINGS_PATH, "r") as fp:
        vocabulary_mapping = json.load(fp)
    # cast songs string to a list
    songs = songs.split()
    # map songs to int
    for i, item in enumerate(songs):
        songs[i] = vocabulary_mapping[item]
    # save songs to a file
    return songs


def generate_training_sequences(sequence_length):
    """
    this function will generate training sequences from the dataset
    i.e. creates input and output sequences, input is a sequence of notes and output is the next note in the sequence
    :param sequence_length:
    :return:
    """
    # load songs and map them to int
    int_songs = convert_songs_to_int(load(UNIVERSAL_DATASET))
    # generate the training sequences
    number_of_sequences = len(int_songs) - sequence_length
    inputs = []
    targets = []
    for i in range(number_of_sequences):
        inputs.append(int_songs[i:i + sequence_length])
        targets.append(int_songs[i + sequence_length])
    # one hot encode the training sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    return inputs, targets


def generate_model_ready_dataset():
    preprocess(DATASET_PATH)
    songs = create_universal_dataset(SONGS_DIR, UNIVERSAL_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPINGS_PATH)
    convert_songs_to_int(songs)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    a = 1


generate_model_ready_dataset()
