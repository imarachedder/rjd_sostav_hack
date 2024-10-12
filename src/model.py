import os.path

import torch
import torchaudio
import librosa
import scipy.signal
import numpy as np
import torch.nn as nn

# Глобальная проблема.
# Точность данных сильно храмает, в основном возвращает 4 лейбл на большитсво аудио - не есть хорошо
# Обучение было выполнено с нуля

COMMANDS = {
    "отказ": 0,
    "отмена": 1,
    "подтверждение": 2,
    "начать осаживание": 3,
    "осадить на (количество) вагон": 4,
    "продолжаем осаживание": 5,
    "зарядка тормозной магистрали": 6,
    "вышел из межвагонного пространства": 7,
    "продолжаем роспуск": 8,
    "растянуть автосцепки": 9,
    "протянуть на (количество) вагон": 10,
    "отцепка": 11,
    "назад на башмак": 12,
    "захожу в межвагонное,пространство": 13,
    "остановка": 14,
    "вперед на башмак": 15,
    "сжать автосцепки": 16,
    "назад с башмака": 17,
    "тише": 18,
    "вперед с башмака": 19,
    "прекратить зарядку тормозной магистрали": 20,
    "тормозить": 21,
    "отпустить": 22,
}
_id2label = {
    0: "отказ",
    1: "отмена",
    2: "подтверждение",
    3: "начать осаживание",
    4: "осадить на (количество) вагон",
    5: "продолжаем осаживание",
    6: "зарядка тормозной магистрали",
    7: "вышел из межвагонного пространства",
    8: "продолжаем роспуск",
    9: "растянуть автосцепки",
    10: "протянуть на (количество) вагон",
    11: "отцепка",
    12: "назад на башмак",
    13: "захожу в межвагонное,пространство",
    14: "остановка",
    15: "вперед на башмак",
    16: "сжать автосцепки",
    17: "назад с башмака",
    18: "тише",
    19: "вперед с башмака",
    20: "прекратить зарядку тормозной магистрали",
    21: "тормозить",
    22: "отпустить",
}

# динамические значения для меток (пока что не реализовал)
NUMERIC_WORDS = {
    "ноль": 0, "один": 1, "два": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60, "семьдесят": 70,
    "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200, "триста": 300
}

class SpeechRecognitionModel(nn.Module):
    """
    Моделька
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(SpeechRecognitionModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

def extract_numeric_attribute(text):
    words = text.lower().split()
    for word in words:
        if word in NUMERIC_WORDS:
            return NUMERIC_WORDS[word]
    return -1

def noise_reduction(waveform, sample_rate):
    """
    подавляем шум на аудио

    Проблема: Пока что не разобрался как правильно обработать аудио для избавления от шума.
    Args:
        waveform:
        sample_rate:
    Returns:

    """
    freqs, times, Sxx = scipy.signal.spectrogram(waveform.numpy(), fs=sample_rate)

    threshold = np.mean(Sxx) * 1.5
    Sxx[Sxx < threshold] = 0

    _, cleaned_waveform = scipy.signal.istft(Sxx)

    return torch.tensor(cleaned_waveform, dtype=torch.float32)


def trim_silence(waveform, sample_rate):
    """
    урезка тишины
    Args:
        waveform:
        sample_rate:

    Returns:

    """
    trimmed_waveform, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
    return torch.tensor(trimmed_waveform, dtype=torch.float32)

# def preprocess_audio(audio_filepath, n_mels=128, output_filepath = r"C:\Users\admin\Desktop\rjd_sostav\outputs\output.mp3"):
#     waveform, sample_rate = torchaudio.load(audio_filepath)
#     cleaned_waveform = noise_reduction(waveform[0], sample_rate)
#     trimmed_waveform = trim_silence(cleaned_waveform, sample_rate)
#     if output_filepath:
#         torchaudio.save(output_filepath, trimmed_waveform.unsqueeze(0), sample_rate)
#     mel_spectrogram = librosa.feature.melspectrogram(y=trimmed_waveform.numpy(), sr=sample_rate, n_mels=n_mels)
#     log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
#     tensor_audio = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
#
#     return tensor_audio.T  # Транспонируем

def preprocess_audio(audio_filepath, n_mels=128):
    """
    Предварительная обработка аудио
    Args:
        audio_filepath:
        n_mels:
        output_filepath:

    Returns:

    """
    waveform, sample_rate = torchaudio.load(audio_filepath)

    mel_spectrogram = librosa.feature.melspectrogram(y=waveform[0].numpy(), sr=sample_rate, n_mels=n_mels)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    tensor_audio = torch.tensor(log_mel_spectrogram, dtype=torch.float32)
    return tensor_audio.T  # Транспонируем для правильной подачи в модель


def predict_command(model, audio_filepath):
    """
    Предиктивка
    Args:
        model:
        audio_filepath:

    Returns:

    """
    # мел-спектрограмма
    audio_tensor = preprocess_audio(audio_filepath)
    print(audio_tensor)

    print(f"размер перед разжатием: {audio_tensor.shape}")

    audio_tensor = audio_tensor.unsqueeze(0)  # размер батча

    print(f"размер ввода после разжатия: {audio_tensor.shape}")

    model.eval()
    with torch.no_grad():
        output = model(audio_tensor)

    predicted_label = torch.argmax(output, dim=1).item()
    print("Предиктивная метка", predicted_label)
    global _id2label
    predicted_text = _id2label[predicted_label]

    return predicted_label, predicted_text


def process_audio_command(audio_filepath, model):
    predicted_label, predicted_text = predict_command(model, audio_filepath)
    print(predicted_label, predicted_text)
    attribute_value = extract_numeric_attribute(predicted_text)
    result = {
        "audio_filepath": audio_filepath,
        "text": predicted_text,
        "label": predicted_label,
        "attribute": attribute_value
    }
    return result


def load_model(filepath, input_size=128, hidden_size=64, output_size=21):
    """
    Функция загрузки модели
    Args:
        filepath:
        input_size:
        hidden_size:
        output_size:

    Returns:

    """
    model = SpeechRecognitionModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    print(f"модельку загрузил{filepath}")
    return model


print("Начал процесс предикта")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'speech_recognition_model.pth')

loaded_model = load_model(model_path)

audio_file_path = os.path.join(current_dir, '..', 'luga', '02_11_2023', '2023_11_02__10_33_44.wav')

result = process_audio_command(audio_file_path, loaded_model)
print(result)
