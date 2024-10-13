import os.path
import time
import torch
import torchaudio
import torch.nn as nn
import noisereduce as nr
import json
import psutil
import jiwer
from sklearn.metrics import f1_score

# Глобальная проблема.
# Точность данных сильно храмает, в основном возвращает 4 лейбл на большинство аудио - не есть хорошо
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
    """
    функция должны была переводить числовое значение
    из транскрибации где есть значения семь восеь и тд в int для передачи в attribute
    пока что это проблема, не смог реализовать.
    Моделька возвращает класс команды, а не наоборот. в общем подумаем
    Args:
        text:

    Returns:

    """
    words = text.lower().split()
    for word in words:
        if word in NUMERIC_WORDS:
            return NUMERIC_WORDS[word]
    return -1


def denoise_audio(input_audio_path, output_audio_path):
    """
    Очищаем от шума и сохраняем пока что для проверки в папку outupts
    Args:
        input_audio_path:
        output_audio_path:

    Returns:

    """
    waveform, sample_rate = torchaudio.load(input_audio_path)

    # переводим в моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # преобразуем тензор в массив NumPy
    audio_data = waveform.numpy().flatten()

    # удаляем шум
    denoised_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    denoised_waveform = torch.tensor(denoised_audio, dtype=torch.float32)

    torchaudio.save(output_audio_path, torch.tensor(denoised_audio).unsqueeze(0), sample_rate)
    return denoised_waveform, sample_rate


def preprocess_audio(audio_filepath, n_mels=128, n_fft=512):
    """
    Функция предвариательной обработки аудио
    Args:
        audio_filepath:
        n_mels:

    Returns:

    """
    out_path = os.path.join(current_dir, '..', 'outputs', audio_filepath.split("\\")[-1])
    denoised_waveform, sample_rate = denoise_audio(audio_filepath,
                                      output_audio_path=out_path)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft
    )
    mel_spectrogram = mel_spectrogram_transform(denoised_waveform.unsqueeze(0))  # Добавляем размер канала

    return mel_spectrogram.squeeze(0).transpose(0, 1)


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
    # print(audio_tensor)

    # print(f"размер перед разжатием: {audio_tensor.shape}")

    audio_tensor = audio_tensor.unsqueeze(0)

    # print(f"размер ввода после разжатия: {audio_tensor.shape}")

    model.eval()
    with torch.no_grad():
        output = model(audio_tensor)

    predicted_label = torch.argmax(output, dim=1).item()
    global _id2label
    predicted_text = _id2label[predicted_label]

    return predicted_label, predicted_text


def process_audio_command(audio_filepath, model):
    predicted_label, predicted_text = predict_command(model, audio_filepath)
    # print(predicted_label, predicted_text)
    attribute_value = extract_numeric_attribute(predicted_text)
    result = {
        "audio_filepath": audio_filepath.split("\\")[-1],
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
    # print(f"модельку загрузил{filepath}")
    return model


if __name__ == "__main__":
    start = time.time()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'speech_recognition_model2.pth')

    loaded_model = load_model(model_path)
    y_true = []
    y_pred = []
    hypothesis = []
    ground_truth = []
    with open('luga.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        directory = item['audio_filepath'].split("/")
        # print(directory)
        audio_file_path = os.path.join(current_dir, '..', 'luga', directory[0], directory[1])
        ground_truth.append(item['text'])
        y_true.append(item['label'])
        output = os.path.join(current_dir, '..', 'commands')
        result = process_audio_command(audio_file_path, loaded_model)
        with open(os.path.join(output, "commands.json"), 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        y_pred.append(result['label'])
        hypothesis.append(result['text'])


    print(f"Время выполнения: {(time.time() - start) * 1000} мс" )

    # пАмять
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss
    print(f"Используемая память: {memory_usage / (1024 ** 2):.2f} MB")
    # WER

    wer = jiwer.wer(ground_truth, hypothesis)
    print(f"Word Error Rate (WER): {wer:.2f}")

    # F1 Score
    # y_true = [6, 7, 9, 4]
    # y_pred = [result["label"]]

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1 Score: {f1:.2f}")

