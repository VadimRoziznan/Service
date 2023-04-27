'''
Голосовой ассистент "Крендель"

from YT channel PythonHubStudio

python 3.8 и выше.

Распаковать в проект языковую модель vosk

Требуется:
pip install vosk
pip install sounddevice
pip install scikit-learn
pip install pyttsx3
pip install --upgrade wheel

Не обязательно:
pip install requests

#На Linux-ax, скорее всего нужно еще, если ошибка pyttsx3:
#sudo apt update && sudo apt install espeak ffmpeg libespeak1
#sudo apt install espeak
sudo apt-get install espeak-ng
sudo apt-get install alsa-utils
sudo apt update && sudo apt install espeak ffmpeg libespeak1
sudo apt-get install festival speech-tools
sudo apt-get install festvox-ru

Установка eSpeak(NG) в Linux
Подружить «пингвина» с eSpeak, в том числе NG, можно за минуту:
sudo apt-get install espeak-ng python-espeak
pip3 install py-espeak-ng pyttsx3
Дальше загружаем и распаковываем словарь ru_dict с официального сайта:
wget http://espeak.sourceforge.net/data/ru_dict-48.zip
unzip ru_dict-48.zip
Теперь ищем адрес каталога espeak-data (или espeak-ng-data) где-то в /usr/lib/ и перемещаем словарь туда. В моем случае команда на перемещение выглядела так:
sudo mv ru_dict-48 /usr/lib/i386-linux-gnu/espeak-data/ru_dict
Обратите внимание: вместо «i386» у вас в системе может быть «x86_64...» или еще что-то. Если не уверены, воспользуйтесь поиском:
find /usr/lib/ -name "espeak-data"
Готово!

# pip install git+https://github.com/nateshmbhat/pyttsx3
#https://github.com/nateshmbhat/pyttsx3

Для получения справки, спроси у него 'Что ты умеешь Крендель?' или 'справка Крендель'

Ссылки на библиотеки и доп материалы:
sounddevice
https://pypi.org/project/sounddevice/
https://python-sounddevice.readthedocs.io/en/0.4.4/
vosk
https://pypi.org/project/vosk/
https://github.com/alphacep/vosk-api
https://alphacephei.com/vosk/
sklearn
https://pypi.org/project/scikit-learn/
https://scikit-learn.org/stable/
pyttsx3
https://pypi.org/project/pyttsx3/
https://pyttsx3.readthedocs.io/en/latest/
requests
https://pypi.org/project/requests/

'''

from sklearn.feature_extraction.text import CountVectorizer     #pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import sounddevice as sd    #pip install sounddevice
import vosk                 #pip install vosk

import json
import queue

import words
from skills.skills import *
import voice


q = queue.Queue()

model = vosk.Model('../vosk-models/vosk-model-small-ru-0.22')       #голосовую модель vosk нужно поместить в папку с файлами проекта
                                        #https://alphacephei.com/vosk/
                                        #https://alphacephei.com/vosk/models

device = sd.default.device     # <--- по умолчанию
                                #или -> sd.default.device = 1, 3, python -m sounddevice просмотр 
samplerate = int(sd.query_devices(device[0], 'input')['default_samplerate'])  #получаем частоту микрофона


def callback(indata, frames, time, status):
    '''
    Добавляет в очередь семплы из потока.
    вызывается каждый раз при наполнении blocksize
    в sd.RawInputStream'''

    q.put(bytes(indata))


def recognize(data, vectorizer, clf):
    '''
    Анализ распознанной речи
    '''

    #проверяем есть ли имя бота в data, если нет, то return
    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        return

    #удаляем имя бота из текста
    data.replace(list(trg)[0], '')

    #получаем вектор полученного текста
    #сравниваем с вариантами, получая наиболее подходящий ответ
    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]

    #получение имени функции из ответа из data_set
    func_name = answer.split()[0]

    #озвучка ответа из модели data_set
    voice.speaker(answer.replace(func_name, ''))

    #запуск функции из skills
    exec(func_name + '()')


def main():
    '''
    Обучаем матрицу ИИ
    и постоянно слушаем микрофон
    '''

    #Обучение матрицы на data_set модели
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(words.data_set.keys()))
    
    clf = LogisticRegression()
    clf.fit(vectors, list(words.data_set.values()))

    del words.data_set

    #постоянная прослушка микрофона
    with sd.RawInputStream(samplerate=samplerate, blocksize = 16000, device=device[0], dtype='int16',
                                channels=1, callback=callback):

        rec = vosk.KaldiRecognizer(model, samplerate)
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                data = json.loads(rec.Result())['text']
                recognize(data, vectorizer, clf)
            # else:
            #     print(rec.PartialResult())


if __name__ == '__main__':
    main()

