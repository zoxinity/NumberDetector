# Установка зависимостей #

pip install -r requirements.txt

# Запуск программы #
## Синтаксис ##
    python main.py [-t | --train] [-m | --model <path_to_model>] [<img1_path> <img2_path> ...]
## Обучение ##
    python main.py -t
## Распознавание ##
    python main.py
    python main.py resources/img.jpg
    python main.py -m recources/knn_clf.sav
    python main.py -m recources/knn_clf.sav resources/img.jpg
    python main.py resources/sample1.jpg resources/sample2.jpg