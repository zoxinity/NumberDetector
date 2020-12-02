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
    python main.py -m resources/knn_clf.sav
    python main.py -m resources/knn_clf.sav resources/img.jpg
    python main.py resources/sample1.jpg resources/sample2.jpg
# Результаты работы программы #
| file name     |  correctly guessed |
|:--------------|:-------------------|
| img           | 9/20               |
| pen_black     | 6/10               |
| pen_blue      | 6/10               |
| pencil        | 6/10               |
| sample1       | 4/5                |
| sample2       | 4/5                |
| sample3       | 4/5                |
| sample4       | 5/5                |
| thick_black   | 7/10               |
| thick_blue    | 8/10               |
| thick_red     | 8/10               |