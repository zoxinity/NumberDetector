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
| file name   | mad_conv |
|:------------|:---------|
| img1        | 15/20    |
| pen_black   | 9/10     |
| pen_blue    | 9/10     |
| pencil      | 7/10     |
| sample1     | 5/6      |
| sample2     | 5/6      |
| sample3     | 5/6      |
| sample4     | 6/6      |
| thick_black | 9/10     |
| thick_blue  | 9/10     |
| thick_red   | 10/10    |