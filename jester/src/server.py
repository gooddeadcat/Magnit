import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Запуск Jester")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  


@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)

@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']

    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        filename = hashlib.md5(file.filename.encode()).hexdigest() 
        input_path = os.path.join(UPLOAD_FOLDER, filename + file.filename[file.filename.find('.'):])
        file.save(input_path)
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
        
@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
   
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(UPLOAD_FOLDER, file + '.' + type)

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемый тип'
        return answer
    
@app.route("/start", methods=['POST']) 
def start_model():
    
    def get_users_predictions(UID, n, matrix):
        recommended_items = pd.DataFrame(matrix.loc[UID]).dropna()
        recommended_items.columns = ['predicted_rating']
        recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
        recommended_items = recommended_items.head(n)
        return recommended_items.index.tolist()

    def get_hybrid_predictions(UID, n, matrix, inject_data, inject_column, n_inject=3):
        recommended_items = pd.DataFrame(matrix.loc[UID])
        recommended_items.columns = ['predicted_rating']
        injection = inject_data.loc[UID, inject_column][:n_inject]
        recommended_items = recommended_items.drop(injection)
        recommended_items = recommended_items.dropna()
        recommended_items = recommended_items.sort_values('predicted_rating', ascending=False)    
        recommended_items = recommended_items.head(n - len(injection))
        injection.extend(recommended_items.index.tolist())
        return injection
    
    full = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'full_antitest.csv', index_col=0))
    nofact = pd.read_csv(input_path, index_col=0)
    
    test = pd.merge(nofact, full, how='left', on=['UID', 'JID'])
    SVD_matrix = test.pivot_table(index='UID', columns='JID', values='SVD_Prediction')
    SVD_fullmatrix = full.pivot_table(index='UID', columns='JID', values='SVD_Prediction')
    testm = test.copy().groupby('UID', as_index=False).SVD_Prediction.agg({'Top 1': 'max'})
    testm = testm.set_index('UID')
    SVD_recs = []
    for user in testm.index:
      SVD_predictions = get_users_predictions(user, 10, SVD_matrix)
      SVD_recs.append(SVD_predictions)     
    testm['SVD Top 10'] = SVD_recs
    SVD_hybridrecs = []
    for user in testm.index:
      SVD_hybridpredictions = get_hybrid_predictions(UID=user, n=10, matrix=SVD_fullmatrix, inject_data=testm, inject_column='SVD Top 10', n_inject=10)
      SVD_hybridrecs.append(SVD_hybridpredictions)   
    testm['SVD HybridTop 10'] = SVD_hybridrecs

    maskl = testm['SVD HybridTop 10'].apply(lambda x: len(x) != 10)
    users = testm[maskl].index

    for i in users:
      missing = 10 - len(testm.loc[i, 'SVD HybridTop 10'])
      joke = 0
      for m in range(missing):
        joke = testm.loc[i, 'SVD HybridTop 10'][m]
        mask = testm['SVD HybridTop 10'].apply(lambda x: x[0] == joke)
        pair = testm[mask]['SVD HybridTop 10'].apply(lambda x: x[1])
        pair = pd.DataFrame(pair)
        pair = pair[pair['SVD HybridTop 10'].isin(testm.loc[i, 'SVD HybridTop 10']) == False]
        testm.loc[i, 'SVD HybridTop 10'].append(pair['SVD HybridTop 10'].mode().loc[0])
    testm['result'] = testm['Top 1'].apply(lambda x: list([[x]])) + testm['SVD HybridTop 10'].apply(lambda x: list([x]))
    testm['result'].to_frame().to_csv('jester_result.csv')