import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *
from surprise import Dataset, Reader, SVD

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
        global input_path
        input_path = os.path.join(UPLOAD_FOLDER, filename + file.filename[file.filename.find('.'):])
        file.save(input_path)
        answer['Сообщение'] = 'Файл успешно загружен! Перейдите на страницу http://127.0.0.1:5000/start'
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
    '''
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path')
    '''
    
    # Проверяем, что указан тип файла
    '''
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')
    '''

    #file_path = os.path.join(UPLOAD_FOLDER, 'full_antitest.csv')
    file_path = output_path
    type = 'csv'

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
    
@app.route("/start", methods=['GET']) 
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
    
    # Считываем трейн и обучаем модель.
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'train_joke_df.csv'))
    
    reader = Reader(rating_scale=(-10, 10))
    data = Dataset.load_from_df(df[['UID', 'JID', 'Rating']], reader)
    
    fulltrainset = data.build_full_trainset()
    antitestset = fulltrainset.build_anti_testset()
    
    params = {'reg_pu': 0.0010398462453200708, 'reg_qi': 0.975481537099458, 'reg_bu': 6.405279534313364e-05, 'reg_bi': 0.008446937909973943, 
          'lr_pu': 0.005037130407846973, 'lr_qi': 0.0001532653071418146, 'lr_bu': 0.0016783726294656905, 'lr_bi': 0.00014842221255242954,
          'n_factors': 1235, 'n_epochs': 44}
    
    algo = SVD(reg_pu=params['reg_pu'], reg_qi=params['reg_qi'], reg_bu=params['reg_bu'], reg_bi=params['reg_bi'], 
           lr_pu=params['lr_pu'], lr_qi=params['lr_qi'], lr_bu=params['lr_bu'], lr_bi=params['lr_bi'],
            n_factors=params['n_factors'], n_epochs=params['n_epochs'])
    
    algo.fit(fulltrainset)
    
    # Делаем предикт на всём антитесте.
    antitest = algo.test(antitestset)
    
    full = pd.DataFrame(antitest)
    full.drop(['r_ui', 'details'], inplace=True, axis=1)
    full.columns = ['UID', 'JID', 'SVD_Prediction']
    
    # Берём тестовый файл из аплуда.
    nofact = pd.read_csv(input_path, index_col=0)
    
    # Добавляем в него предикты. Предполагается, что все они есть в антитесте.
    test = pd.merge(nofact, full, how='left', on=['UID', 'JID'])
    
    SVD_matrix = test.pivot_table(index='UID', columns='JID', values='SVD_Prediction')
    SVD_fullmatrix = full.pivot_table(index='UID', columns='JID', values='SVD_Prediction')
    
    testm = test.copy().groupby('UID', as_index=False).SVD_Prediction.agg({'Top 1': 'max'})
    testm = testm.set_index('UID')
    
    # Делаем топ по взаимодействиям из файла.
    SVD_recs = []
    for user in testm.index:
        SVD_predictions = get_users_predictions(user, 10, SVD_matrix)
        SVD_recs.append(SVD_predictions)     
    testm['SVD Top 10'] = SVD_recs
    
    # Дополняем взаимодействиями из антитеста.
    SVD_hybridrecs = []
    for user in testm.index:
        SVD_hybridpredictions = get_hybrid_predictions(UID=user, n=10, matrix=SVD_fullmatrix, inject_data=testm, inject_column='SVD Top 10', n_inject=10)
        SVD_hybridrecs.append(SVD_hybridpredictions)   
    testm['SVD HybridTop 10'] = SVD_hybridrecs

    # Дополняем топ шутками других пользователей, которым понравились такие же шутки.
    maskl = testm['SVD HybridTop 10'].apply(lambda x: len(x) != 10)
    users = testm[maskl].index

    for i in users:
        missing = (10 - len(testm.loc[i, 'SVD HybridTop 10']))

        # Первая шутка для холодных пользователей:
        if missing == 10:
            freq = pd.DataFrame(full.JID.value_counts())
            med = freq[freq.JID <= freq.JID.quantile(0.6)].index
            med = np.array(med)
            testm.loc[i, 'SVD HybridTop 10'].append(np.random.choice(med))
            missing = 9

        joke = 0
        for m in range(missing):
            yeah = False
            for k in range(m, len(testm.loc[i, 'SVD HybridTop 10'])):
                if yeah == False:
                    joke = testm.loc[i, 'SVD HybridTop 10'][k]
                    for j in range(10):
                        mask1 = testm['SVD HybridTop 10'].apply(lambda x: len(x) >= j + 1)
                        testm1 = testm[mask1]
                        mask2 = testm1['SVD HybridTop 10'].apply(lambda x: x[j] == joke)
                        testm2 = testm1[mask2]
                        if len(testm2) != 0:
                            for l in range(10):
                                if l != j:
                                    mask3 = testm2['SVD HybridTop 10'].apply(lambda x: len(x) >= l + 1)
                                    pair = testm2[mask3]['SVD HybridTop 10'].apply(lambda x: x[l])
                                    pair = pd.DataFrame(pair)
                                    pair = pair[pair['SVD HybridTop 10'].isin(testm.loc[i, 'SVD HybridTop 10']) == False]
                                    if len(pair) != 0:
                                        testm.loc[i, 'SVD HybridTop 10'].append(pair['SVD HybridTop 10'].mode().loc[0])
                                        yeah = True
                                        break
                            break
    
    testm['result'] = testm.index
    testm['result'] = testm.result.apply(lambda x: [x])
    testm['result'] = testm.result + testm[['SVD HybridTop 10', 'Top 1']].apply(lambda x: [{x[0][0]: x[1]}], axis=1) + testm['SVD HybridTop 10'].apply(lambda x: list([x]))
    
    global output_path
    output_path = os.path.join(UPLOAD_FOLDER, 'jester_result.csv')
    testm['result'].to_frame().to_csv(output_path, index=False)
    
    return 'Приехали! Перейдите на страницу http://127.0.0.1:5000/show_data'