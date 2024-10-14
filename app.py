from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = joblib.load('../model/XGB3_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

#업로드 버턴
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'no file', 400
    file = request.files['file']
    if file.filename == '':
        return 'no file name', 400
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        data = pd.read_csv(filepath)

        return render_template('analysis.html', tables=[data.to_html(classes='data')], titles=data.columns.values, filename=file.filename)

'''
        predictions = model.predict(data)

        data['quality_label'] = predictions

        result_filepath = os.path.join('result', 'result_' + file.filename)
        data.to_csv(result_filepath, index=False)

        return redirect(url_for('result', filename='result_' + file.filename))
''' 

#분석 버턴
@app.route('/analysis', methods=['POST'])
def analysis():
    #filename = 'Metal_Manufacturing_Dataset_test.csv'   
    filename = request.form['filename']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    data = pd.read_csv(filepath)

    predictions = model.predict(data)

    data['quality_label'] = predictions

    result_filepath = os.path.join('result', 'result_' + filename)
    data.to_csv(result_filepath, index=False)

    return redirect(url_for('result', filename='result_' + filename))    


@app.route('/result/<filename>')
def result(filename):
    result_filepath = os.path.join('result', filename)
    data = pd.read_csv(result_filepath)
    return render_template('result.html', tables=[data.to_html(classes='data')], titles=data.columns.values)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80) 

