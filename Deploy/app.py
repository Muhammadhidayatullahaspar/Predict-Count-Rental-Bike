import flask
import pandas as pd
from joblib import dump, load


with open(f'Bike_count_prediction.joblib', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        Tahun = flask.request.form['Tahun']
        Bulan = flask.request.form['Bulan']
        Kecepatan_Angin = flask.request.form['Kecepatan_Angin']
        Hari = flask.request.form['Hari']
        Hari_Kerja = flask.request.form['Hari_Kerja']
        Cuaca = flask.request.form['Cuaca']
        Suhu = flask.request.form['Suhu']
        Kelembaban = flask.request.form['Kelembaban']

        input_variables = pd.DataFrame([[Tahun, Bulan, Kecepatan_Angin, Hari, Hari_Kerja, Cuaca, Suhu, Kelembaban]],
                                       columns=['Tahun', 'Bulan', 'Kecepatan_Angin', 'Hari', 'Hari_Kerja',
                                                'Cuaca', 'Suhu', 'Kelembaban'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', original_input={'Tahun': Tahun, 'Bulan': Bulan, 'Kecepatan_Angin': Kecepatan_Angin, 'Hari': Hari, 'Hari_Kerja': Hari_Kerja, 'Cuaca': Cuaca, 'Suhu': Suhu, 'Kelembaban': Kelembaban},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)
