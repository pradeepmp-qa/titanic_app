from flask import Flask, render_template, request
import joblib

# initialse the application
app = Flask(__name__)

#load the model
model = joblib.load('SVCmodel_new_clf.pickle')

@app.route('/')
def hello():
    return render_template('form.html')


@app.route('/submit' , methods = ["POST"])
def form_data():

   Pclass = request.form.get('Pclass')
   Age = request.form.get('Age')
   SibSp = request.form.get('SibSp')
   Parch = request.form.get('Parch')
   Fare = request.form.get('Fare')
   Sex_female = request.form.get('Sex_female')
   Sex_male = request.form.get('Sex_male')
   Embarked_Q = request.form.get('Embarked_Q')
   Embarked_S = request.form.get('Embarked_S')

   output = model.predict([[Pclass,Age,SibSp,Parch,Fare,Sex_female,Sex_male,Embarked_Q,Embarked_S]])

   print(output)

   if output[0] == 1:
        out = 'survived'
   else:
        out = 'died'


   return render_template('predict.html' , data = f'Person {out}')

if __name__ == '__main__':
    app.run(debug = True)

