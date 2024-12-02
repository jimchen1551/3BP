from flask import Flask, request, url_for, redirect, render_template, send_file, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import joblib
import pandas as pd
import flask_login
import os

app = Flask(__name__)
uri = os.getenv("DATABASE_URL")
if uri:
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
else:
    uri = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'students.sqlite3')
app.config['SQLALCHEMY_DATABASE_URI'] = uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = "hypertension prediction"
db = SQLAlchemy(app)

login_manager = flask_login.LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

model = joblib.load('rf_WCH.joblib')
model_ = joblib.load('rf_MH.joblib')
best_threshold = 0.14596254
best_threshold_ = 0.05963105

users = {'yohai': {'password': 'predicthypertension'}}

class User(flask_login.UserMixin):
    pass

@login_manager.user_loader
def user_loader(userid):
    if userid not in users:
        return
    user = User()
    user.id = userid
    return user

@login_manager.request_loader
def request_loader(request):
    userid = request.form.get('userid')
    if userid not in users:
        return
    user = User()
    user.id = userid
    return user

@app.route('/logout')
def logout():
    flask_login.logout_user()
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    userid = request.form['userid']
    if userid in users and request.form['password'] == users[userid]['password']:
        user = User()
        user.id = userid
        flask_login.login_user(user)
        return redirect(url_for('show_result'))
    return render_template('login.html')

class Databases(db.Model):
    id = db.Column('case_id', db.Integer, primary_key=True)
    id_ = db.Column(db.Float)
    age = db.Column(db.Float)
    gender = db.Column(db.String(10))
    height = db.Column(db.Float)
    bw = db.Column(db.Float)
    bmi = db.Column(db.Float)
    sbp1 = db.Column(db.Float)
    dbp1 = db.Column(db.Float)
    hr1 = db.Column(db.Float)
    sbp2 = db.Column(db.Float)
    dbp2 = db.Column(db.Float)
    hr2 = db.Column(db.Float)
    sbp3 = db.Column(db.Float)
    dbp3 = db.Column(db.Float)
    hr3 = db.Column(db.Float)
    prob = db.Column(db.Float)
    label = db.Column(db.String(50))
    prob_ = db.Column(db.Float)
    label_ = db.Column(db.String(50))
    created_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def __init__(self, id_, age, gender, height, bw, bmi, sbp1, dbp1, hr1, sbp2, dbp2, hr2, sbp3, dbp3, hr3, prob, label, prob_, label_):
        self.id_ = id_
        self.age = age
        self.gender = gender
        self.height = height
        self.bw = bw
        self.bmi = bmi
        self.sbp1 = sbp1
        self.dbp1 = dbp1
        self.hr1 = hr1
        self.sbp2 = sbp2
        self.dbp2 = dbp2
        self.hr2 = hr2
        self.sbp3 = sbp3
        self.dbp3 = dbp3
        self.hr3 = hr3
        self.prob = prob
        self.label = label
        self.prob_ = prob_
        self.label_ = label_
        self.created_date = datetime.now() + timedelta(hours=8)

def to_dict(row):
    if row is None:
        return None
    rtn_dict = dict()
    keys = row.__table__.columns.keys()
    for key in keys:
        if key != 'case_id':
           rtn_dict[key] = getattr(row, key)
    return rtn_dict

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/show_result')
@flask_login.login_required
def show_result():
    return render_template('result.html', databases=Databases.query.all())

@app.errorhandler(500)
def internal_server_error(e):
    flash('!! Internal server ERROR !!')
    return redirect(url_for('home'))

@app.errorhandler(405)
def method_not_allowed(e):
    flash('!! Method not allowed ERROR !!')
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    msg = ''
    
    # Get form values
    id_ = int(request.form["id"])
    age = int(request.form["age"])
    gender = 1 if request.form["gender"]=="male" else 0
    print(request.form["gender"])
    height = float(request.form["height"])
    bw = float(request.form["bw"])
    sbp1 = float(request.form["sbp1"])
    dbp1 = float(request.form["dbp1"])
    hr1 = float(request.form["hr1"])
    sbp2 = float(request.form["sbp2"])
    dbp2 = float(request.form["dbp2"])
    hr2 = float(request.form["hr2"])
    sbp3 = float(request.form["sbp3"])
    dbp3 = float(request.form["dbp3"])
    hr3 = float(request.form["hr3"])

    # Calculate BMI
    bmi = bw / ((height / 100) ** 2)

    # Validate SBP, DBP ranges
    if (sbp1 < 10 or sbp1 > 300) or (sbp2 < 10 or sbp2 > 300) or (sbp3 < 10 or sbp3 > 300):
        msg = "SBP out of range (10-300)!"
        return render_template('show.html', info=msg)
    if (dbp1 < 10 or dbp1 > 300) or (dbp2 < 10 or dbp2 > 300) or (dbp3 < 10 or dbp3 > 300):
        msg = "DBP out of range (10-300)!"
        return render_template('show.html', info=msg)

    # Prepare DataFrame for prediction
    df = pd.DataFrame(data={
        'Age': [age], 'Gender': [gender], 'Height': [height], 'BW': [bw], 'BMI': [bmi],
        'SBP1': [sbp1], 'DBP1': [dbp1], 'HR1': [hr1],
        'SBP2': [sbp2], 'DBP2': [dbp2], 'HR2': [hr2],
        'SBP3': [sbp3], 'DBP3': [dbp3], 'HR3': [hr3]
    })

    # Prepare for prediction
    prediction = model.predict(df)[1]
    output = 0 if (sbp2+sbp3)/2<130 and (dbp2+dbp3)/2<80 else prediction.round(4)
    pred_class = "WCH" if output > best_threshold else "Non-WCH"
    
    prediction_ = model_.predict(df)[1]
    output_ = 0 if (sbp2+sbp3)/2 >= 130 or (dbp2+dbp3)/2>=80 else prediction_.round(4)
    pred_class_ = "MH" if output_ > best_threshold_ else "Non-MH"

    # Store record in the database
    record = Databases(id_, age, gender, height, bw, bmi, sbp1, dbp1, hr1, sbp2, dbp2, hr2, sbp3, dbp3, hr3, output, pred_class, output_, pred_class_)
    db.session.add(record)
    db.session.commit()

    msg = f"WCH: {str(output)}/{pred_class} | MH: {str(output_)}/{pred_class_}"
    return render_template('show.html', info=msg)

@app.route('/delete/<mid>')
@flask_login.login_required
def delete_record(mid):
    record = Databases.query.filter_by(id=mid).first()
    if record:
        db.session.delete(record)
        db.session.commit()
    return redirect(url_for('show_result'))

@app.route('/delete_all')
@flask_login.login_required
def delete_all():
    all_records = Databases.query.all()
    if all_records:
        for s in all_records:
           db.session.delete(s)
        db.session.commit()
    return redirect(url_for('show_result'))

@app.route('/excel', methods=['GET', 'POST'])
@flask_login.login_required
def exportexcel():
    data = Databases.query.all()
    data_list = [to_dict(item) for item in data]
    df = pd.DataFrame(data_list)
    df['created_date'] = df['created_date'].dt.tz_localize(None)
    filename = "dataset.xlsx"
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, sheet_name='Datasets')
    writer.save()
    return send_file(filename)

@app.route('/insert_test_data')
def insert_test_data():
    # Insert a sample record into the database
    test_record = Databases(
        id_=1, age=30, gender='Male', height=175, bw=70, bmi=22.9,
        sbp1=120, dbp1=80, hr1=70, sbp2=122, dbp2=82, hr2=72,
        sbp3=124, dbp3=84, hr3=74, prob=0.9, label='WCH', prob_=0.8, label_='MH'
    )
    db.session.add(test_record)
    db.session.commit()
    return "Test data inserted!"

if __name__ == "__main__":
    app.app_context().push()
    db.create_all()
    app.run()
